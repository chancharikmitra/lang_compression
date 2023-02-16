import pdb
import argparse
import os
from tqdm.auto import tqdm
import h5py
import numpy as np
import re
import json
from multiprocessing import Pool 

# TODO Seems like EOS thing at beginning has already been added. 
# TODO Check to see if using the tokenizer fast argument actually changes the tokenizations? 
# TODO Check if EOS embedding differs across sentences? If so, can try using!
# TODO Can also just use the mean of each caption. Maybe we can make a dataset like that. 

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from torch.utils.data import Dataset, DataLoader

class CapDataset(Dataset):

    def __init__(self, captions, tokenizer):
        self.captions = captions
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        
        tokenization = self.tokenizer(self.captions[idx])
        tokenization['input_ids'] = torch.tensor(tokenization['input_ids'])
        tokenization['attention_mask'] = torch.tensor(tokenization['attention_mask'])

        return tokenization

    @staticmethod 
    def collate_fn(batch):
        input_ids = [element['input_ids'] for element in batch]
        attention_masks = [element['attention_mask'] for element in batch]

        return_batch = {
            'input_ids': pad_sequence(input_ids, batch_first=True),
            'attention_masks': pad_sequence(attention_masks, batch_first=True)
        }

        return return_batch

def clean_number(w):    
    new_w = re.sub('[0-9]{1,}([,.]?[0-9]*)*', 'N', w)
    return new_w
    
def collect_all_tokens(args): 

    # Sentence tokenizer and feature extraction pipeline. 
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    extractor = pipeline('feature-extraction', model=args.model_name, device=0)

    # Format path to out file. 
    if 'train' in args.in_file: 
        split = 'train'
    elif 'val' in args.in_file: 
        split = 'val'
    elif 'test' in args.in_file:
        split = 'test'
    else:
        raise 

    model_basename = os.path.basename(args.model_name)
    out_path = os.path.join(args.out_dir, '{}_{}.hdf5'.format(split, model_basename))

    # Load all captions from input file. 
    captions = [json.loads(line)[0].strip() for line in open(args.in_file, 'r')]
    captions = [' '.join([clean_number(w) for w in caption.strip().lower().split()]) for caption in captions]

    # Tokenize all captions. 
    print('Tokenizing captions')
    tokenizations = [tokenizer.encode(caption) for caption in tqdm(captions)]

    # Get maximal caption length (in tokens). 
    print('Computing lengths.')
    lens = [len(tokenization) for tokenization in tqdm(tokenizations)]
    max_len = max(lens)
    dataset_sz = len(lens)

    # Get maximal number of words. 
    print('Computing word counts.')
    word_counts = [len(caption.strip().lower().split()) for caption in tqdm(captions)]
    max_words = max(word_counts)

    # Free up space. 
    del lens
    del word_counts

    # Will hold dataset. 
    features = np.zeros((dataset_sz, max_len, args.hdim)) # LM embeddings. 
    lens = np.zeros((dataset_sz, max_words))        # Will hold number of tokens for each word. 

    # Extract features.
    print('Extracting features...')

    for cap_id in tqdm(range(dataset_sz)): 
        
        # Get caption and words. 
        words = [clean_number(w) for w in captions[cap_id].strip().lower().split()]
        caption = ' '.join(words)

        for word_id in range(len(words)):

            # Tokenize word to see how many tokens it's composed of. 
            word = words[word_id]

            # Different cases for first and inside words. 
            if word_id == 0 or 'bert-' in model_basename or 'roberta' in model_basename: 
                num_toks = len(tokenizer.encode(word, add_special_tokens=False))
            else:
                num_toks = len(tokenizer.encode(word, add_special_tokens=False, add_prefix_space=True))

            # Store number of tokens for word.  
            lens[cap_id][word_id] = num_toks

        # Now extract token embeddings. 
        embeddings = np.asarray(extractor(caption)[0])
        features[cap_id][:embeddings.shape[0]] = embeddings

        # Make sure no mistakes were made, account for special start token. 
        total_toks = int(sum(lens[cap_id]))

        """ # TODO Roberta tokenization is off somehow in terms of word alignment. 
        # Compute expected number of tokens given LLM. 
        if 'opt' in model_basename: 
            expected_toks = len(embeddings) - 1
        elif 'bert-' in model_basename: 
            expected_toks = len(embeddings) - 2
        elif 'roberta' in model_basename:
            expected_toks = len(embeddings)

        try: 
            assert total_toks == expected_toks
        except: 
            pdb.set_trace()
        """

    # Write dataset file. 
    print('Writing dataset to {}'.format(out_path))
    hfile = h5py.File(out_path, 'w')
    hfile.create_dataset('features', features.shape, dtype='f', data=features)
    hfile.create_dataset('lens', lens.shape, dtype='i', data=lens)

def collect_file_means(args, in_file, out_path, model, tokenizer):

    # Load all captions from input file. 
    print('Loading captions...')
    captions = [json.loads(line)[0].strip() for line in open(in_file, 'r')]
    captions = [' '.join([clean_number(w) for w in caption.strip().lower().split()]) for caption in captions]
    dataset_sz = len(captions)

    # Dataloder for processing captions. 
    dataset = CapDataset(captions, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0, collate_fn=CapDataset.collate_fn)

    # Get sizes for all layers (and concatenation) by running a test sentence.
    test_outs = model(torch.tensor([[0]]).cuda(), torch.tensor([[0]]).cuda())['hidden_states']
    dims = [test_outs[-i].size(-1) for i in range(1, args.n_layers + 1)]
    dims.append(sum(dims))
    print('Model feature dimensions: {}'.format(dims))

    # Will store all means. 
    print('Allocating memory for dataset matrices...')
    means = [np.zeros((dataset_sz, dim)).astype(np.float32) for dim in dims]

    # Extract features.
    print('Extracting features...')

    # Used to keep track of where in dataset to store features from each batch.  
    idx_pointer = 0

    for batch in tqdm(dataloader):

        # Extract features from model. 
        input_ids, attention_masks = batch['input_ids'].cuda(), batch['attention_masks'].cuda()

        with torch.no_grad():

            # Get output of all layers. 
            outs = model(input_ids, attention_masks)['hidden_states']
            
            # Compute means for last n hidden states. 
            attention_masks = attention_masks.unsqueeze(-1)
            means_buff = []

            for i in range(1, args.n_layers + 1):
                
                # Hidden state and mask for computing mean. 
                hidden_state = outs[-i]
                mask = attention_masks.expand_as(hidden_state)
                
                # Compute mean for layer.
                mean = (hidden_state * mask).sum(dim=1) / mask.sum(dim=1).float() 
                means_buff.append(mean)

            # Also store concatenation of all layers' means.
            means_buff.append(torch.cat(means_buff, dim=1)) 

            # Convert to numpy and store in dataset. 
            bz = means_buff[0].size(0)

            for i in range(len(means_buff)):
                means[i][idx_pointer : idx_pointer + bz] = means_buff[i].cpu().numpy()

            # Increment pointer to next batch. 
            idx_pointer = idx_pointer + bz 

    # Write dataset file. 
    print('Writing dataset to {}'.format(out_path))
    hfile = h5py.File(out_path, 'w')

    # Store all single layer means. 
    for i in range(1, args.n_layers + 1):
        
        # Name based on distance from output layer. 
        hfile.create_dataset(str(-i), means[i-1].shape, dtype='f', data=means[i-1])

    # Finally create a dataset for the concatenations and close. 
    hfile.create_dataset('concat', means[-1].shape, dtype='f', data=means[-1])
    hfile.close()

# Collect for all dataset splits. 
def collect_means(args): 

    # Sentence tokenizer and feature extraction pipeline. 
    print('Loading {} and its tokenizer...'.format(args.model_name))
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, output_hidden_states=True).cuda()
    model_basename = os.path.basename(args.model_name)

    for split in ('test', 'val', 'train'):
        
        print('\nExtracting features for {} set...'.format(split))

        # Paths for input captions and output feature files. 
        in_file = os.path.join(args.in_dir, '{}_caps.json'.format(split))
        out_path = os.path.join(args.out_dir, '{}_{}_means.hdf5'.format(split, model_basename))

        # Compute features for split. 
        collect_file_means(args, in_file, out_path, model, tokenizer)

if __name__ == '__main__':
    
    arg_parser = argparse.ArgumentParser(description='Extract text corpus features using HuggingFace LM')

    arg_parser.add_argument('model_name', type=str, help='HuggingFace model name to use.')
    arg_parser.add_argument('in_dir', type=str, help='Path to directory where caption json files are stored.')
    arg_parser.add_argument('out_dir', type=str, help='Path to directory in which to store extracted features (.hdf5).')
    arg_parser.add_argument('--hdim', type=int, default=768, help='Embedding dimensions for LLM')
    arg_parser.add_argument('--embedding_type', type=str, default='all', choices=['all', 'mean'],
        help='Type of embeddings to collect (all tokens or means)')
    arg_parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing, only valid for means collection.')
    arg_parser.add_argument('--num_workers', type=int, default=4, help='Number of workers to use in dataloader.')
    arg_parser.add_argument('--n_layers', type=int, default=4, help='Number of layers to extract features from (if taking means).')

    args = arg_parser.parse_args()

    if args.embedding_type == 'all':
        collect_all_tokens(args)
    elif args.embedding_type == 'mean':
        collect_means(args)