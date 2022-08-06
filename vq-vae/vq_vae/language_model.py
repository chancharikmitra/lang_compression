import math
from typing import Tuple
import pdb

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

from transformers import BertModel

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerModel(nn.Module):

    def __init__(self, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, 30000)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


def generate_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

class AutoEncoderLM(nn.Module): 

    def __init__(self, autoencoder, d_model, nhead, d_hid, n_layers, dropout):
        super().__init__()

        self.autoencoder = autoencoder
        self.d_model = d_model
        self.nhead = nhead
        self.d_hid = d_hid, 
        self.n_layers = n_layers
        self.dropout = dropout

        # BERT for embedding words. 
        self.bert_embedding_matrix = BertModel.from_pretrained('bert-base-cased').embeddings.word_embeddings
     
        for param in self.bert_embedding_matrix.parameters():
            param.requires_grad = False

        # Linear layer to map image tokens to dimensionality of BERT word embeddings.  
        self.linear = nn.Linear(d_model, 768)
        
        # Transformer language model. 
        self.transformer = TransformerModel(768, nhead, d_hid, n_layers, dropout)

        # Encoder to reduce dimensionality of latent code before feeding to language model. 
        if self.autoencoder.model_type == 'vqvae': 
            self.img_encoder = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            )

        # Initilize loss variable, used to keep track,
        self.c_loss = 0

    def forward(self, x): 
        """
        Wrapper method for autoencoder's forward. 
        """
        return self.autoencoder(x)

    def loss_function(self, x, *kargs):
        """
        Wrapper method for autoencoder's loss_function.
        """
        return self.autoencoder.loss_function(x, *kargs)

    def latest_losses(self): 
        """
        Wrapper method for autoencoder's latest_losses
        """
        loss_dict = self.autoencoder.latest_losses()
        loss_dict['caption'] = self.c_loss

        return loss_dict

    def print_atom_hist(self, argmin):
        """
        Wrapper method for autonceoder's print_atom_hist
        """
        self.autoencoder.print_atom_hist(argmin)

    def caption_loss(self, captions, outputs):

        # Embed words using BERT.     
        lang_tokens = self.bert_embedding_matrix(captions)

        # TODO move to autoencoder class level as a single shared method to avoid clutter. 
        # Format visual tokens for transfomer.
        if self.autoencoder.model_type == 'vqvae': 
            vis_tokens = self.img_encoder(outputs[2])
            vis_tokens = vis_tokens.view(vis_tokens.shape[0], vis_tokens.shape[1], -1)
            vis_tokens = torch.transpose(vis_tokens, 1, 2)
      
        elif self.autoencoder.model_type == 'vae' or self.autoencoder.model_type == 'ae': 
            vis_tokens = outputs[-1].unsqueeze(1)

        # Project visual tokens to same dimentionality as BERT embeddings. 
        vis_tokens = self.linear(vis_tokens)

        # Prepare all tokens for transformer.
        all_tokens = torch.cat((vis_tokens, lang_tokens), 1)
        mask = generate_mask(all_tokens.shape[1])
        mask[:,:vis_tokens.shape[1]] = 0 # Visual tokens can all attend to each other. 

        # Pass through transformers.
        preds = self.transformer(all_tokens, mask.cuda())

        # Make target for captioning task.
        targets = captions[:,1:captions.shape[1]+1]
        dummy_targets = torch.zeros((captions.shape[0], vis_tokens.shape[1])).long()
        all_targets = torch.cat((dummy_targets.cuda(), targets, torch.zeros(captions.shape[0], 1).long().cuda()), 1) # Last one is for shifted target. 

        # Captioning loss. 
        preds = preds.view(preds.shape[0] * preds.shape[1], -1)
        all_targets = all_targets.view(-1)
        self.c_loss = F.cross_entropy(preds, all_targets, ignore_index=0)

        return self.c_loss
