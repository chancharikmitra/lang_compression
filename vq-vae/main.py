import os
import sys
import time
import logging
import argparse
import pdb
from tqdm import tqdm

import torch.utils.data
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
from transformers import BertTokenizer

from vq_vae.util import setup_logging_from_args
from vq_vae.auto_encoder import *
from vq_vae.language_model import AutoEncoderLM
from inception_score.inception_score import inception_score

models = {
    'custom': {'vqvae': VQ_CVAE,
               'vqvae2': VQ_CVAE2},
    'imagenet': {'vqvae': VQ_CVAE,
                 'vqvae2': VQ_CVAE2},
    'cifar10': {'vae': CVAE,
                'vqvae': VQ_CVAE,
                'vqvae2': VQ_CVAE2},
    'mnist': {'vae': VAE,
              'vqvae': VQ_CVAE},
    'coco': {'vae': VAE, 
              'vqvae': VQ_CVAE, 
              'vqvae2': VQ_CVAE2,
              'ae': AutoEncoder},
}
datasets_classes = {
    'custom': datasets.ImageFolder,
    'coco': datasets.CocoCaptions,
    'imagenet': datasets.ImageFolder,
    'cifar10': datasets.CIFAR10,
    'mnist': datasets.MNIST
}
dataset_train_args = {
    'custom': {},
    'imagenet': {},
    'coco': {},
    'cifar10': {'train': True, 'download': True},
    'mnist': {'train': True, 'download': True},
}
dataset_test_args = {
    'custom': {},
    'imagenet': {},
    'coco': {},
    'cifar10': {'train': False, 'download': True},
    'mnist': {'train': False, 'download': True},
}
dataset_n_channels = {
    'custom': 3,
    'imagenet': 3,
    'cifar10': 3,
    'mnist': 1,
    'coco': 3,
}

dataset_transforms = {
    'custom': transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    'imagenet': transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    'cifar10': transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    'mnist': transforms.ToTensor(),
    'coco': transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]), 
}
default_hyperparams = {
    'custom': {'lr': 2e-4, 'k': 512, 'hidden': 128},
    'imagenet': {'lr': 2e-4, 'k': 512, 'hidden': 128},
    'cifar10': {'lr': 2e-4, 'k': 10, 'hidden': 256},
    'mnist': {'lr': 1e-4, 'k': 10, 'hidden': 64},
    'coco': {'lr': 1e-4, 'k': 512, 'hidden': 64},
}

# Tokenization function for text using BERT. 
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

def main(args):
    parser = argparse.ArgumentParser(description='Variational AutoEncoders')

    model_parser = parser.add_argument_group('Model Parameters')
    model_parser.add_argument('--model', default='vae', choices=['vae', 'vqvae', 'ae'],
                              help='autoencoder variant to use: vae | vqvae | ae')
    model_parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                              help='input batch size for training (default: 128)')
    model_parser.add_argument('--hidden', type=int, metavar='N',
                              help='number of hidden channels')
    model_parser.add_argument('--k', '--dict-size', type=int, dest='k', metavar='K',
                              help='number of atoms in dictionary')
    model_parser.add_argument('--lr', type=float, default=None,
                              help='learning rate')
    model_parser.add_argument('--vq_coef', type=float, default=None,
                              help='vq coefficient in loss')
    model_parser.add_argument('--commit_coef', type=float, default=None,
                              help='commitment coefficient in loss')
    model_parser.add_argument('--kl_coef', type=float, default=None,
                              help='kl-divergence coefficient in loss')

    # Captioning model auxiliary task. 
    model_parser.add_argument('--use_language', action='store_true', default=False, 
            help='use language captioning auxiliary task.')

    training_parser = parser.add_argument_group('Training Parameters')
    training_parser.add_argument('--dataset', default='cifar10', choices=['mnist', 'cifar10', 'imagenet', 'coco',
                                                                          'custom'],
                                 help='dataset to use: mnist | cifar10 | imagenet | coco | custom')
    training_parser.add_argument('--dataset_dir_name', default='',
                                 help='name of the dir containing the dataset if dataset == custom')
    training_parser.add_argument('--data-dir', default='/media/ssd/Datasets',
                                 help='directory containing the dataset')
    training_parser.add_argument('--epochs', type=int, default=5, metavar='N',
                                 help='number of epochs to train (default: 10)')
    training_parser.add_argument('--max-epoch-samples', type=int, default=50000,
                                 help='max num of samples per epoch')
    training_parser.add_argument('--no-cuda', action='store_true', default=False,
                                 help='enables CUDA training')
    training_parser.add_argument('--seed', type=int, default=1, metavar='S',
                                 help='random seed (default: 1)')
    training_parser.add_argument('--gpus', default='0',
                                 help='gpus used for training - e.g 0,1,3')

    logging_parser = parser.add_argument_group('Logging Parameters')
    logging_parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                                help='how many batches to wait before logging training status')
    logging_parser.add_argument('--results-dir', metavar='RESULTS_DIR', default='./results',
                                help='results dir')
    logging_parser.add_argument('--save-name', default='',
                                help='saved folder')
    logging_parser.add_argument('--data-format', default='json',
                                help='in which format to save the data')


    args = parser.parse_args(args)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    dataset_dir_name = args.dataset if args.dataset != 'custom' else args.dataset_dir_name

    lr = args.lr or default_hyperparams[args.dataset]['lr']
    k = args.k or default_hyperparams[args.dataset]['k']
    hidden = args.hidden or default_hyperparams[args.dataset]['hidden']
    num_channels = dataset_n_channels[args.dataset]

    save_path = setup_logging_from_args(args)
    writer = SummaryWriter(save_path)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)
        args.gpus = [int(i) for i in args.gpus.split(',')]
        torch.cuda.set_device(args.gpus[0])
        cudnn.benchmark = True
        torch.cuda.manual_seed(args.seed)

    if args.model == 'vqvae': 
        model = models[args.dataset][args.model](hidden, k=k, num_channels=num_channels)
    
    elif args.model == 'vae' or args.model == 'ae':
        model = models[args.dataset][args.model](hidden, num_channels=num_channels) # TODO should we tune kl_coeff?

    # Add language auxiliary loss through wrapper class. 
    if args.use_language: 
        
        if model.model_type == 'vqvae': 
            d_model = 512 
        elif model.model_type == 'vae' or model.model_type == 'ae':
            d_model = hidden

        model = AutoEncoderLM(model, d_model, 4, 64, 2, 0.2)

    if args.cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10 if args.dataset == 'imagenet' else 30, 0.5,)

    kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}
    dataset_train_dir = os.path.join(args.data_dir, dataset_dir_name)
    dataset_test_dir = os.path.join(args.data_dir, dataset_dir_name)
    
    if args.dataset == 'coco': 

        # Paths to image splits. 
        train_dir = os.path.join(args.data_dir, 'train2017')
        val_dir = os.path.join(args.data_dir, 'val2017')

        # Paths to annotations
        train_anns = os.path.join(args.data_dir, 'annotations/captions_train2017.json')
        val_anns = os.path.join(args.data_dir, 'annotations/captions_val2017.json')

        # Return bert tokenizer ids tensor. 
        def tokenize(text):
            
            # Select random caption from example. 
            ex_id = np.random.randint(0, len(text))
            example = text[ex_id]

            # Tokenize.
            return tokenizer(example, return_tensors='pt', max_length=75, padding='max_length', truncation=True)['input_ids']

        # Datasets
        train_dataset = datasets_classes['coco'](train_dir, train_anns, \
                                                 transform=dataset_transforms['coco'],
                                                 target_transform=tokenize)

        val_dataset = datasets_classes['coco'](val_dir, val_anns, \
                                                 transform=dataset_transforms['coco'],
                                                 target_transform=tokenize)
        
        # Dataloaders
        train_loader = torch.utils.data.DataLoader(
                train_dataset, 
                batch_size=args.batch_size, 
                shuffle=True, 
                **kwargs)

        test_loader = torch.utils.data.DataLoader(
                val_dataset, 
                batch_size=args.batch_size, 
                shuffle=False,
                **kwargs)

    else:
        if args.dataset in ['imagenet', 'custom']:
            dataset_train_dir = os.path.join(dataset_train_dir, 'train')
            dataset_test_dir = os.path.join(dataset_test_dir, 'val')
        train_loader = torch.utils.data.DataLoader(
            datasets_classes[args.dataset](dataset_train_dir,
                                           transform=dataset_transforms[args.dataset],
                                           **dataset_train_args[args.dataset]),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets_classes[args.dataset](dataset_test_dir,
                                           transform=dataset_transforms[args.dataset],
                                           **dataset_test_args[args.dataset]),
            batch_size=args.batch_size, shuffle=False, **kwargs)

    for epoch in range(1, args.epochs + 1):
        train_losses = train(epoch, model, train_loader, optimizer, args.cuda,
                             args.log_interval, save_path, args, writer)
        test_losses = test_net(epoch, model, test_loader, args.cuda, save_path, args, writer)

        for name in train_losses.keys():
            writer.add_scalar(name, train_losses[name], epoch)

        for name in test_losses.keys():
            writer.add_scalar(name, test_losses[name], epoch)
        
        scheduler.step()


def train(epoch, model, train_loader, optimizer, cuda, log_interval, save_path, args, writer):
    model.train()
    loss_dict = model.latest_losses()
    losses = {k + '_train': 0 for k, v in loss_dict.items()}
    epoch_losses = {k + '_train': 0 for k, v in loss_dict.items()}
    start_time = time.time()
    batch_idx, data = None, None
    
    for batch_idx, batch in enumerate(train_loader):
       
        if args.dataset == 'coco':
            data, captions = batch

            # Format captions for model.
            if args.use_language:
                captions = captions.squeeze()
                lengths = (captions != 0).long().sum(dim=-1)
                captions = captions[:,:lengths.max().item()]
        else:
            data, _ = batch
        
        if cuda:
            data = data.cuda()
        optimizer.zero_grad()
        outputs = model(data)

        loss = model.loss_function(data, *outputs)
        
        # Combined loss.
        if args.use_language:
            loss = model.caption_loss(captions.cuda(), outputs) + loss

        loss.backward()
        optimizer.step()
        latest_losses = model.latest_losses()
        
        for key in latest_losses:
            losses[key + '_train'] += float(latest_losses[key])
            epoch_losses[key + '_train'] += float(latest_losses[key])

        if batch_idx % log_interval == 0:
            for key in latest_losses:
                losses[key + '_train'] /= log_interval
            loss_string = ' '.join(['{}: {:.6f}'.format(k, v) for k, v in losses.items()])
            logging.info('Train Epoch: {epoch} [{batch:5d}/{total_batch} ({percent:2d}%)]   time:'
                         ' {time:3.2f}   {loss}'
                         .format(epoch=epoch, batch=batch_idx * len(data), total_batch=len(train_loader) * len(data),
                                 percent=int(100. * batch_idx / len(train_loader)),
                                 time=time.time() - start_time,
                                 loss=loss_string))
            start_time = time.time()
            # logging.info('z_e norm: {:.2f}'.format(float(torch.mean(torch.norm(outputs[1][0].contiguous().view(256,-1),2,0)))))
            # logging.info('z_q norm: {:.2f}'.format(float(torch.mean(torch.norm(outputs[2][0].contiguous().view(256,-1),2,0)))))
            for key in latest_losses:
                losses[key + '_train'] = 0
        if batch_idx == (len(train_loader) - 1):
            save_reconstructed_images(data, epoch, outputs[0], save_path, 'reconstruction_train')

            write_images(data, outputs, writer, 'train')

        if args.dataset in ['imagenet', 'custom'] and batch_idx * len(data) > args.max_epoch_samples:
            break

    for key in epoch_losses:
        if args.dataset != 'imagenet':
            epoch_losses[key] /= (len(train_loader.dataset) / train_loader.batch_size)
        else:
            epoch_losses[key] /= (len(train_loader.dataset) / train_loader.batch_size)
    loss_string = '\t'.join(['{}: {:.6f}'.format(k, v) for k, v in epoch_losses.items()])
    logging.info('====> Epoch: {} {}'.format(epoch, loss_string))
   
    if args.model == 'vqvae': 
        writer.add_histogram('dict frequency', outputs[3], bins=range(args.k + 1))
        model.print_atom_hist(outputs[3])
    
    return epoch_losses


def test_net(epoch, model, test_loader, cuda, save_path, args, writer):
    model.eval()
    loss_dict = model.latest_losses()
    losses = {k + '_test': 0 for k, v in loss_dict.items()}
    losses['inception_score_test'] = 0
    i, data = None, None
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader)):

            if args.dataset == 'coco':
                data, captions = batch

                # Format captions for model.
                if args.use_language:
                    captions = captions.squeeze()
                    lengths = (captions != 0).long().sum(dim=-1)
                    captions = captions[:,:lengths.max().item()]
            else:
                data, _ = batch
                
            if cuda:
                data = data.cuda()
            outputs = model(data)
            model.loss_function(data, *outputs)

            # Combined loss.
            if args.use_language:
                model.caption_loss(captions.cuda(), outputs)

            latest_losses = model.latest_losses()

            # Add inception score. 
            imgs = outputs[0].cpu().numpy()
            imgs = 2*((imgs - imgs.min()) / (imgs.max() - imgs.min())) - 1
            batch_size = 16 if imgs.shape[0] >= 16 else 1
            i_score, i_std = inception_score(imgs, cuda=True, batch_size=batch_size, resize=True)
            latest_losses['inception_score'] = i_score

            for key in latest_losses:
                losses[key + '_test'] += float(latest_losses[key])
            if i == 0:
                write_images(data, outputs, writer, 'test')

                save_reconstructed_images(data, epoch, outputs[0], save_path, 'reconstruction_test')
                save_checkpoint(model, epoch, save_path)
            if args.dataset == 'imagenet' and i * len(data) > 1000:
                break

    for key in losses:
        if args.dataset not in ['imagenet', 'custom']:
            losses[key] /= (len(test_loader.dataset) / test_loader.batch_size)
        else:
            losses[key] /= (i * len(data))
    loss_string = ' '.join(['{}: {:.6f}'.format(k, v) for k, v in losses.items()])
    logging.info('====> Test set losses: {}'.format(loss_string))
    return losses


def write_images(data, outputs, writer, suffix):
    original = data.mul(0.5).add(0.5)
    original_grid = make_grid(original[:6])
    writer.add_image(f'original/{suffix}', original_grid)
    reconstructed = outputs[0].mul(0.5).add(0.5)
    reconstructed_grid = make_grid(reconstructed[:6])
    writer.add_image(f'reconstructed/{suffix}', reconstructed_grid)


def save_reconstructed_images(data, epoch, outputs, save_path, name):
    size = data.size()
    n = min(data.size(0), 8)
    batch_size = data.size(0)
    comparison = torch.cat([data[:n],
                            outputs.view(batch_size, size[1], size[2], size[3])[:n]])
    save_image(comparison.cpu(),
               os.path.join(save_path, name + '_' + str(epoch) + '.png'), nrow=n, normalize=True)


def save_checkpoint(model, epoch, save_path):
    os.makedirs(os.path.join(save_path, 'checkpoints'), exist_ok=True)
    checkpoint_path = os.path.join(save_path, 'checkpoints', f'model_{epoch}.pth')
    torch.save(model.state_dict(), checkpoint_path)


if __name__ == "__main__":
    main(sys.argv[1:])
