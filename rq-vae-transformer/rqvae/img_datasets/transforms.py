# Copyright (c) 2022-present, Kakao Brain Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#import torchvision.transforms as transforms
from torchvision import transforms as transforms
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from virtex.data import transforms as T
import torch
import torchvision.transforms.functional as F

class randHorFlipWithCaption(transforms.RandomHorizontalFlip):
    def __init__(self, p=0.5):
        super().__init__(p=0.5)
    def forward(self, img_caption: tuple):
        img, caption = img_caption
        if torch.rand(1) < self.p:
            caption = (
                caption.replace(" left ", "[TMP]").replace(" right ", " left ").replace("[TMP]", " right ")
            )
            return (F.hflip(img), caption)
        else:
            return (img, caption)
        
def captionTransform(transform: transforms):
    def imgOnlyTransform(img_caption: tuple):
        img, caption = img_caption
        return (transform(img), caption)
    return imgOnlyTransform



def create_transforms(config, split='train', is_eval=False):
    if config.transforms.type == 'imagenet256x256' or 'coco': #-------Add coco to the config file 
        if split == 'train' and not is_eval:
            transforms_ = [
                captionTransform(transforms.Resize(256)),
                captionTransform(transforms.RandomCrop(256)),
                randHorFlipWithCaption(p=0.5),
                captionTransform(transforms.ToTensor()),
                captionTransform(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
            ]
        else:
            transforms_ = [
                captionTransform(transforms.Resize(256)),
                captionTransform(transforms.CenterCrop(256)),
                captionTransform(transforms.Resize((256, 256))),
                captionTransform(transforms.ToTensor()),
                captionTransform(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
            ]
    elif 'ffhq' in config.transforms.type:
        resolution = int(config.transforms.type.split('_')[0].split('x')[-1])
        if split == 'train' and not is_eval:
            transforms_ = [
                transforms.RandomResizedCrop(resolution, scale=(0.75, 1.0), ratio=(1.0, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        else:
            transforms_ = [
                transforms.Resize(resolution),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]
    elif config.transforms.type in ['LSUN', 'LSUN-cat', 'LSUN-church', 'LSUN-bedroom']:
        resolution = 256 # only 256 resolution is supoorted for LSUN
        transforms_ = [
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]
    elif config.transforms.type == 'none':
        transforms_ = []
    else:
        raise NotImplementedError('%s not implemented..' % config.transforms.type)

    transforms_ = transforms.Compose(transforms_)

    return transforms_

def create_alb_transforms(config, split='train', is_eval=False):
    if config.transforms.type == 'imagenet256x256' or config.transforms.type == 'coco':
        if split == 'train' and not is_eval:
            transforms_ = A.Compose([
                A.augmentations.geometric.resize.SmallestMaxSize(256), 
                A.RandomCrop(256, 256),
                T.HorizontalFlip(p=0.5),
                A.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
                ),
                ToTensorV2()
            ])
        else:
            transforms_ = A.Compose([
                A.augmentations.geometric.resize.SmallestMaxSize(256,interpolcation=InterpolationMode.Bi),
                A.RandomCrop(256, 256),
                A.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
                ),
                ToTensorV2()
            ])
            """transforms_ = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])"""
    return transforms_