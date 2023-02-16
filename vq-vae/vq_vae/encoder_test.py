from vq_vae.auto_encoder import VQ_CVAE
import torch
import torch.nn as nn

class EncoderBridge(VQ_CVAE):
    def __init__(self,d=64, k=10, bn=True, vq_coef=1, commit_coef=0.5, num_channels=3,frozen=False, **kwargs):
        super().__init__(d=64, k=10, bn=True, vq_coef=1, commit_coef=0.5, num_channels=3, **kwargs)

        self.bridge = bridge = nn.Sequential(
            nn.Conv2d(64,128,4,2,1),
            nn.BatchNorm2d(128),
            nn.Conv2d(128,256,4,2,1),
            nn.BatchNorm2d(256),
            nn.Conv2d(256,512,4,2,1),
            nn.BatchNorm2d(512),
            nn.Conv2d(512,1024,4,1,1),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024,2048,3,1,1),
            nn.BatchNorm2d(2048)
        )
        #Freezing encoder paramters
        '''print("frozen: ", frozen)
        if frozen:
            for name, param in self.named_parameters():
                if "encoder" in name:
                    print("Freezing: ", name)
                    param.requires_grad = False'''

    def forward(self, x):
        encoder_out = self.encode(x)
        return self.bridge(encoder_out)
