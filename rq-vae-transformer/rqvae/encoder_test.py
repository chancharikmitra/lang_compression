from models.rqvae.rqvae import RQVAE
import torch
import torch.nn as nn

class EncoderBridge(RQVAE):
    def __init__(self,
                 *,
                 embed_dim=64,
                 n_embed=512,
                 decay=0.99,
                 loss_type='mse',
                 latent_loss_weight=0.25,
                 bottleneck_type='rq',
                 ddconfig=None,
                 checkpointing=False,
                 **kwargs):
        super().__init__(
                 *,
                 embed_dim=64,
                 n_embed=512,
                 decay=0.99,
                 loss_type='mse',
                 latent_loss_weight=0.25,
                 bottleneck_type='rq',
                 ddconfig=None,
                 checkpointing=False,
                 **kwargs)

        self.bridge = bridge = nn.Sequential(
            nn.Conv2d(256, 512, 1,1,1),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 1024, 1,1,1),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024,2048, 1,2,1),
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
