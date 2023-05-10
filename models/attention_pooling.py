import torch
import torch.nn as nn
import torch.nn.functional as F



class AttentionPooling(nn.Module):
    def __init__(self, in_channel = 1024, compression_ratio = 16):
        super(AttentionPooling, self).__init__()
        self.autoencoder = nn.Sequential(nn.Linear(in_channel, in_channel // compression_ratio),
                                            nn.ReLU(),
                                            nn.Linear(in_channel // compression_ratio, in_channel))
        self.sigmoid = nn.Sigmoid()
    def forward(self,x_mp, x_ap):
        batch_size = x_ap.size(0)
        w_ap = self.sigmoid(self.autoencoder(x_ap))
        w_mp = self.sigmoid(self.autoencoder(x_mp))
        # x = torch.cat((x_ap, x_mp), dim=1)
        # x = self.autoencoder(x)
        # x = self.sigmoid(x)
        x_ap = x_ap * w_ap
        x_mp = x_mp * w_mp
        return x_ap +  x_mp
