import torch
import torch.nn as nn



class CNNet(nn.Module):
    def __init__(self,
                 in_channel=3,
                 num_joints=13,
                 window_size=3,
                 out_channel=16,
                 num_class = 48,
                 drop_prob = 0.5,
                 device = 'cpu'
                 ):
        super(CNNet, self).__init__()
        
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channel,6,kernel_size=[2,2], padding='same'),
            nn.LeakyReLU(),
            #nn.MaxPool2d(6),
            nn.BatchNorm2d(6),
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(6,9,kernel_size=[2,2], padding='same'),
            nn.LeakyReLU(),
            #nn.MaxPool2d(9),
            nn.BatchNorm2d(9),
        )
        
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(9,12,1),
            nn.LeakyReLU(),
            #nn.MaxPool2d(12),
            nn.BatchNorm2d(12),
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(12,out_channel,1),
            nn.LeakyReLU(),
            #nn.MaxPool2d(16),
            nn.BatchNorm2d(16),
        )
        
        self.l1 = nn.Sequential(
            nn.Linear((out_channel*window_size*num_joints),512),
            nn.Dropout(drop_prob),
            nn.LeakyReLU(),
            nn.Linear(512,128),
            nn.Dropout(drop_prob),
            nn.LeakyReLU(),
            nn.Linear(128,64),
            nn.Dropout(drop_prob),
            nn.LeakyReLU(),
            nn.Linear(64,num_class)
        )
        
        self.sm=nn.Softmax(dim=0)

    def forward(self, x):
        out=self.conv_block1(x)
        out=self.conv_block2(out)
        out=self.conv_block3(out)
        out=self.conv_block4(out)
        out=torch.flatten(out,1)
        encode_out=out
        
        out=self.l1(out)

        out_sm=self.sm(out)

        return out, encode_out
