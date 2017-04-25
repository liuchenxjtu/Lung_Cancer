import torch
import torch.nn as nn
from math import sqrt
import torch.nn.functional as F

class Xnet(nn.Module):
    def __init__(self):
        super(Xnet, self).__init__()

        self.features = nn.Sequential(
            # input is 1 x 64 x 64 x 64
            nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),

            # state size. 64 x 64 x 32 x 32
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2)),

            # state size. 128 x 32 x 16 x 16
            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2)),
            
            # state size. 256 x 16 x 8 x 8    
            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2)),

            # state size. 256 x 8 x 4 x 4 
            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2)),
            # state size. 256 x 4 x 2 x 2 
        )

        self.ReLU = nn.ReLU(inplace=True)
        self.Dropout = nn.Dropout(p=0.5)
        self.fc6 = nn.Linear(256 * 4 * 2 * 2, 2048)
        self.fc7 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 2)
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def forward(self, input):
        out = self.features(input)
        
        out = out.view(out.size(0), -1)
        
        out = self.fc6(out)
        out = self.ReLU(out)
        out = self.Dropout(out)

        out = self.fc7(out)
        out = self.ReLU(out)
        out = self.Dropout(out)
        
        out = self.fc2(out)

        return F.log_softmax(out)