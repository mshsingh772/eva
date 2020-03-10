import torch
import torch.nn as nn
import torch.nn.functional as F


class Cifar10Model(nn.Module):
    def __init__(self):
        super(Cifar10Model, self).__init__()
        self.dropout_value = 0.1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.Dropout(self.dropout_value),
            nn.ReLU()
        ) # output_size = 32, RF - 3
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), dilation = 2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.Dropout(self.dropout_value),
            nn.ReLU()
        ) # output_size = 32, RF - 5
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 16, RF - 6


        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=64, bias = False),
            nn.BatchNorm2d(64),
            nn.Dropout(self.dropout_value),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1), # output_size = 16, RF - 10
            nn.BatchNorm2d(32),
            nn.Dropout(self.dropout_value),
            nn.ReLU()

        ) # output_size = 16, RF - 10
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Dropout(self.dropout_value),
            nn.ReLU()
        ) # output_size = 16, RF - 14
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 8, RF - 16


        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.Dropout(self.dropout_value),
            nn.ReLU()
        ) # output_size = 8, RF - 22
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.Dropout(self.dropout_value),
            nn.ReLU()
        ) # output_size = 8, RF - 30
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 4, RF - 38


        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.Dropout(self.dropout_value),
            nn.ReLU()
        ) # output_size = 4, RF - 54

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.Dropout(self.dropout_value),
            nn.ReLU()
        ) # output_size = 4, RF - 70
        
        self.gap = nn.AvgPool2d(kernel_size=(4,4)) # output_size = 1, RF - 86

        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 1, RF - 110


    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.pool1(x)
        
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.pool1(x)

        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.pool1(x)

        x = self.convblock7(x)
        x = self.convblock8(x)
        x = self.gap(x)
        x = self.convblock9(x)
        x = x.view(-1, 10)
        return x
    
  


