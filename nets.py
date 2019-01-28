import torch.nn as nn
import torch.nn.functional as F
import torch

class ConvBlock(nn.Module):
    def __init__(self, inc, outc, rep=2):
        super().__init__()
        layers = self.conv(inc, outc, times=rep)
        self.model = nn.Sequential(*layers)


    def conv(self, inc, outc, p=1, times=2):
        ls = []
        ls += [nn.Conv2d(inc, outc, kernel_size=3, stride=1, padding=p),
                    nn.BatchNorm2d(outc),
                    nn.ReLU()]
        for _ in range(times-1):
            ls += [nn.Conv2d(outc, outc, kernel_size=3, stride=1, padding=p),
                        nn.BatchNorm2d(outc),
                        nn.ReLU()]
        return ls


    def forward(self, x):
        out = self.model(x)
        return out



class Unet(nn.Module):
    def __init__(self, in_channels, ls):
        super().__init__()

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.input = ConvBlock(in_channels, ls[0])
        self.output = ConvBlock(ls[0], in_channels)

        # down part
        for i in range(len(ls)-1):
            inc,outc = ls[i],ls[i+1]
            self.downs += [nn.Sequential(nn.MaxPool2d(2,2), ConvBlock(inc, outc))]

        # up part
        ls2 = ls[::-1]
        for i in range(len(ls2)-1):
            inc, outc = ls2[i], ls2[i+1]
            self.ups += [nn.ConvTranspose2d(inc, outc, kernel_size=2, stride=2)]
            self.convs += [ConvBlock(inc,outc)]


    def forward(self, x):
        
        x = self.input(x)
        ds = [x]
        for down in self.downs:
            x = down(x)
            ds.append(x)
        ds.pop() # remove last one
        
        for up,conv in zip(self.ups,self.convs):
            x = up(x)
            t = ds.pop()
            x = torch.cat((x,t),1)
            x = conv(x)
        x = self.output(x)
        return x
           

"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m = torch.rand([2,3,256,256]).to(device)
channels = [64, 128, 256, 512, 1024]
net = Unet(3, channels).to(device)
out = net(m)
print(m.size(),out.size())
"""