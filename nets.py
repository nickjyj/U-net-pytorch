import torch.nn as nn
import torch.nn.functional as F
import torch

class Conv(nn.Module):
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
    def __init__(self, image_size, ls):
        super().__init__()

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.convs = nn.ModuleList()

        # down part
        tmp = [image_size[-1]] + ls
        for i in range(len(tmp)-1):
            inc,outc = tmp[i],tmp[i+1]
            if i==0:
                self.downs += [Conv(inc, outc)]
            else:
                self.downs += [nn.Sequential(Conv(inc, outc),nn.MaxPool2d(2,2))]

        # up part
        ls2 = ls[::-1]
        for i in range(len(ls2)-1):
            inc, outc = ls2[i], ls2[i+1]
            self.ups += [nn.ConvTranspose2d(inc, outc, kernel_size=2, stride=2)]
            self.convs += [Conv(inc,outc)]


    def forward(self, x):
        ds = []
        for down in self.downs:
            x = down(x)
            ds.append(x)
        ds.pop() # remove last one
        
        for up,conv in zip(self.ups,self.convs):
            x = up(x)
            t = ds.pop()
            x = torch.cat((x,t),1)
            x = conv(x)
        return x
           

"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m = torch.rand([2,3,256,256]).to(device)
image_size = [256, 256, 3]
channels = [64, 128, 256, 512, 1024]
net = Unet(image_size, channels).to(device)
out = net(m)
"""