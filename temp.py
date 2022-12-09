import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, init_channels, inputs):
        super(MLP, self).__init__()
        self.firstStage = self._make_layers(init_channels,[16, 'M', 32, 'M', 64, 'M',128,'M',256,'M',512,'M',1024,'M'])
        self.classifier = nn.Linear(1024, 12)
        self.mlp = nn.Sequential(
            nn.Linear(inputs+12, (inputs+12)*2),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear((inputs+12)*2, 4),
            nn.Softmax(dim=1)
        )


    def forward(self, conv,x):
        conv_out = self.firstStage(conv)
        conv_out = conv_out.view(conv_out.size(0), -1)
        conv_out = self.classifier(conv_out)
        c = torch.concat((x,conv_out), dim=1)
        print(c.shape)
        out = self.mlp(c)
        return out

    def _make_layers(self,init_channels, cfg):
        layers = []
        in_channels = init_channels
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool1d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv1d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm1d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool1d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

batchsize = 4
init_channels = 3
a = torch.arange(batchsize*init_channels*180).reshape(batchsize,init_channels,180).float()
b = torch.arange(400*batchsize).reshape(batchsize,400).float()
print(a.shape)
print(b.shape)

model = MLP(init_channels,400)

l = model(a,b)

print(l)
