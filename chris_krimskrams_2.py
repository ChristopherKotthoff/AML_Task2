from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TrainDataset(Dataset):
    def __init__(self, argX, FFTfeatures, argy, mean=None, std=None):
        argX = argX.astype(np.float32)
        if mean is None or std is None:
            self.mean_X = np.mean(argX, axis=0)
            self.std_X = np.std(argX, axis=0)
        else:
            self.mean_X = mean
            self.std_X = std
        argX = (argX-self.mean_X)/self.std_X

        features = argX[:,:FFTfeatures]
        convs = argX[:,FFTfeatures:]

        assert convs.shape[1] % 180 == 0
        channels = int(convs.shape[1]/180)
        

        self._convs = convs.reshape(-1,channels,180)
        self._convs = torch.from_numpy(self._convs)

        self._features = features.reshape(-1,FFTfeatures)
        self._features = torch.from_numpy(self._features)


        self._y = torch.from_numpy(argy)

        assert self._y.shape[1]==1
        assert not torch.is_floating_point(self._y)
        self._y = torch.nn.functional.one_hot(self._y.unsqueeze(0).to(torch.int64), num_classes = 4).reshape(-1,4).float()

        assert self._features.shape[0] > 0
        assert self._features.shape[0] == self._y.shape[0]

         
    def __len__(self):
        return self._features.shape[0]
    
    def __getitem__(self, idx):
        return self._convs[idx], self._features[idx], self._y[idx]
    
class TestDataset(TrainDataset):
    def __init__(self, argX,FFTfeatures, mean, std):
        argX = argX.astype(np.float32)
        self.mean_X = mean
        self.std_X = std
        argX = (argX - self.mean_X) / self.std_X

        features = argX[:, :FFTfeatures]
        convs = argX[:, FFTfeatures:]

        assert convs.shape[1] % 180 == 0
        channels = int(convs.shape[1] / 180)

        self._convs = convs.reshape(-1, channels, 180)
        self._convs = torch.from_numpy(self._convs)

        self._features = features.reshape(-1, FFTfeatures)
        self._features = torch.from_numpy(self._features)

        assert self._features.shape[0] > 0

    def __len__(self):
        return self._features.shape[0]

    def __getitem__(self, idx):
        return self._convs[idx], self._features[idx]


#X = np.arange(4*180).reshape(-1,180)
#y = np.arange(4).reshape(-1,1)

#t = TrainDataset(X,y)

#t[0][0].shape
#no


class MLP(nn.Module):
    def __init__(self, init_channels, FFTfeatures):
        super(MLP, self).__init__()
        self.only_conv=False
        self.firstStage = self._make_layers(init_channels,[16, 'M', 32, 'M', 64, 'M',128,'M',256,'M',512,'M',1024,'M'])
        self.classifier = nn.Linear(1024, 12)
        self.mlp = nn.Sequential(
            nn.Linear(FFTfeatures+12, (FFTfeatures+12)*2),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear((FFTfeatures+12)*2, 4),
            nn.Softmax(dim=1)
        )


    def forward(self, conv,x):
        conv_out = self.firstStage(conv)
        conv_out = conv_out.view(conv_out.size(0), -1)
        conv_out = self.classifier(conv_out)
        if self.only_conv:
            return conv_out
        c = torch.concat((x,conv_out), dim=1)
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
'''
batchsize = 4
init_channels = 3
a = torch.arange(batchsize*init_channels*180).reshape(batchsize,init_channels,180).float()

a = nn.Conv1d(parallels, 32, kernel_size=3, padding=1)(a)
print(a.shape)
a = nn.MaxPool1d(kernel_size=2, stride=2)(a)
print(a.shape)
a = nn.MaxPool1d(kernel_size=2, stride=2)(a)
print(a.shape)
a = nn.MaxPool1d(kernel_size=2, stride=2)(a)
print(a.shape)
a = nn.MaxPool1d(kernel_size=2, stride=2)(a)
print(a.shape)
a = nn.MaxPool1d(kernel_size=2, stride=2)(a)
print(a.shape)
a = nn.MaxPool1d(kernel_size=2, stride=2)(a)
print(a.shape)
a = nn.MaxPool1d(kernel_size=2, stride=2)(a)
print(a.shape)
print(a.shape)
model = MLP(init_channels)
out = model(a)

print(out.shape)
'''




def soft_f1_loss(y, y_hat):
    tp = torch.sum(y_hat * y, dim=1)
    fp = torch.sum(y_hat * (1 - y), dim=1)
    fn = torch.sum((1 - y_hat) * y, dim=1)
    
    precision = tp/(tp+fp)
    recall = tp/(fn+tp)

    soft_f1 = 2*precision*recall/(precision+recall+1e-16)

    return torch.mean(1 - soft_f1)
    
def _predict(model, X, mean, std, FFTfeatures):
    test_set  = TestDataset(X,FFTfeatures, mean, std)
    test_loader  = DataLoader(test_set,  batch_size=X.shape[0], shuffle=False)
    model.eval()
    with torch.no_grad():
        for x in test_loader:
            conv, feat = x
            conv = conv.to(device).float()
            feat = feat.to(device).float()
            predictions = model(conv,feat).detach().cpu().numpy()
            return np.argmax(predictions, axis=1)

def _saveConvFeaturess(model, X_train, X_test, mean, std, FFTfeatures):
    train_set  = TestDataset(X_train,FFTfeatures, mean, std)
    train_loader  = DataLoader(train_set,  batch_size=X_train.shape[0], shuffle=False)
    test_set  = TestDataset(X_test,FFTfeatures, mean, std)
    test_loader  = DataLoader(test_set,  batch_size=X_test.shape[0], shuffle=False)

    model.only_conv = True
    model.eval()


    with torch.no_grad():
        for x in train_loader:
            conv, feat = x
            conv = conv.to(device).float()
            feat = feat.to(device).float()
            predictions = model(conv,feat).detach().cpu().numpy()
            np.savetxt("x_train_conv_features.txt",predictions)
            break

    with torch.no_grad():
        for x in test_loader:
            conv, feat = x
            conv = conv.to(device).float()
            feat = feat.to(device).float()
            predictions = model(conv,feat).detach().cpu().numpy()
            np.savetxt("x_test_conv_features.txt",predictions)
            break


    model.only_conv = False

def train(epochs, X_train, y_train, X_val, y_val, batch_size=64, FFTfeatures=None):
    print(f"using device {device}")
    train_set = TrainDataset(X_train,FFTfeatures, y_train)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    train_X_mean = train_set.mean_X
    train_X_std = train_set.std_X

    if X_val is not None:
        val_set = TrainDataset(X_val,FFTfeatures, y_val, train_X_mean, train_X_std)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    assert (X_train.shape[1]-FFTfeatures) % 180 == 0

    init_channels = int((X_train.shape[1]-FFTfeatures)/180)

    model = MLP(init_channels,FFTfeatures).to(device)

    optimizer = torch.optim.Adam(model.parameters())
    crossELoss = nn.CrossEntropyLoss()
    #mse_loss = nn.MSELoss()
    print(model)

    
    val_loss_timeseries = []
    train_loss_timeseries = []
    from os.path import exists

    for epoch in range(epochs):
        losses = []
        model.train()
        for batch_num, input_data in enumerate(train_loader):
            optimizer.zero_grad()
            conv, feat, y = input_data
            conv = conv.to(device).float()
            feat = feat.to(device).float()
            y = y.to(device)

            output = model(conv,feat)
            #print(output.shape)
            #print(y.shape)
            #loss = soft_f1_loss(y, output)
            #loss = mse_loss(output,y)
            loss = crossELoss(output,y)
            #loss = mse_loss(output, y)
            #print(loss)
            loss.backward()
            losses.append(loss.item())

            optimizer.step()
        
            #if batch_num % 40 == 0:
                #print('\tEpoch %d | Batch %d | Loss %6.2f' % (epoch, batch_num, loss.item()))
        print('--- TRAINING Epoch %d | Loss %6.2f' % (epoch, sum(losses)/len(losses)))
        train_loss_timeseries.append(sum(losses)/len(losses))
        
        if X_val is not None:
            valid_losses = []
            model.eval()     # Optional when not using Model Specific layer
            for input_data in val_loader:
                conv, feat, y = input_data
                conv = conv.to(device).float()
                feat = feat.to(device).float()
                y = y.to(device)

                output = model(conv, feat)
                loss = soft_f1_loss(y, output)
                valid_losses.append(loss.item())
            print('--- VALIDATION Epoch %d | Loss %6.5f' % (epoch, sum(valid_losses)/len(valid_losses)))
            #print("")
            torch.save(model.state_dict(), f"conv_models/e{epoch}_v{sum(valid_losses)/len(valid_losses)}")
            val_loss_timeseries.append(sum(valid_losses)/len(valid_losses))
            

    #plt.semilogy(range(epochs),val_loss_timeseries)
    #plt.show()
    predict_funct = lambda X: _predict(model, X, train_X_mean, train_X_std, FFTfeatures)
    save_funct = lambda Xtrain, Xtest: _saveConvFeaturess(model, Xtrain, Xtest, train_X_mean, train_X_std, FFTfeatures)

    if X_val is not None:
        return train_loss_timeseries, val_loss_timeseries, predict_funct,save_funct
    else:
        return train_loss_timeseries, predict_funct, save_funct
    