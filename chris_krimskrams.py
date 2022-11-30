from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TrainDataset(Dataset):
    def __init__(self, argX, argy, mean=None, std=None):
        argX = argX.astype(np.float32)

        if mean is None or std is None:
            self.mean_X = np.mean(argX, axis=0)
            self.std_X = np.std(argX, axis=0)
        else:
            self.mean_X = mean
            self.std_X = std
        self._x = (argX-self.mean_X)/self.std_X

        assert np.isnan(self._x).any() == False

        self._x = torch.from_numpy(self._x)

        self._y = torch.from_numpy(argy)
        #print(self._y)
        assert self._y.shape[1]==1
        assert not torch.is_floating_point(self._y)
        self._y = torch.nn.functional.one_hot(self._y.unsqueeze(0).to(torch.int64), num_classes = 4).reshape(-1,4).float()

        assert self._x.shape[0] > 0
        assert self._x.shape[0] == self._y.shape[0]

         
    def __len__(self):
        return self._x.shape[0]
    
    def __getitem__(self, idx):
        return self._x[idx], self._y[idx]
    
class TestDataset(TrainDataset):
    def __init__(self, argX, mean, std):
        #takes df
        argX = argX.astype(np.float32)

        self._x = (argX-mean)/std

        assert np.isnan(self._x).any() == False

        self._x = torch.from_numpy(self._x)
        #print(self._y)
        assert self._x.shape[0] > 0


         
    def __len__(self):
        return self._x.shape[0]
    
    def __getitem__(self, idx):
        return self._x[idx]

class MLP(nn.Module):
    def __init__(self, inputs):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(inputs, inputs*2),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(inputs*2, 4),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        out = self.mlp(x)
        return out
    
def soft_f1_loss(y, y_hat):
    tp = torch.sum(y_hat * y, dim=1)
    fp = torch.sum(y_hat * (1 - y), dim=1)
    fn = torch.sum((1 - y_hat) * y, dim=1)
    
    precision = tp/(tp+fp)
    recall = tp/(fn+tp)

    soft_f1 = 2*precision*recall/(precision+recall+1e-16)

    return torch.mean(1 - soft_f1)
    
def _predict(model, X, mean, std):
    test_set  = TestDataset(X, mean, std)
    test_loader  = DataLoader(test_set,  batch_size=X.shape[0], shuffle=False)
    model.eval()
    with torch.no_grad():
        for x in test_loader:
            x = x.to(device).float()
            predictions = model(x).detach().cpu().numpy()
            return np.argmax(predictions, axis=1)

def train(epochs, X_train, y_train, X_val, y_val, batch_size=32):
    train_set = TrainDataset(X_train, y_train)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    train_X_mean = train_set.mean_X
    train_X_std = train_set.std_X

    if X_val is not None:
        val_set = TrainDataset(X_val, y_val, train_X_mean, train_X_std)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)


    model = MLP(X_train.shape[1]).to(device)

    optimizer = torch.optim.Adam(model.parameters())
    #criterion = nn.CrossEntropyLoss()
    #mse_loss = nn.MSELoss()
    #print(model)

    
    val_loss_timeseries = []
    train_loss_timeseries = []

    for epoch in tqdm(range(epochs)):
        losses = []
        model.train()
        for batch_num, input_data in enumerate(train_loader):
            optimizer.zero_grad()
            x, y = input_data
            x = x.to(device).float()
            y = y.to(device)

            output = model(x)
            #print(output.shape)
            #print(y.shape)
            loss = soft_f1_loss(y, output)
            #loss = mse_loss(output, y)
            #print(loss)
            loss.backward()
            losses.append(loss.item())

            optimizer.step()
        train_loss_timeseries.append(sum(losses)/len(losses))
        
            #if batch_num % 4 == 0:
                #print('\tEpoch %d | Batch %d | Loss %6.2f' % (epoch, batch_num, loss.item()))
        #print('--- TRAINING Epoch %d | Loss %6.2f' % (epoch, sum(losses)/len(losses)))
        if X_val is not None:
            valid_losses = []
            model.eval()     # Optional when not using Model Specific layer
            for input_data in val_loader:
                x, y = input_data
                x = x.to(device).float()
                y = y.to(device)


                output = model(x)
                loss = soft_f1_loss(y, output)
                valid_losses.append(loss.item())
            #print('--- VALIDATION Epoch %d | Loss %6.5f' % (epoch, sum(valid_losses)/len(valid_losses)))
            #print("")
            val_loss_timeseries.append(sum(valid_losses)/len(valid_losses))
            

    #plt.semilogy(range(epochs),val_loss_timeseries)
    #plt.show()

    predict_funct = lambda X: _predict(model, X, train_X_mean, train_X_std)

    if X_val is not None:
        return train_loss_timeseries, val_loss_timeseries, predict_funct
    else:
        return train_loss_timeseries, predict_funct


'''
model = MLP(11)
l = torch.from_numpy(np.arange(0,22).reshape((2,11))).float()
print(l)
res= model(l).detach().numpy()
res = np.argmax(res, axis=1, keepdims=True)
print(res)
'''

'''A = torch.tensor([[0], [2], [1], [0], [1], [3]])
output = torch.nn.functional.one_hot(A.unsqueeze(0).to(torch.int64), num_classes = 4).reshape(-1,4)

print(output)
'''
'''target=torch.tensor([[1,0,0],[0,1,0],[1,0,0]]).float()
pred = torch.tensor([[0.1,0.6,0.3],[0.0,1,0],[0.3,0.3,0.4]])
loss = torch.nn.CrossEntropyLoss()(pred,target)

print(loss)'''