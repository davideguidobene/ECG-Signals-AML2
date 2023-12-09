import pickle
from datetime import datetime
from colorama import Fore, Back, Style

import torch
import numpy as np
import torchvision
import lightning as L
import torch.nn.functional as F
from torch import nn, optim, utils
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split, TensorDataset


# GLOBAL VARS #######################################################################
epoch_n = 30
batch_size = 100
training_loader, validation_loader = None, None

# DATA LOADING ######################################################################
def loadData(trainPath, validPath):
    global training_loader, validation_loader
    with open(trainPath, 'rb') as file:
        train_data = pickle.load(file)
    with open(validPath, 'rb') as file:
        valid_data = pickle.load(file)

    X, y = train_data
    X, y = torch.tensor(X, dtype=torch.double), torch.tensor(y, dtype=torch.double)
    training_set = TensorDataset(X, y)

    X, y = valid_data
    X, y = torch.tensor(X, dtype=torch.double), torch.tensor(y, dtype=torch.double)
    validation_set = TensorDataset(X, y)

    training_loader = utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
    validation_loader = utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False)

# MODEL #############################################################################
class ReshapeLayer(nn.Module):
    def forward(self, x):
        return x.permute(0,2,1)

class sequential_LTSM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first, bidirectional):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=self.input_size, 
            hidden_size=self.hidden_size, 
            batch_first=self.batch_first, 
            bidirectional=self.bidirectional)

    def forward(self, x):
        x = x.permute(0,2,1)
        x, _ = self.lstm(x)
        return x  

class Residual_Block_1(nn.Module):
    def __init__(self):
        super().__init__()

        self.residual_block_1a = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=16, stride=2, padding=7), ##STRIDE2 PADDING7 INDEPENDENT OF SIZE
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=16, stride=1, padding="same")
        )
        self.residual_block_1b = nn.MaxPool1d(kernel_size=2)

    def forward(self, x):
        return self.residual_block_1a(x) + self.residual_block_1b(x)

class Residual_Block_2(nn.Module):
    def __init__(self):
        super().__init__()

        self.residual_block_2a = nn.Sequential(
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=16, stride=2, padding=7),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=16, stride=1, padding="same")            
        )
        self.residual_block_2b = nn.MaxPool1d(kernel_size=2)

    def forward(self, x):
        return self.residual_block_2a(x) + self.residual_block_2b(x)

class DNN1(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=16, stride=2, padding=7),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),

            Residual_Block_1(),
            Residual_Block_2(),
            Residual_Block_2(),
            Residual_Block_2(),
            Residual_Block_2(),

            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            sequential_LTSM(input_size=64, hidden_size=64, batch_first=True, bidirectional=True),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=128, out_features=1),
            nn.Sigmoid()
        )
        self.train_phase = nn.Sequential(
            nn.Linear(in_features=16, out_features=4),  
            nn.Softmax(dim=2)
        )

    def forward(self, x):
        mod = self.model(x)
        res = self.train_phase(mod.permute(0,2,1)).squeeze()
        return res
    
    def features(self, x):
        return self.model(x)

# TRAINING ##########################################################################
class dnn1_module(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = DNN1()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.CrossEntropyLoss()
    
    def training_step(self, batch, batch_idx):
        #print("\nparams None:")
        #print(None in [i.grad for i in self.model.parameters()])   

        inputs, labels = batch
        inputs = torch.reshape(inputs.float(),(inputs.shape[0], 1, inputs.shape[1])).to(device())
        labels = F.one_hot(labels.to(torch.int64), num_classes = 4).squeeze().to(torch.float).to(device())

        outputs = self.model(inputs)
        #print(outputs, labels, outputs.shape, labels.shape)
        loss = self.loss_fn(outputs, labels)
        #print("loss:", loss, "mean: ", torch.mean(outputs-labels))#, "outputs: ", outputs)
        return loss
    
    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        inputs = torch.reshape(inputs.float(),(inputs.shape[0], 1, inputs.shape[1])).to(device())
        labels = F.one_hot(labels.to(torch.int64), num_classes = 4).squeeze().to(torch.float).to(device())
        x, y = inputs, labels

        out = self.model(x)
        loss = self.loss_fn(out, y)
        out = torch.argmax(out, dim=1)
        y = torch.argmax(y, dim=1)
        valid_acc = torch.sum(y == out).item() / (len(y) * 1.0)

        self.log_dict({'valid_loss': loss, 'valid_acc': valid_acc})
    
    def configure_optimizers(self):
        return self.optimizer

def device():
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    return torch.device(dev)

def train(trainPath, validPath, checkpoint, retrain):
    global training_loader, validation_loader
    print("Cuda: ", torch.cuda.is_available)

    loadData(trainPath, validPath)

    model = DNN1()
    model.to(device())
    if retrain:
        model_class = dnn1_module()
        model_class.to(device())
        trainer = L.Trainer(max_epochs=epoch_n, accelerator="gpu", devices=1)
        ##training metrics
        trainer.fit(model=model_class, train_dataloaders=training_loader)
        ##validation metrics
        trainer.test(model=model_class, dataloaders=validation_loader)
    else:
        model_class = dnn1_module.load_from_checkpoint(checkpoint)
        trainer = L.Trainer(max_epochs=epoch_n, accelerator="gpu", devices=1)
        trainer.test(model=model_class, dataloaders=validation_loader)

def getDeepFeatures1024(inData, checkpoint):
    # inData = array of len 1024
    model_class = dnn1_module.load_from_checkpoint(checkpoint)
    return model_class.model.features(inData)

def getDeepFeatures(inData, checkpoint, chunkSize = 1024, overlap = False):
    # inData = full ecg signal
    model_class = dnn1_module.load_from_checkpoint(checkpoint)

    featureArray = []

    if overlap:
        pass
    else:
        for i in range(0, len(inData), chunkSize):
            if i+chunkSize > len(inData): break
            featureArray.append(model_class.model.features(inData[i:i+chunkSize]))
    


    featureArray = torch.tensor(featureArray).to(device())
    return torch.mean(featureArray, dim=0)
