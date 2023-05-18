import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import f1_score

from utils import read_yaml

class LightningModel(pl.LightningModule) :
    config = read_yaml('config.yaml')
    def __init__(self, config = config, num_classes = 10) :
        super(LightningModel, self ).__init__()

        # configuration 
        self.config = config

        # layers
        self.conv1 = nn.Conv2d(1, 10, kernel_size = 7) # B, 10, 22, 22
        self.conv2 = nn.Conv2d(10, 20, kernel_size = 7) # B, 20, 16, 16
        self.fc1 = nn.Linear(16*16*20, 50)
        self.classifier = nn.Linear(50, 10)
        
        # loss function
        self.criterion = nn.NLLLoss()

        # lr
        self.lr = float(self.config['lr'])

        self.val_preds = []
        self.val_labels = []

    def forward(self, x) :
        output = F.relu(self.conv1(x))
        output = F.relu(self.conv2(output))
        output = output.view(-1, 16*16*20)
        output = F.relu(self.fc1(output))
        output = self.classifier(output)
        output = F.log_softmax(output)

        return output

    def configure_optimizers(self) :
        optimizer = optim.Adam(self.parameters(), lr = self.lr)
        return optimizer

    def training_step(self, batch, batch_idx) :
        imgs, labels = batch
        pred = self(imgs)
        train_loss = self.criterion(pred, labels)
        self.log('train_loss', train_loss)

        return train_loss

    def validation_step(self, batch, batch_idx) :
        imgs, labels = batch
        pred = self(imgs)
        val_loss = self.criterion(pred, labels)
        self.log('val_loss', val_loss)

        self.val_preds += pred.argmax(1).detach().cpu().numpy().tolist()
        self.val_labels += labels.detach().cpu().numpy().tolist()

        return val_loss

    def on_validation_epoch_end(self) :
        val_f1score = f1_score(self.val_labels, self.val_preds, average = 'weighted')
        self.log('val_score', val_f1score)

        self.val_labels.clear()
        self.val_preds.clear()

        return val_f1score

    def predict_step(self, batch, batch_idx) :
        x = batch
        pred = self(x)
        return pred


