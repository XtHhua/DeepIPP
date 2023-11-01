'''
Descripttion: DeepModelIPProtection
version: 1.0
Author: XtHhua
Date: 2023-10-28 10:48:42
LastEditors: XtHhua
LastEditTime: 2023-10-28 11:44:25
'''
import torch
from torch.optim import Adam
from torch import Module, device
from torch.utils.data import DataLoader

class Trainer():
    def __init__(self, model:Module, dataloader:DataLoader, loss_fn:Module,optimizer:Adam,device:device) -> None:
        self.model = model
        self.dataloader = dataloader
        self.loss_fn = loss_fn
        self.optimizer=optimizer
        self.device = device
    
    
    def train(self):
        total_batches = len(self.dataloader)
        loss_record = []
        #
        self.model.train()
        for _, batch_data in enumerate(self.dataloader):
            b_x = batch_data[0].to(self.device)
            b_y = batch_data[1].to(self.device)
            output = self.model(b_x)
            loss = self.loss_fn(output, b_y)
            #
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            #
            loss = loss.detach().item()
            loss_record.append(loss)
        mean_train_loss = sum(loss_record) / total_batches
        return mean_train_loss
    
class Tester():
    def __init__(self, model:Module, dataloader:DataLoader, loss_fn:Module,device:device) -> None:
        self.model = model
        self.dataloader = dataloader
        self.loss_fn = loss_fn
        self.device = device


    def test(self):
        total_sample_num = len(self.dataloader.dataset)
        num_batches = len(self.dataloader)
        self.model.eval()
        test_loss, correct_sample_num = 0, 0
        with torch.no_grad():
            for batch_data in self.dataloader:
                b_x = batch_data[0].to(device)
                b_y = batch_data[1].to(device)
                output = self.model(b_x)
                test_loss += self.loss_fn(output, b_y).item()
                correct_sample_num += (
                    (output.argmax(1) == b_y).type(torch.float).sum().item()
                )
        #
        test_loss /= num_batches
        #
        test_accuracy = correct_sample_num / total_sample_num
        return test_loss, test_accuracy