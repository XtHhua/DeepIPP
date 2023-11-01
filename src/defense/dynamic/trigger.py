'''
Descripttion: DeepModelIPProtection
version: 1.0
Author: XtHhua
Date: 2023-10-26 22:19:10
LastEditors: XtHhua
LastEditTime: 2023-10-29 10:08:34
'''
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import DefaultDict
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import model

class ImageTrigger:
    def __init__(self, model:nn.Module, config:DefaultDict) -> None:
        self.model = model
        self.config = config

    def _pre_process(self):
        dataset = self.config['dataset']
        batch_size = self.config['batch_size']
        dataloader = DataLoader(dataset, batch_size, True)
        triggerset = self.config['triggerset']
        
    
    def register_ip(self):
        dataloader = self.config['dataloader']
        loss_fn = self.config['loss_fn']
        optimizer = self.config['optimizer']
        device = self.config['device']
        epoches = self.config['epoches']
        trainer = model.Trainer(self.model, dataloader, loss_fn, optimizer, device)
        
        if self.config['verbose']:
            sw = SummaryWriter(self.config['verbose_file'])

        for epoch in tqdm(range(epoches)):
            train_loss = trainer.train()
            sw.add_scalar('Loss/Train',train_loss, epoch)
            