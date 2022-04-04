import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from flash.image import ImageClassificationData, ImageClassifier
import argparse
from pytorch_lightning import seed_everything
from flash import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

seed_everything(7)

parser = argparse.ArgumentParser()
parser.add_argument('--gpus', type=int, default=0,
                            help='number of gpus to use for training')
parser.add_argument('--strategy', type=str, default='ddp',
                            help='strategy to use for training')
parser.add_argument('--batch_size', type=int, default=64,
                            help='batch size to use for training')
parser.add_argument('--epochs', type=int, default=5,
                            help='maximum number of epochs for training')
parser.add_argument('--data_dir', type=str, default='/datastores/cifar5',
                            help='the directory to load data from')
parser.add_argument('--learning_rate', type=float, default=1e-4, 
                            help='the learning rate to use during model training')
parser.add_argument('--optimizer', type=str, default='Adam',
                            help='the optimizer to use during model training')
args = parser.parse_args()

transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.4913, 0.482, 0.446], std=[0.247, 0.243, 0.261])
    ])

datamodule = ImageClassificationData.from_folders(
        train_folder=args.data_dir + '/train',
        val_folder=args.data_dir + '/test',
        test_folder=args.data_dir + '/test',
        batch_size=args.batch_size,
        transform_kwargs={'mean': (0.4913, 0.482, 0.446), 'std': (0.247, 0.243, 0.261)}
     )


# %%
model = ImageClassifier(backbone="resnet18", num_classes=datamodule.num_classes, learning_rate=args.learning_rate, optimizer=args.optimizer)

trainer = Trainer(
    progress_bar_refresh_rate=10,
    max_epochs=args.epochs,
    gpus=args.gpus,
    #logger=TensorBoardLogger("lightning_logs/", name="resnet"),
    #callbacks=[LearningRateMonitor(logging_interval="step")],
)

if __name__ == '__main__':
    trainer.fit(model, datamodule=datamodule)
    #print('finished fitting')
    trainer.test(model, datamodule=datamodule)

