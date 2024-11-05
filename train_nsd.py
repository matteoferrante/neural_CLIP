from ml_collections import config_dict

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np

from tqdm import tqdm

from fmri_nsd_models import ContrastiveModel

import argparse

import matplotlib.pyplot as plt


sweep_config = config_dict.ConfigDict()
sweep_config.lr = 0.0001
sweep_config.max_epochs = 20
sweep_config.batch_size = 256
sweep_config.latent_dim = 768
sweep_config.base_channel_size = 768
sweep_config.n_encoder_net_layers = 1
sweep_config.act_fn = "ReLU"
sweep_config.optimizer_name = "AdamW"
sweep_config.loss_type = "contrastive"
sweep_config.PROJECT_NAME = 'sweep-nsd-clip-3'


def parse_args():
    argparser = argparse.ArgumentParser(description="Process hyper-parameters")
    argparser.add_argument("--lr", type=float, default=sweep_config.lr)
    argparser.add_argument("--act_fn", type=str, default=sweep_config.optimizer_name)
    argparser.add_argument("--loss_type", type=str, default=sweep_config.loss_type)
    argparser.add_argument("--optimizer_name", type=str, default=sweep_config.act_fn)
    argparser.add_argument("--latent_dim", type=int, default=sweep_config.latent_dim)
    argparser.add_argument("--base_channel_size", type=int, default=sweep_config.base_channel_size)
    argparser.add_argument("--n_encoder_net_layers", type=int, default=sweep_config.n_encoder_net_layers)
    argparser.add_argument("--max_epochs", type=int, default=sweep_config.max_epochs)
    argparser.add_argument("--batch_size", type=int, default=sweep_config.batch_size)
    return argparser.parse_args()

def get_dataloaders(config):
    train_data = np.load("../data/data_fmri_nsd/train_data.npy")
    test_data = np.load("../data/data_fmri_nsd/test_data.npy")

    num_input_channels = train_data.shape[-1]

    train_clip_img_embeds = torch.load("../data/data_fmri_nsd/train_clip_img_embeds.pt")
    test_clip_img_embeds = torch.load("../data/data_fmri_nsd/test_clip_img_embeds.pt")

    subject_train_ids = np.load("../data/data_fmri_nsd/subject_train_ids.npy").tolist()
    subject_test_ids = np.load("../data/data_fmri_nsd/subject_test_ids.npy").tolist()

    subject_train_ids=[int(i[-1]) for i in subject_train_ids]
    subject_test_ids=[int(i[-1]) for i in subject_test_ids]

    train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_data).float(),train_clip_img_embeds, torch.tensor(subject_train_ids))
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(test_data).float(),test_clip_img_embeds, torch.tensor(subject_test_ids))

    clip_train_dataloader=DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    clip_test_dataloader=DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    return clip_train_dataloader, clip_test_dataloader, num_input_channels


def train(config):
    run =  wandb.init(project=config.PROJECT_NAME, job_type="training", config=dict(config))

    clip_train_dataloader, clip_test_dataloader, num_input_channels = get_dataloaders(config)

    wandb_logger = WandbLogger(project=config.PROJECT_NAME)

    act_fn_ = nn.ReLU
    if config.act_fn == "ReLU":
        act_fn_ = nn.ReLU
    elif config.act_fn == "GELU":
        act_fn_ = nn.GELU
    elif config.act_fn == "Identity":
        act_fn_ = nn.Identity
        config.n_encoder_net_layers =1

    brain_model = ContrastiveModel(num_input_channels=num_input_channels,
                                   base_channel_size=[config.base_channel_size]*config.n_encoder_net_layers,
                                   latent_dim=config.latent_dim,
                                   act_fn=act_fn_,
                                   loss_type=config.loss_type,
                                   lr=config.lr,
                                   optimizer_name=config.optimizer_name)

    mc = ModelCheckpoint(
        save_top_k=-1
    )

    # Instantiate a PyTorch Lightning Trainer
    trainer = pl.Trainer(max_epochs=config.max_epochs,
                        logger=wandb_logger,
                        accelerator="gpu",
                        devices=[2],
                        callbacks=[mc])

    # Train the model
    trainer.fit(brain_model, clip_train_dataloader, clip_test_dataloader)
    trainer.test(dataloaders=clip_test_dataloader,ckpt_path=None)


if __name__ == "__main__":
    key = "..."
    
    wandb.login(key=key)

    sweep_config.update(vars(parse_args()))
    train(sweep_config)

