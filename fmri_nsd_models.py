import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
import numpy as np
import wandb


class Clip2Accuracy(torchmetrics.Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.cosine=torch.nn.CosineSimilarity()

    def update(self, test_embeddings, pred_embeddings):
        rnd_idx=torch.randperm(len(pred_embeddings))
        pred_embeddings_random=pred_embeddings[rnd_idx]

        cor=torch.stack([self.cosine(test_embeddings,pred_embeddings_random),self.cosine(test_embeddings,pred_embeddings)],dim=1)

        self.correct += torch.sum(torch.argmax(cor,dim=1))
        self.total += len(cor)

    def compute(self):
        return self.correct.float() / self.total
    

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim,act_fn=nn.ReLU, alignment_layers_keys=[1,2,5,7],common_dim=1024):
        super(Encoder, self).__init__()
        self.common_dim=common_dim
        self.alignment_layers={}
        for k in alignment_layers_keys:
            self.alignment_layers[k]=nn.LazyLinear(common_dim)

        layers = []
        prev_dim = input_dim
        if len(hidden_dims):

            for hidden_dim in hidden_dims:
                layers.append(nn.LazyLinear(hidden_dim))
                layers.append(act_fn())
            layers.append(nn.LazyLinear(output_dim))
            self.net = nn.Sequential(*layers)     
        else:
            self.net = nn.Linear(input_dim, output_dim)

    def _apply(self, fn):
        super(Encoder, self)._apply(fn)        
        for k,v in self.alignment_layers.items():
            self.alignment_layers[k]._apply(fn)

    def forward(self, x, k=None):

        def apply_alignment_layers(x, k, alignment_layers):
            # Create an empty tensor to store the results
            result = torch.empty_like(x)
            result = result[:,:self.common_dim]

            # Iterate through each unique key in k
            for key in k.unique():
                # Create a mask for all elements that match the current key
                mask = (k == key.item())

                # print(x.shape, result.shape, mask)

                # Apply the corresponding alignment layer to the masked elements
                result[mask] = alignment_layers[key.item()](x[mask])

            return result

        if k is None:
            k=torch.ones(len(x))
        # Apply alignment layers to x using the custom function
        x = apply_alignment_layers(x, k, self.alignment_layers)

        # x = self.alignment_layers[k](x)
        return self.net(x)
    
class ContrastiveModel(pl.LightningModule):

    def contrastive_loss(self, z_i, z_j):

        z_i = nn.functional.normalize(z_i, dim=1)
        z_j = nn.functional.normalize(z_j, dim=1)

        logits = (z_i @ z_j.T) / self.temperature
        similarities = z_j @ z_j.T
        # targets = torch.nn.functional.softmax(similarities * self.temperature, dim=-1)

        targets = torch.arange(logits.shape[0]).long().to(logits.device)

        loss = torch.nn.functional.cross_entropy(logits, targets)
        return loss
    
    def mean_contrastive(self, z_i, z_j, temperature=1.0):
        return nn.functional.mse_loss(z_i, z_j)+self.contrastive_loss(z_i, z_j, temperature=temperature)/8
    
    def cosine_loss(self, z_i, z_j, temperature=1.0):
        cosine_similarity = torch.nn.functional.cosine_similarity(z_i, z_j).mean()
        return 1- cosine_similarity

    def contrastive_2dir(self, z_i, z_j):
        return self.contrastive_loss(z_i, z_j) + self.contrastive_loss(z_i.T, z_j.T)

    def __init__(self, 
                 num_input_channels: int,
                 base_channel_size: int,
                 latent_dim: int,
                 act_fn: object = nn.GELU,
                 temperature=.1,
                 loss_type="contrastive",
                 lr=1e-4,
                 optimizer_name="AdamW"):
        super().__init__()

        self.lr_=lr
        self.optimizer_name=optimizer_name
        self.temperature=temperature
        self.num_input_channels = num_input_channels

        self.model = Encoder(num_input_channels, base_channel_size, latent_dim, act_fn)

        self.loss_type=loss_type
        if loss_type=="contrastive":
            self.loss_fn=self.contrastive_loss
        elif loss_type=="mean_contrastive":
            self.loss_fn=self.mean_contrastive    
        elif loss_type=="mse":
            self.loss_fn=torch.nn.functional.mse_loss
        elif loss_type=="cosine":
            self.loss_fn=self.cosine_loss
        elif loss_type=="contrastive_2dir":
            self.loss_fn=self.contrastive_2dir

        self.train_losses = []
        self.train_mse=[]
        self.train_cosine=[]
        self.val_losses = []
        self.val_mse=[]
        self.val_cosine=[]
    
        self.train_history={}
        self.val_history={}

        #init history
        self.train_history["train_loss"]=[]
        self.train_history["train_mse"]=[]
        self.train_history["train_cosine"]=[]
        self.val_history["val_loss"]=[]
        self.val_history["val_mse"]=[]
        self.val_history["val_cosine"]=[]

        self.test_acc = Clip2Accuracy()
        self.val_acc = Clip2Accuracy()

    def forward(self, x, **kwargs):
        return self.model(x,**kwargs)
    
    def training_step(self, batch, batch_idx):
        x, y,idx= batch
        # x = x.float()
        y_hat = self(x,k=idx)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        self.train_losses.append(loss.item())

        mse_loss = torch.nn.functional.mse_loss(y_hat, y)
        cosine_similarity = torch.nn.functional.cosine_similarity(y_hat, y).mean()
        self.train_mse.append(mse_loss.item())

        return loss
    
    def validation_step(self, batch, batch_idx):

        x, y, idx= batch
        #print('x', x.size(), 'y', y.size(), 'idx',idx)
        # x = x.float()

        y_hat = self(x,k=idx)

        loss=self.loss_fn(y_hat, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)

        mse_loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log('val_mse_loss', mse_loss, on_epoch=True, prog_bar=True)

        cosine_similarity = torch.nn.functional.cosine_similarity(y_hat, y).mean()
        self.log('val_cosine_similarity', cosine_similarity, on_epoch=True, prog_bar=True)

        self.val_losses.append(loss.item())
        self.val_mse.append(mse_loss.item())
        self.val_cosine.append(cosine_similarity.item())

        self.val_acc(y_hat, y)
        self.log("val/acc_epoch", self.val_acc, on_step=False, on_epoch=True)

        return mse_loss
    
    def on_train_epoch_end(self) -> None:
        self.train_history["train_loss"].append(np.mean(self.train_losses))
        self.train_history["train_mse"].append(np.mean(self.train_mse))
        self.train_history["train_cosine"].append(np.mean(self.train_cosine))
        self.train_losses = []
        self.train_mse=[]
        self.train_cosine=[]
        return super().on_train_epoch_end()
    
    def on_validation_epoch_end(self) -> None:
        self.val_history["val_loss"].append(np.mean(self.val_losses))
        self.val_history["val_mse"].append(np.mean(self.val_mse))
        self.val_history["val_cosine"].append(np.mean(self.val_cosine))
        self.val_losses = []

        self.val_mse=[]
        self.val_cosine=[]
        return super().on_validation_epoch_end()
    def test_step(self, batch, batch_idx):

        x, y, idx= batch
        # x = x.float()

        y_hat = self(x,k=idx)

        self.test_acc(y_hat, y)
        self.log("test/acc_epoch", self.test_acc, on_step=False, on_epoch=True)

    def on_test_epoch_end(self):
        print("epoch end placeholder")
        #dummy_input = torch.zeros(self.num_input_channels, device=self.device)
        #model_filename = "model_final.onnx"
        #torch.onnx.export(self, dummy_input, model_filename)
        #artifact = wandb.Artifact(name="model.ckpt", type="model")
        #artifact.add_file(model_filename)
        #wandb.log_artifact(artifact)

    def configure_optimizers(self):
        # return torch.optim.AdamW(self.parameters(), lr=3e-4, weight_decay=0)
        # add a scheduler
        if self.optimizer_name=="AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr_, weight_decay=1e-3)
        elif self.optimizer_name=="SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr_)
        # use a scheduler that every 100 steps, it will reduce the learning rate by 0.1
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=50, verbose=True)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
