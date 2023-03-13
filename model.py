import pytorch_lightning as pl
from torch import nn
import torch
from torch.utils.data import TensorDataset, DataLoader
from pytorch_metric_learning import miners, losses

torch.multiprocessing.set_sharing_strategy('file_system')


class PoseRAC(pl.LightningModule):

    def __init__(self, train_x, train_y, valid_x, valid_y, dim, heads, enc_layer, learning_rate, seed, num_classes, alpha):
        super().__init__()
        self.save_hyperparameters()

        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=dim, nhead=heads),
                                                         num_layers=enc_layer)

        self.fc1 = nn.Linear(dim, num_classes)

        self.train_x = train_x
        self.train_y = train_y
        self.valid_x = valid_x
        self.valid_y = valid_y
        self.learning_rate = learning_rate
        self.seed = seed
        self.dim = dim
        self.alpha = alpha
        self.loss = nn.BCELoss()
        self.num_classes = num_classes
        self.miner = miners.MultiSimilarityMiner()
        self.loss_func = losses.TripletMarginLoss()

    def forward(self, x):
        x = x.view(-1, 1, self.dim)
        x = self.transformer_encoder(x)
        x = x.view(-1, self.dim)
        x = self.fc1(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(-1, 1, self.dim)
        x = self.transformer_encoder(x)
        embedding = x.view(-1, self.dim)

        hard_pairs = self.miner(embedding, torch.argmax(y.float(),dim=1))
        loss_metric = -self.loss_func(embedding, torch.argmax(y.float(), dim=1), hard_pairs)

        y_hat = self.fc1(embedding)
        y_pred = torch.sigmoid(y_hat)
        loss_classify = self.loss(y_pred, y.float())

        alpha = self.alpha
        loss = alpha * loss_metric + (1 - alpha) * loss_classify

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        y_pred = torch.sigmoid(y_hat)
        loss = self.loss(y_pred, y.float())
        return loss

    def validation_epoch_end(self, val_step_outputs):
        loss = sum(val_step_outputs) / len(val_step_outputs)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.75, patience=6, verbose=1,
                                                               mode='min', cooldown=0, min_lr=10e-7)
        optimizer_dict = {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
        return optimizer_dict

    def train_dataloader(self):
        dataset = TensorDataset(torch.FloatTensor(self.train_x), torch.LongTensor(self.train_y))
        train_loader = DataLoader(dataset, batch_size=16, num_workers=8, shuffle=True)
        return train_loader

    def val_dataloader(self):
        val_dataset = TensorDataset(torch.FloatTensor(self.valid_x), torch.LongTensor(self.valid_y))
        val_loader = DataLoader(val_dataset, batch_size=16, num_workers=8, shuffle=False)
        return val_loader

    def training_epoch_end(self, training_step_outputs):
        print(f"GOOD")
    #
    # def validation_epoch_end(self, validation_step_outputs):
    #     # compute metrics
    #     val_loss = torch.tensor(validation_step_outputs).mean()
    #     self.log("val_loss", val_loss)


class Action_trigger(object):
    """
        Trigger the salient action 1 or 2 during inference.
        This is used to calculate the repetitive count.
    """
    def __init__(self, action_name, enter_threshold=0.8, exit_threshold=0.4):
        self._action_name = action_name

        # If the score larger than the given enter_threshold, then that pose will enter the triggering.
        # If the score smaller than the given exit_threshold, then that pose will complete the triggering.
        self._enter_threshold = enter_threshold
        self._exit_threshold = exit_threshold

        # Whether the pose has entered the triggering.
        self._pose_entered = False

    def __call__(self, pose_score):
        # We use two thresholds.
        # First, you need to enter the pose from a higher position above,
        # and then you need to exit from a lower position below.
        # The difference between the thresholds makes it stable against prediction jitter
        # (which would lead to false counts if there was only one threshold).

        triggered = False

        # On the very first frame or if we were out of the pose,
        # just check if we entered it on this frame and update the state.
        if not self._pose_entered:
            self._pose_entered = pose_score > self._enter_threshold
            return triggered

        # If we are in a pose and are exiting it, update the state.
        if pose_score < self._exit_threshold:
            self._pose_entered = False
            triggered = True

        return triggered
