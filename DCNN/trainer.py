from omegaconf import OmegaConf
import torch

from torch.optim.lr_scheduler import MultiStepLR
from DCNN.models.model import Model
from DCNN.loss import BinauralLoss
from DCNN.utils.base_trainer import (
    BaseTrainer, BaseLightningModule
)


class DCNNTrainer(BaseTrainer):
    def __init__(self, config):
        lightning_module = DCNNLightningModule(config)
        super().__init__(lightning_module,
                         config["training"]["n_epochs"],
                         early_stopping_config=config["training"]["early_stopping"],
                         checkpoint_path=None,
                        #  strategy=config["training"]["strategy"],
                         accelerator=config["training"]["accelerator"])
                        # accelerator='mps')

    def fit(self, train_dataloaders, val_dataloaders=None):
        super().fit(self._lightning_module, train_dataloaders,
                    val_dataloaders=val_dataloaders)

    def test(self, test_dataloaders):
        super().test(self._lightning_module, test_dataloaders, ckpt_path="best")


class DCNNLightningModule(BaseLightningModule):
    """This class abstracts the
       training/validation/testing procedures
       used for training a DCNN
    """

    def __init__(self, config):
        config = OmegaConf.to_container(config)
        self.config = config

        model = Model(**self.config["model"])
        loss = BinauralLoss(
        
            ild_weight=self.config["loss"]["ild_weight"],
            ipd_weight=self.config["loss"]["ipd_weight"],
            
            stoi_weight=self.config["loss"]["stoi_weight"],
            
            snr_loss_weight=self.config["loss"]["snr_loss_weight"],
            
            )

        super().__init__(model, loss)

    def configure_optimizers(self):
        lr = self.config["training"]["learning_rate"]
        decay_step = self.config["training"]["learning_rate_decay_steps"]
        decay_value = self.config["training"]["learning_rate_decay_values"]

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = MultiStepLR(optimizer, decay_step, decay_value)

        return [optimizer], [scheduler]
# ── DCNN/trainer.py ────────────────────
    def _step(
        self,
        batch,
        batch_idx: int,
        log_model_output: bool = False,
        log_labels: bool = True,
    ):
        mix, voice_t, noise_t = batch
        voice, noise, vlc, vrc, nlc, nrc = self.model(mix)

        total_loss, metrics = self.loss(
            voice, noise, vlc, vrc, nlc, nrc,
            voice_t, noise_t
        )

        # 1) Lightning logging
        self.log("loss", total_loss, prog_bar=True, sync_dist=True)
        for k, v in metrics.items():
            self.log(k, v, prog_bar=False, sync_dist=True)

        # 2) 回傳 dict，給 BaseLightningModule 收集做 epoch 統計
        output_dict = {"loss": total_loss}
        output_dict.update(metrics)
        return output_dict

