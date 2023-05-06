import sys
import os

if os.name != 'nt':
    sys.path.append(os.path.join(os.pardir, os.pardir, os.pardir))

from datetime import datetime
import pytz
from pytorch_lightning.loggers import WandbLogger
import wandb 
import pytorch_lightning as pl

from GAN.Discriminator.src.DiscriminatorSBERT import DiscriminatorSBert
from GAN.GANPubMed.src.PubMedModule import PubMedModule
from GAN.Discriminator.src.DiscriminatorWorkflow import PrepareArguments

"""This workflow run show how to conduct and experiment using 'DiscriminatorSBERT' 
"""


def Run():
    new_hparams = PrepareArguments()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    dm = PubMedModule(new_hparams)
    model = DiscriminatorSBert(new_hparams)
    logger = WandbLogger(
        name=new_hparams['name'],
        version=datetime.now(pytz.timezone('Asia/Jerusalem')).strftime('%y%m%d_%H%M%S.%f'),
        project='Discriminator_test',
        config=new_hparams
    )
    wandb.define_metric("discriminator/val_dataset_loss", summary="mean")

    trainer = pl.Trainer(gpus=new_hparams['gpus'],
                         max_epochs=new_hparams['max_epochs'],
                         logger=logger,
                         log_every_n_steps=1,
                         accumulate_grad_batches=1,
                         num_sanity_val_steps=0,
                         )
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)


if __name__ == '__main__':
    # to run on Technion lambda server use: sbatch -c 2 --gres=gpu:1 run_on_server_sbert.sh -o run.out
    sys.exit(Run())
