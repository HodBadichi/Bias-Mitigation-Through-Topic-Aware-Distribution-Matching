import sys
import os

from datetime import datetime
import pytz
from pytorch_lightning.loggers import WandbLogger
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
        name='Discriminator_over_topic_and_gender_70_15_15_SBERT',
        version=datetime.now(pytz.timezone('Asia/Jerusalem')).strftime('%y%m%d_%H%M%S.%f'),
        project='Discriminator_test',
        config=new_hparams
    )

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
    sys.exit(Run())
