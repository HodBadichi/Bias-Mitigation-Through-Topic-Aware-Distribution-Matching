import os
import sys

if os.name != 'nt':
    sys.path.append(os.path.join(os.pardir, os.pardir, os.pardir))

import pytz
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from GAN.GANPubMed.src.PubMedGAN import PubMedGAN
from GAN.GANPubMed.src.PubMedModule import PubMedModule
from GAN.GANPubMed.src.hparams_config import hparams

"""
workflow for running 'PubMedGAN'
"""


def Run():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    dm = PubMedModule(hparams)
    model = PubMedGAN(hparams)
    logger = WandbLogger(
        name=f'GAN_over_topic_and_gender_70_15_15_v2_all-MiniLM-L6-v2',
        version=datetime.now(pytz.timezone('Asia/Jerusalem')).strftime('%y%m%d_%H%M%S.%f'),
        project='GAN_test',
        config={'lr': hparams['learning_rate'], 'batch_size': hparams['batch_size']}
    )
    trainer = pl.Trainer(
        gpus=hparams['gpus'],
        max_epochs=hparams['max_epochs'],
        logger=logger,
        log_every_n_steps=1,
        accumulate_grad_batches=1,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)


if __name__ == '__main__':
    # to run on Technion lambda server use: sbatch -c 2 --gres=gpu:1 run_on_server.sh -o run.out
    sys.exit(Run())
