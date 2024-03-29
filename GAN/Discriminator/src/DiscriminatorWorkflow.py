import sys
import os
import argparse

if os.name != 'nt':
    sys.path.append(os.path.join(os.pardir, os.pardir, os.pardir))

from datetime import datetime
import pytz
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
import torch


from GAN.Discriminator.src.Discriminator import Discriminator
from GAN.Discriminator.src.DiscriminatorGPT import DiscriminatorGPT
from GAN.Discriminator.src.DiscriminatorBERT import DiscriminatorBioElectra, DiscriminatorSciBert, DiscriminatorTinyBert, DiscriminatorLinkBert
from GAN.GANPubMed.src.PubMedModule import PubMedModule
from GAN.Discriminator.src.hparams_config import hparams

"""This workflow run show how to conduct and experiment using 'Discriminator' 
"""


def ParseCLI():
    parser = argparse.ArgumentParser(description='model arguments')
    parser.add_argument("-n", "--name", type=str, help="name for wandb", default='Discriminator_over_topic_and_gender_70_15_15_SBERT')
    parser.add_argument("--max-epochs", type=int, help="max_epochs", required=False)
    parser.add_argument("--test_dataset-size", type=float, help="test_size", required=False)
    parser.add_argument("--lr", type=float, help="Learning rate", required=False)
    parser.add_argument("--drop-rate", type=float, help="Drop rate", required=False)
    parser.add_argument(
        "--hidden-sizes",
        "-H",
        type=int,
        nargs="+",
        help="Output size of hidden linear layers",
        metavar="H",
        required=False,
    )
    parser.add_argument(
        "--similarity_metric",
        "-M",
        choices=['cross_entropy', 'cosine_similarity'],
        help="Which similarity to Create",
        required=False
    )
    args = parser.parse_args()
    return vars(args)


def PrepareArguments():
    new_hparams = hparams.copy()
    cli_args = ParseCLI()
    for key in cli_args:
        if cli_args[key] is not None:
            new_hparams[key] = cli_args[key]
    return new_hparams


def Run():
    new_hparams = PrepareArguments()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    dm = PubMedModule(new_hparams)
    model = DiscriminatorGPT(new_hparams)
    logger = WandbLogger(
        name=model.name,
        version=datetime.now(pytz.timezone('Asia/Jerusalem')).strftime('%y%m%d_%H%M%S.%f'),
        project='Discriminator_test',
        config=new_hparams
    )
    print(model)
    trainer = pl.Trainer(
        gpus=new_hparams['gpus'],
        max_epochs=new_hparams['max_epochs'],
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
