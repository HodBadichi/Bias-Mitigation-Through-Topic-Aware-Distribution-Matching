import os
import sys
sys.path.append(os.path.join(os.pardir,os.pardir))

from pytorch_lightning.loggers import WandbLogger
from datetime import datetime
import pytorch_lightning as pl
import pytz

from GAN.FrozenBert.src.FrozenBert import FrozenBert
from GAN.FrozenBert.src.FrozenBertDataModule import FrozenBertDataModule

"""
Workflow for training bert model on the GANPubMed data using the FrozenBertDataModule
"""


def RunWorkflow():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    dm = FrozenBertDataModule()
    model = FrozenBert()
    logger = WandbLogger(
        name=f'bert_over_topic_and_gender_70_15_15',
        version=datetime.now(pytz.timezone('Asia/Jerusalem')).strftime('%y%m%d_%H%M%S.%f'),
        project='Document_embeddings_test',
        config={'lr': 5e-5, 'batch_size': 16}
    )
    trainer = pl.Trainer(
        gpus=1,
        auto_select_gpus=False,
        max_epochs=40,
        logger=logger,
        log_every_n_steps=10,
        accumulate_grad_batches=1,  # no accumulation
        precision=16
    )
    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    # to run on server sbatch -c 2 --gres=gpu:1 run_on_server.sh -o run.out
    sys.exit(RunWorkflow())
