import sys
import os
if os.name != 'nt':
    sys.path.append('/home/mor.filo/nlp_project/')

from datetime import datetime
import pytz
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl

from DistributionMatching.DiscriminatorPubMed.PubMedDiscriminatorPart import PubMedDiscriminator
from DistributionMatching.PubMed.PubMedModule import PubMedModule
from DistributionMatching.DiscriminatorPubMed.discriminator_hparams_config import hparams


def Run():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    dm = PubMedModule(hparams)
    model = PubMedDiscriminator(hparams)
    logger = WandbLogger(name=f'Discriminator_over_topic_and_gender_70_15_15_v2',
                         version=datetime.now(pytz.timezone('Asia/Jerusalem')).strftime('%y%m%d_%H%M%S.%f'),
                         project='Discriminator_test',
                         config=hparams
                         )

    trainer = pl.Trainer(gpus=hparams['gpus'],
                         max_epochs=hparams['max_epochs'],
                         logger=logger,
                         log_every_n_steps=1,
                         accumulate_grad_batches=1,
                         num_sanity_val_steps=0,
                         )
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)


if __name__ == '__main__':
    sys.exit(Run())
