import sys
sys.path.append('/home/mor.filo/nlp_project/')

from DistributionMatching.PubMed.PubMedGAN import PubMedGAN
from DistributionMatching.PubMed.PubMedModule import PubMedModule
from DistributionMatching.PubMed.hparams_config import hparams
from datetime import datetime
import pytz
from pytorch_lightning.loggers import WandbLogger
import sys
import pytorch_lightning as pl


def Run():
    dm = PubMedModule(hparams)
    model = PubMedGAN(hparams)
    logger = WandbLogger(name=f'GAN_over_topic_and_gender_70_15_15_v2',
                         version=datetime.now(pytz.timezone('Asia/Jerusalem')).strftime('%y%m%d_%H%M%S.%f'),
                         project='GAN_test',
                         config={'lr': hparams['learning_rate'], 'batch_size': hparams['batch_size']})
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
