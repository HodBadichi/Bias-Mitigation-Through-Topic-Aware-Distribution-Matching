import sys
import os
import argparse

if os.name != 'nt':
    sys.path.append('/home/mor.filo/nlp_project/')

from datetime import datetime
import pytz
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl

from DistributionMatching.DiscriminatorPubMed.PubMedDiscriminatorPart import PubMedDiscriminator
from DistributionMatching.PubMed.PubMedModule import PubMedModule
from DistributionMatching.DiscriminatorPubMed.discriminator_hparams_config import hparams


def parse_cli():
    parser = argparse.ArgumentParser(description='model arguments')
    parser.add_argument("--max-epochs", type=int, help="max_epochs", required=False)
    parser.add_argument("--test-size", type=float, help="test_size", required=False)
    parser.add_argument("--lr", type=float, help="Learning rate", required=False)
    parser.add_argument("--drop-rate", type=float, help="Drop rate", required=False)
    parser.add_argument("--hidden-sizes",
                        "-H",
                        type=int,
                        nargs="+",
                        help="Output size of hidden linear layers",
                        metavar="H",
                        required=False, )
    parser.add_argument(
        "--similarity_metric",
        "-M",
        choices=['cross_entropy', 'cosine_similarity'],
        help="Which similarity to create",
        required=False
    )
    args = parser.parse_args()
    return vars(args)


def prepare_arguments():
    cli_args = parse_cli()
    for key in cli_args:
        if cli_args[key] is not None:
            hparams[key] = cli_args[key]

    sim_matrix_str = None
    prob_matrix_str = None
    if hparams['similarity_metric'] == 'cross_entropy':
        reset_str = ["no_reset", "reset"]
        reset_flag = reset_str[hparams['reset_different_topic_entries_flag']]
        prob_matrix_str = f'CE_prob_matrix_{reset_flag}_different_topic_entries_flag_'
        sim_matrix_str = 'CE_sim_matrix_'
    else:
        prob_matrix_str = 'CS_prob_matrix_with_BERTopic_clean_'
        sim_matrix_str = 'CS_sim_matrix_'

    hparams['SimilarityMatrixPathTrain'] = os.path.join(os.pardir, 'PubMed', sim_matrix_str + 'train')
    hparams['ProbabilityMatrixPathTrain'] = os.path.join(os.pardir, 'PubMed', prob_matrix_str + 'train')
    hparams['SimilarityMatrixPathVal'] = os.path.join(os.pardir, 'PubMed', sim_matrix_str + 'val')
    hparams['ProbabilityMatrixPathVal'] = os.path.join(os.pardir, 'PubMed', prob_matrix_str + 'val')
    hparams['SimilarityMatrixPathTest'] = os.path.join(os.pardir, 'PubMed', sim_matrix_str + 'test')
    hparams['ProbabilityMatrixPathTest'] = os.path.join(os.pardir, 'PubMed', prob_matrix_str + 'test')

def Run():
    prepare_arguments()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    dm = PubMedModule(hparams)
    model = PubMedDiscriminator(hparams)
    logger = WandbLogger(
        name='Discriminator_over_topic_and_gender_70_15_15_v2',
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
