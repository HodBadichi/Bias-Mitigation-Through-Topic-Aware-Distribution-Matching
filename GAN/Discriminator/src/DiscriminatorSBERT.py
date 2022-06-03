import torch

from sentence_transformers import SentenceTransformer

from GAN.Discriminator.src.Discriminator import Discriminator

"""DiscriminatorSBert Implementation
This class inherits from Discriminator, A basic network with linear layers using RELU between each layer
which tries to detect which one of a documents duo given is a biased and an unbiased one based Sentence Bert embeddings
"""


class DiscriminatorSBert(Discriminator):
    def __init__(self, hparams):
        super(DiscriminatorSBert, self).__init__(hparams)
        self.SentenceTransformerModel = SentenceTransformer('all-MiniLM-L6-v2')
        self.sentence_embedding_size = 384

    def _discriminator_SBERT_embeddings_to_predictions(self, sentence_embeddings):
        sample_embedding = []

        for i in range(0, len(sentence_embeddings), 2):
            first_document_embeddings = sentence_embeddings[i]
            second_document_embeddings = sentence_embeddings[i + 1]
            curr_concat_embeddings = torch.cat((
                first_document_embeddings,
                second_document_embeddings,
                abs(first_document_embeddings - second_document_embeddings))
            )
            sample_embedding.append(curr_concat_embeddings)

        aggregated = torch.stack(sample_embedding)
        y_predictions = self.classifier(aggregated).squeeze(1)
        return y_predictions

    def _discriminator_get_predictions(self, batch, shuffle_vector):
        """
        :param batch: a batch of PubMedGan in the shape of {'origin':int,'biased':string,'unbiased':string}
        might include `None` values in the `biased` and `unbiased` entry in case the origin document has no match.

        :return: This function wil return the classifier predictions over bertmodel output embeddings
        shuffle_vector: list in the shape of [0,...,1,0...] which can be interpreted like that:
        1 - the couple of matching docs was [biased,unbiased]
        0 - the couple of matching docs was [unbiased,biased]
        """

        discriminator_batch = Discriminator._discriminator_get_batch(batch, shuffle_vector)
        sentence_embeddings = self.SentenceTransformerModel.encode(discriminator_batch, convert_to_tensor=True)
        return self._discriminator_SBERT_embeddings_to_predictions(sentence_embeddings)