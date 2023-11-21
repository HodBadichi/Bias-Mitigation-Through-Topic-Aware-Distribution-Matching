from sentence_transformers import SentenceTransformer,  models, losses
from sentence_transformers.losses import SoftmaxLoss
import torch
import torch.nn.functional as F
from torch import nn

""" These loss functions are used in PubMedGanSentenceTransformer.py and are copied from Sentence Transformers module and modified to fit our needs"""

class SoftMaxLoss(SoftmaxLoss):
    def forward(self, sentence_embeddings, labels, size_average=False):
        embeddings = [sentence_feature for sentence_feature in sentence_embeddings]

        vectors_concat = []
        vectors_concat.append(embeddings[0])
        vectors_concat.append(embeddings[1])
        vectors_concat.append(torch.abs(embeddings[0] - embeddings[1]))

        features = torch.cat(vectors_concat, 1)

        output = self.classifier(features)


        loss = self.loss_fct(output, labels.view(-1))
        return loss



class OnlineContrastiveLoss(nn.Module):
    """
    Online Contrastive loss. Similar to ConstrativeLoss, but it selects hard positive (positives that are far apart)
    and hard negative pairs (negatives that are close) and computes the loss only for these pairs. Often yields
    better performances than  ConstrativeLoss.

    :param model: SentenceTransformer model
    :param distance_metric: Function that returns a distance between two emeddings. The class SiameseDistanceMetric contains pre-defined metrices that can be used
    :param margin: Negative samples (label == 0) should have a distance of at least the margin value.
    :param size_average: Average by the size of the mini-batch.

    Example::

        from sentence_transformers import SentenceTransformer, LoggingHandler, losses, InputExample
        from torch.utils.data import DataLoader

        model = SentenceTransformer('all-MiniLM-L6-v2')
        train_examples = [
            InputExample(texts=['This is a positive pair', 'Where the distance will be minimized'], label=1),
            InputExample(texts=['This is a negative pair', 'Their distance will be increased'], label=0)]

        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=2)
        train_loss = losses.OnlineContrastiveLoss(model=model)

        model.fit([(train_dataloader, train_loss)], show_progress_bar=True)
    """

    def __init__(self,  distance_metric, margin: float = 0.5):
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.distance_metric = distance_metric

    def forward(self, sentence_embeddings, labels, size_average=False):
        embeddings = [sentence_feature for sentence_feature in sentence_embeddings]

        distance_matrix = self.distance_metric(embeddings[0], embeddings[1])
        negs = distance_matrix[labels == 0]
        poss = distance_matrix[labels == 1]

        # select hard positive and hard negative pairs
        negative_pairs = negs[negs < (poss.max() if len(poss) > 1 else negs.mean())]
        positive_pairs = poss[poss > (negs.min() if len(negs) > 1 else poss.mean())]

        positive_loss = positive_pairs.pow(2).sum()
        negative_loss = F.relu(self.margin - negative_pairs).pow(2).sum()
        loss = positive_loss + negative_loss
        total_samples = len(positive_pairs) + len(negative_pairs)
        return loss / total_samples
    
class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/ContrastiveLoss.py
    """
    def __init__(self,  distance_metric, margin: float = 0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.distance_metric = distance_metric

    def forward(self, sentence_embeddings, labels, size_average=False):
        embeddings = [sentence_feature for sentence_feature in sentence_embeddings]

        distance_matrix = self.distance_metric(embeddings[0], embeddings[1])
        contrastive_loss = 0.5 * (labels.float() * distance_matrix.pow(2) + (1 - labels).float() * F.relu(self.margin - distance_matrix).pow(2))
        contrastive_loss = contrastive_loss.sum()
        total_samples = len(labels)
        return contrastive_loss / total_samples

class CLMLoss(nn.Module):
    def __init__(self, model):
        super(CLMLoss, self).__init__()
        self.model = list(model._modules.items())[0][1] # get the transformer model from sentence transformer wrapper

    def forward(self, inputs):
        outputs = self.model.auto_model(**inputs)
        # loss is CLM loss because of the CLM head we use when creating the transformer model for sentence transformers to use (GPT2LMHeadModel/BioGptForCausalLM)
        # TODO: this breaks when we try to load the GPT-based sentence transformer model from a checkpoint because it doesn't have the CLM head - need to fix this
        return outputs.loss