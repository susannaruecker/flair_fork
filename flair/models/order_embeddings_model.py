from typing import List, Union

import torch

import flair.embeddings
import flair.nn
from flair.data import Sentence, TextPair, DT
from flair.datasets import DataLoader, FlairDatapointDataset
from flair.training_utils import store_embeddings

from tqdm import tqdm


#import flair.nn.model

from typing import List, Tuple, Union, Optional



class OrderEmbeddingsModel(flair.nn.DefaultClassifier[TextPair]):
    """
    #todo
    """

    def __init__(
        self,
        token_embeddings: flair.embeddings.TokenEmbeddings,
        label_type: str,
        positive_samples_label_name: str = "hyper",
        negative_samples_label_name: str = "random",
        margin : float = 1.0,
        **classifierargs,
    ):
        """
        Initializes a TextClassifier
        :param token_embeddings: embeddings used to embed each data point # TODO: Achtung, gerade mache ich Token-Level, wäre DocEmbs besser?
        :param label_dictionary: dictionary of labels you want to predict
        :param positive_samples_label_name
        :param negative_samples_label_name
        :param margin

        """
        super().__init__(
            **classifierargs,
            #final_embedding_size=token_embeddings.embedding_length,
            final_embedding_size=2 * token_embeddings.embedding_length

        )

        self.token_embeddings: flair.embeddings.TokenEmbeddings = token_embeddings

        self._label_type = label_type

        self.positive_samples_label_name = positive_samples_label_name
        self.negative_samples_label_name = negative_samples_label_name
        self.margin = margin

        self.to(flair.device)

    @property
    def label_type(self):
        return self._label_type

    def forward_pass(
        self,
        datapairs: Union[List[TextPair], TextPair],
        for_prediction: bool = False,
    ):

        if not isinstance(datapairs, list):
            datapairs = [datapairs]

        embedding_names = self.token_embeddings.get_names()

        # embed both sentences separately
        first_elements = [pair.first for pair in datapairs]
        second_elements = [pair.second for pair in datapairs]

        self.token_embeddings.embed(first_elements)
        self.token_embeddings.embed(second_elements)

        pair_embedding_list = [
            torch.cat(
                [
                    a.tokens[0].get_embedding(embedding_names),
                    b.tokens[0].get_embedding(embedding_names),
                ],
                0,
            ).unsqueeze(0)
            for (a, b) in zip(first_elements, second_elements)
        ]

        pair_embedding_tensor = torch.cat(pair_embedding_list, 0).to(flair.device)

        labels = []
        for pair in datapairs:
            labels.append([label.value for label in pair.get_labels(self.label_type)])

        if for_prediction:
            return pair_embedding_tensor, labels, datapairs

        # Note: first return is concatenated embeddings! So need to cut again for penalty calculation
        return pair_embedding_tensor, labels

    def forward_loss(self, sentences: Union[List[DT], DT]) -> Tuple[torch.Tensor, int]:
        # make a forward pass to produce embedded data points and labels
        pair_embedding_tensor, labels = self.forward_pass(sentences)
        embedded_first = pair_embedding_tensor[:,:self.token_embeddings.embedding_length]
        embedded_second = pair_embedding_tensor[:,self.token_embeddings.embedding_length:]

        # no loss can be calculated if there are no labels
        if not any(labels):
            return torch.tensor(0.0, requires_grad=True, device=flair.device), 1

        # calculate the loss (the order violation penalty from Vendrov et al. 2016)
        return self._calculate_loss(embedded_first, embedded_second, labels)

    def order_violation_penalty(self, vector1, vector2):
        """
        computes order violation as proposed by Vendrov et al. 2016
        ordered pair is (vector1, vector2)
        """
        maxed_vector = torch.clamp((vector2-vector1), max=0) # todo: ist das die richtige Richtung so?
        penalty_score = (maxed_vector.norm(p=2))**2
        # see:
        # https://stackoverflow.com/questions/68489765/what-is-the-correct-way-to-calculate-the-norm-1-norm-and-2-norm-of-vectors-in

        return penalty_score

    def _calculate_loss(self, embedded_first, embedded_second, labels) -> Tuple[torch.Tensor, int]:
        """
        using the order violation penalty from Vendrov et al. 2016, equation 2
        """
        loss_sum = torch.zeros(1, device=flair.device)
        for vector1, vector2, label in zip(embedded_first, embedded_second, labels):
            penalty_score = self.order_violation_penalty(vector1, vector2)
            if label[0] == self.positive_samples_label_name: # todo: labels[0] not very pretty (labels is list of lists, but here just one each)
                loss_sum += penalty_score
            elif label[0] == self.negative_samples_label_name:
                loss_sum += max(0, self.margin - penalty_score)
        return loss_sum

### TODO: make own predict method? because in the DefaultClassifier one the _calculaate_loss is used, but "wrongly"

    def predict(
        self,
        sentences: Union[List[DT], DT],
        mini_batch_size: int = 32,
        return_probabilities_for_all_classes: bool = False,
        verbose: bool = False,
        label_name: Optional[str] = None,
        return_loss=False,
        embedding_storage_mode="none",
    ):
        """
        Predicts the class labels for the given sentences. The labels are directly added to the sentences.  # noqa: E501
        :param sentences: list of sentences
        :param mini_batch_size: mini batch size to use
        :param return_probabilities_for_all_classes : return probabilities for all classes instead of only best predicted  # noqa: E501
        :param verbose: set to True to display a progress bar
        :param return_loss: set to True to return loss
        :param label_name: set this to change the name of the label type that is predicted  # noqa: E501
        :param embedding_storage_mode: default is 'none' which is always best. Only set to 'cpu' or 'gpu' if you wish to not only predict, but also keep the generated embeddings in CPU or GPU memory respectively.  # noqa: E501
        'gpu' to store embeddings in GPU memory.
        """
        if label_name is None:
            label_name = self.label_type if self.label_type is not None else "label"

        with torch.no_grad():
            if not sentences:
                return sentences

            if not isinstance(sentences, list):
                sentences = [sentences]

            reordered_sentences = self._sort_data(sentences)

            if len(reordered_sentences) == 0:
                return sentences

            if len(sentences) > mini_batch_size:
                batches: Union[DataLoader, List[List[DT]]] = DataLoader(
                    dataset=FlairDatapointDataset(reordered_sentences),
                    batch_size=mini_batch_size,
                )
                # progress bar for verbosity
                if verbose:
                    progress_bar = tqdm(batches)
                    progress_bar.set_description("Batch inference")
                    batches = progress_bar
            else:
                batches = [reordered_sentences]

            overall_loss = torch.zeros(1, device=flair.device)
            label_count = 0
            for batch in batches:
                # stop if all sentences are empty
                if not batch:
                    continue

                pair_embedding_tensor, gold_labels, data_points = self.forward_pass(  # type: ignore
                    batch, for_prediction=True
                )
                # if anything could possibly be predicted
                if len(data_points) > 0:
                    #scores = self.decoder(embedded_data_points)
                    embedded_first = pair_embedding_tensor[:, :self.token_embeddings.embedding_length]
                    embedded_second = pair_embedding_tensor[:, self.token_embeddings.embedding_length:]

                    penalty_scores = [ self.order_violation_penalty(vector1, vector2)
                                       for vector1, vector2 in zip(embedded_first, embedded_second) ]

                    #print(penalty_scores)
                    print("here am I!")
                    ### TODO:
                    # threshold definieren, ob dem hypernym ja nein?
                    # das ist ja dann statt decoder...
                    # _calculate_loss unten ändern

                    # remove previously predicted labels of this type
                    for data_point in data_points:
                        data_point.remove_labels(label_name)

                    if return_loss:
                        overall_loss += self._calculate_loss(embedded_first, embedded_second, gold_labels)
                        label_count += len(data_points)

                    # TODO: das hier zu Verwendung mit Threshold ändern
                    #softmax = torch.nn.functional.softmax(scores, dim=-1)

                    #if return_probabilities_for_all_classes:
                    #    n_labels = softmax.size(1)
                    #    for s_idx, data_point in enumerate(data_points):
                    #        for l_idx in range(n_labels):
                    #            label_value = self.label_dictionary.get_item_for_index(l_idx)
                    #            if label_value == "O":
                    #                continue
                    #            label_score = softmax[s_idx, l_idx].item()
                    #            data_point.add_label(typename=label_name, value=label_value, score=label_score)
                    #else:
                    #    conf, idx = torch.max(softmax, dim=-1)
                    #    for data_point, c, i in zip(data_points, conf, idx):
                    #        label_value = self.label_dictionary.get_item_for_index(i.item())
                    #        if label_value == "O":
                    #            continue
                    #        data_point.add_label(typename=label_name, value=label_value, score=c.item())

                store_embeddings(batch, storage_mode=embedding_storage_mode)

            if return_loss:
                return overall_loss, label_count


    def _get_state_dict(self):
        model_state = {
            **super()._get_state_dict(),
            "document_embeddings": self.document_embeddings,
            "label_dictionary": self.label_dictionary,
            "label_type": self.label_type,
            "multi_label": self.multi_label,
            "multi_label_threshold": self.multi_label_threshold,
            "weight_dict": self.weight_dict,
            "embed_separately": self.embed_separately,
        }
        return model_state

    @classmethod
    def _init_model_with_state_dict(cls, state, **kwargs):
        return super()._init_model_with_state_dict(
            state,
            document_embeddings=state["document_embeddings"],
            label_dictionary=state["label_dictionary"],
            label_type=state["label_type"],
            multi_label=state["multi_label"],
            multi_label_threshold=0.5
            if "multi_label_threshold" not in state.keys()
            else state["multi_label_threshold"],
            loss_weights=state["weight_dict"],
            embed_separately=state["embed_separately"],
            **kwargs,
        )
