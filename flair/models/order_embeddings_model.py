from typing import List, Union

import torch

import flair.embeddings
import flair.nn
from flair.data import Sentence, TextPair, DT
from flair.datasets import DataLoader, FlairDatapointDataset
from flair.training_utils import store_embeddings

from tqdm import tqdm
import numpy.random as random

from typing import List, Tuple, Union, Optional


class OrderEmbeddingsModel(flair.nn.DefaultClassifier[TextPair]):
    def __init__(
        self,
        token_embeddings: flair.embeddings.TokenEmbeddings,
        label_type: str,
        positive_samples_label_name: str,
        negative_samples_label_name: str = "negative",
        use_existing_negative_samples_label_names: List[str] =None,
        margin : float = 1.0,
        create_negative_samples : bool = True,
        **classifierargs,
    ):
        """
        Initializes a OrderEmbeddingsModel.
        :param token_embeddings: embeddings used to embed each data point
        :param positive_samples_label_name: how the positive samples are labeled, eg. "hypernymy"
        :param negative_samples_label_name: how the negative samples will be labeld, eg. "negatove" or "random"
        :param use_existing_negative_samples_label_names: if they are labels in Dataset that are to be used as negative examples (eg. another relation type, maybe "meronymy")
        :param create_negative_samples: whether negative samples should be created per positive example in batch (via random corruption)
        :param margin

        """
        super().__init__(
            **classifierargs,
            final_embedding_size=2 * token_embeddings.embedding_length

        )

        self.token_embeddings: flair.embeddings.TokenEmbeddings = token_embeddings

        self._label_type = label_type

        self.positive_samples_label_name = positive_samples_label_name
        self.negative_samples_label_name = negative_samples_label_name
        self.use_existing_negative_samples_label_names = use_existing_negative_samples_label_names
        self.margin = margin
        self.penalty_threshold = self.margin
        self.create_negative_samples = create_negative_samples

        self.to(flair.device)

        del self.decoder  # decoder from parent class (DefaultClassifier) not in use, so delete it

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

        if self.use_existing_negative_samples_label_names == None:

            # filter out negatives (if they are in the corpus)
            ids_positive_in_batch = []
            ids_to_delete = []
            for i, d in enumerate(datapairs):
                if d.get_labels(self.label_type)[0].value == self.positive_samples_label_name:
                    ids_positive_in_batch.append(i)

                # TODO: debug this! ids change during deletion!
                #else:
                #    ids_to_delete.append(i)
                #    datapairs.remove(datapairs[i])

        else:
            ids_positive_in_batch = []
            # set all negative samples to having the same label name
            for i, d in enumerate(datapairs):
                if d.get_labels(self.label_type)[0].value in self.use_existing_negative_samples_label_names:
                    d.set_label(self.label_type, value = self.negative_samples_label_name, score = 1.0)
                else:
                    ids_positive_in_batch.append(i)

        embedding_names = self.token_embeddings.get_names()

        # embed both sentences separately
        first_elements = [pair.first for pair in datapairs]
        second_elements = [pair.second for pair in datapairs]

        # make random negative samples from the positive ones (not really necessary if there are existing negatives that are used (above))
        if self.create_negative_samples:
            corrupted_first = []
            corrupted_second = []
            replace_which = random.choice(["first", "second"], size=len(ids_positive_in_batch)) # to decide which to replace
            replace_which = zip(ids_positive_in_batch, replace_which)
            #print(replace_which)

            for (i, which) in replace_which:
                # get a random word from dict (to replace one of the true concepts with it)
                random_from_dict = random.choice(self.token_embeddings.vocab_dictionary.idx2item).decode("utf-8")
                # TODO: like so, could happen that some REAL pairs get created but labeled as negative... can I prevent that?
                # also: could be a pair like ("frog","frog", "random") which is bad
                # could look if the random pair IS in the dataset, then use the positive label (to prevent inserting noise)

                if which == "first":
                    corrupted_first.append(Sentence(random_from_dict, use_tokenizer=False)) # replace first with random
                    corrupted_second.append(second_elements[i])

                if which == "second":
                    corrupted_first.append(first_elements[i])
                    corrupted_second.append(Sentence(random_from_dict, use_tokenizer=False)) # replace second with random

            # make datapairs out of them (with the negative label name)
            corrupted_pairs = [flair.data.DataPair(corrupted_first[i], corrupted_second[i])
                                    .add_label(self.label_type, value=self.negative_samples_label_name)
                                for i in range(len(corrupted_first))]

            # and extend the original datapairs with them
            datapairs.extend(corrupted_pairs)

            # append the created negative examples:
            first_elements.extend(corrupted_first)
            second_elements.extend(corrupted_second)

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

        loss = self._calculate_loss(embedded_first, embedded_second, labels)

        #for s,vec1,vec2,loss in zip(sentences, embedded_first, embedded_second, loss):
        #    print(s.text, s.tag, vec1[:5], vec2[:5], loss)
        return loss

    def order_violation_penalty(self, vector1, vector2):
        """
        computes order violation as proposed by Vendrov et al. 2016
        ordered pair is (vector1, vector2), e.g. (dog, animal)
        vector1 (hyponym) should be longer (higher value in each dimension) than vector2 (hypernym)
        so: calculate difference vector vector2-vector1
        set each element where vector1 is bigger than vector2 to 0 so that they don't get punished (because this is the "right" order)
        """
        diff_vector = vector2-vector1
        maxed_vector = torch.clamp(diff_vector, min=0)
        # confusing max/min notations!: in the paper, they use max(0, diff), meaning (I think) "Take 0 in case diff is negative"
        # here we use min(), meaning "clamp in a range with min==0" which should be the same think
        penalty_score = (torch.linalg.norm(maxed_vector, ord=2))**2 #todo: is this the right norm? ord=1 or ord=2? ||.||
        # see:
        # https://stackoverflow.com/questions/68489765/what-is-the-correct-way-to-calculate-the-norm-1-norm-and-2-norm-of-vectors-in-pytorch

        return penalty_score

    def _calculate_loss(self, embedded_first, embedded_second, labels) -> Tuple[torch.Tensor, int]:
        """
        using the order violation penalty from Vendrov et al. 2016, equation 2
        """
        loss_sum = torch.zeros(1, device=flair.device)
        for vector1, vector2, label in zip(embedded_first, embedded_second, labels):
            penalty_score = self.order_violation_penalty(vector1, vector2)
            if label[0] == self.positive_samples_label_name: # todo: labels[0] not very pretty (labels is list of lists, but here just one each)
                loss_sample = penalty_score
            elif label[0] == self.negative_samples_label_name:
                loss_sample = torch.max(torch.zeros(1, device=flair.device),
                                        self.margin - penalty_score)
            else:
                loss_sample = 0
            #print(penalty_score, label, loss_sample)
            loss_sum += loss_sample
        return loss_sum, len(labels)

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
                    embedded_first = pair_embedding_tensor[:, :self.token_embeddings.embedding_length]
                    embedded_second = pair_embedding_tensor[:, self.token_embeddings.embedding_length:]

                    penalty_scores = [ self.order_violation_penalty(vector1, vector2)
                                       for vector1, vector2 in zip(embedded_first, embedded_second) ]

                    #print(penalty_scores)

                    # remove previously predicted labels of this type
                    for data_point in data_points:
                        data_point.remove_labels(label_name)

                    if return_loss:
                        overall_loss += self._calculate_loss(embedded_first, embedded_second, gold_labels)[0]
                        label_count += len(data_points)

                    #TODO: how to get a sensible "score"? sth. like distance to threshold...?
                    # penalty_score is between [0, inf], threshold at 1.0

                    for data_point, penalty_score in zip(data_points, penalty_scores):
                        if penalty_score > self.penalty_threshold:
                            label_value = self.negative_samples_label_name
                        else:
                            label_value = self.positive_samples_label_name
                        score = penalty_score # TODO: can be [0, inf], so not really "score"... think about better solution
                        data_point.add_label(typename=label_name, value=label_value, score=score)
                        #print(data_point, label_value, score)

                store_embeddings(batch, storage_mode=embedding_storage_mode)

            if return_loss:
                return overall_loss, label_count

    def _print_predictions(self, batch, gold_label_type):
        lines = []
        for datapoint in batch:
            # check if there is a label mismatch
            g = [label.labeled_identifier for label in datapoint.get_labels(gold_label_type)]
            p = [label.labeled_identifier for label in datapoint.get_labels("predicted")]
            g.sort()
            p.sort()
            correct_string = " -> ❌\n" if g != p else " -> ✓\n"
            # print info
            eval_line = (
                f"{datapoint.text}\n"
                f" - Gold: {', '.join(label.value if label.data_point == datapoint else label.labeled_identifier for label in datapoint.get_labels(gold_label_type))}\n"
                f" - Pred: {', '.join(label.value if label.data_point == datapoint else label.labeled_identifier for label in datapoint.get_labels('predicted'))}\t{correct_string}\n"
            )
            lines.append(eval_line)
        return lines

    def _get_state_dict(self):
        model_state = {
            **super()._get_state_dict(),
            "token_embeddings": self.token_embeddings,
            "label_dictionary": self.label_dictionary,
            "label_type": self.label_type,
            "weight_dict": self.weight_dict,
            "positive_samples_label_name": self.positive_samples_label_name,
            "negative_samples_label_name": self.negative_samples_label_name,
            "margin": self.margin,
            "create_negative_samples": self.create_negative_samples,
        }
        return model_state

    @classmethod
    def _init_model_with_state_dict(cls, state, **kwargs):
        return super()._init_model_with_state_dict(
            state,
            token_embeddings=state["token_embeddings"],
            label_dictionary=state["label_dictionary"],
            label_type=state["label_type"],
            loss_weights=state["weight_dict"],
            positive_samples_label_name=state["positive_samples_label_name"],
            negative_samples_label_name=state["negative_samples_label_name"],
            margin=state["margin"],
            create_negative_samples=state["create_negative_samples"],
            **kwargs,
        )
