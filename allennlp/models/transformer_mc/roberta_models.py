from typing import Dict, Optional, List, Any

import logging
from overrides import overrides
from pytorch_transformers.modeling_roberta import RobertaClassificationHead, RobertaConfig, RobertaModel
from pytorch_transformers.tokenization_gpt2 import bytes_to_unicode
import re
import torch
from torch.nn.modules.linear import Linear
from torch.nn.functional import binary_cross_entropy_with_logits

from allennlp.common.params import Params
from allennlp.data import Vocabulary
from allennlp.models.archival import load_archive
from allennlp.models.model import Model
from allennlp.models.reading_comprehension.util import get_best_span
from allennlp.nn import RegularizerApplicator, util
from allennlp.training.metrics import BooleanAccuracy, CategoricalAccuracy, SquadEmAndF1, F1Measure, FBetaMeasure
from allennlp.modules.token_embedders import Embedding

@Model.register("roberta_mc_qa")
class RobertaMCQAModel(Model):
    """

    """
    def __init__(self,
                 vocab: Vocabulary,
                 pretrained_model: str = None,
                 requires_grad: bool = True,
                 top_layer_only: bool = True,
                 transformer_weights_model: str = None,
                 reset_classifier: bool = False,
                 per_choice_loss: bool = False,
                 layer_freeze_regexes: List[str] = None,
                 mc_strategy: str = None,
                 on_load: bool = False,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        if on_load:
            logging.info(f"Skipping loading of initial Transformer weights")
            transformer_config = RobertaConfig.from_pretrained(pretrained_model)
            self._transformer_model = RobertaModel(transformer_config)

        elif transformer_weights_model:
            logging.info(f"Loading Transformer weights model from {transformer_weights_model}")
            transformer_model_loaded = load_archive(transformer_weights_model)
            self._transformer_model = transformer_model_loaded.model._transformer_model
        else:
            self._transformer_model = RobertaModel.from_pretrained(pretrained_model)

        for name, param in self._transformer_model.named_parameters():
            grad = requires_grad
            if layer_freeze_regexes and grad:
                grad = not any([bool(re.search(r, name)) for r in layer_freeze_regexes])
            param.requires_grad = grad

        transformer_config = self._transformer_model.config

        self._output_dim = transformer_config.hidden_size
        classifier_input_dim = self._output_dim
        classifier_output_dim = 1
        transformer_config.num_labels = classifier_output_dim
        self._classifier = None
        if not on_load and transformer_weights_model \
                and hasattr(transformer_model_loaded.model, "_classifier") \
                and not reset_classifier:
            self._classifier = transformer_model_loaded.model._classifier
            old_dims = (self._classifier.dense.in_features, self._classifier.out_proj.out_features)
            new_dims = (classifier_input_dim, classifier_output_dim)
            if old_dims != new_dims:
                logging.info(f"NOT copying Transformer classifier weights, incompatible dims: {old_dims} vs {new_dims}")
                self._classifier = None
        if self._classifier is None:
            self._classifier = RobertaClassificationHead(transformer_config)

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
        self._debug = 2
        self._padding_value = 1  # The index of the RoBERTa padding token

    def forward(self,
                question: Dict[str, torch.LongTensor],
                segment_ids: torch.LongTensor = None,
                label: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> torch.Tensor:

        self._debug -= 1
        input_ids = question['tokens']

        batch_size = input_ids.size(0)
        num_choices = input_ids.size(1)

        question_mask = (input_ids != self._padding_value).long()

        if self._debug > 0:
            print(f"batch_size = {batch_size}")
            print(f"num_choices = {num_choices}")
            print(f"question_mask = {question_mask}")
            print(f"input_ids.size() = {input_ids.size()}")
            print(f"input_ids = {input_ids}")
            print(f"segment_ids = {segment_ids}")
            print(f"label = {label}")

        # Segment ids are not used by RoBERTa

        transformer_outputs = self._transformer_model(input_ids=util.combine_initial_dims(input_ids),
                                                      # token_type_ids=util.combine_initial_dims(segment_ids),
                                                      attention_mask=util.combine_initial_dims(question_mask))

        cls_output = transformer_outputs[0]

        if self._debug > 0:
            print(f"cls_output = {cls_output}")

        label_logits = self._classifier(cls_output)
        label_logits = label_logits.view(-1, num_choices)

        output_dict = {}
        output_dict['label_logits'] = label_logits

        output_dict['label_probs'] = torch.nn.functional.softmax(label_logits, dim=1)
        output_dict['answer_index'] = label_logits.argmax(1)

        if label is not None:
            loss = self._loss(label_logits, label)
            self._accuracy(label_logits, label)
            output_dict["loss"] = loss

        if self._debug > 0:
            print(output_dict)
        return output_dict


    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            'accuracy': self._accuracy.get_metric(reset),
        }

    @classmethod
    def _load(cls,
              config: Params,
              serialization_dir: str,
              weights_file: str = None,
              cuda_device: int = -1,
              **kwargs) -> 'Model':
        model_params = config.get('model')
        model_params.update({"on_load": True})
        config.update({'model': model_params})
        return super()._load(config=config,
                             serialization_dir=serialization_dir,
                             weights_file=weights_file,
                             cuda_device=cuda_device,
                             **kwargs)


@Model.register("roberta_classifier")
class RobertaClassifierModel(Model):
    """

    """
    def __init__(self,
                 vocab: Vocabulary,
                 pretrained_model: str = None,
                 requires_grad: bool = True,
                 num_labels: int = None,
                 transformer_weights_model: str = None,
                 reset_classifier: bool = False,
                 layer_freeze_regexes: List[str] = None,
                 on_load: bool = False,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        if on_load:
            logging.info(f"Skipping loading of initial Transformer weights")
            transformer_config = RobertaConfig.from_pretrained(pretrained_model)
            self._transformer_model = RobertaModel(transformer_config)

        elif transformer_weights_model:
            logging.info(f"Loading Transformer weights model from {transformer_weights_model}")
            transformer_model_loaded = load_archive(transformer_weights_model)
            self._transformer_model = transformer_model_loaded.model._transformer_model
        else:
            self._transformer_model = RobertaModel.from_pretrained(pretrained_model)

        for name, param in self._transformer_model.named_parameters():
            grad = requires_grad
            if layer_freeze_regexes and grad:
                grad = not any([bool(re.search(r, name)) for r in layer_freeze_regexes])
            param.requires_grad = grad

        transformer_config = self._transformer_model.config

        self._output_dim = transformer_config.hidden_size
        classifier_input_dim = self._output_dim
        self._num_labels = num_labels
        classifier_output_dim = self._num_labels
        transformer_config.num_labels = classifier_output_dim
        self._classifier = None
        if not on_load and transformer_weights_model \
                and hasattr(transformer_model_loaded.model, "_classifier") \
                and not reset_classifier:
            self._classifier = transformer_model_loaded.model._classifier
            old_dims = (self._classifier.dense.in_features, self._classifier.out_proj.out_features)
            new_dims = (classifier_input_dim, classifier_output_dim)
            if old_dims != new_dims:
                logging.info(f"NOT copying Transformer classifier weights, incompatible dims: {old_dims} vs {new_dims}")
                self._classifier = None
        if self._classifier is None:
            self._classifier = RobertaClassificationHead(transformer_config)

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
        self._debug = 2
        self._padding_value = 1  # The index of the RoBERTa padding token

    def forward(self,
                tokens: Dict[str, torch.LongTensor],
                segment_ids: torch.LongTensor = None,
                label: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> torch.Tensor:

        self._debug -= 1
        input_ids = tokens['tokens']

        batch_size = input_ids.size(0)
        num_choices = input_ids.size(1)

        question_mask = (input_ids != self._padding_value).long()

        if self._debug > 0:
            print(f"batch_size = {batch_size}")
            print(f"num_choices = {num_choices}")
            print(f"question_mask = {question_mask}")
            print(f"input_ids.size() = {input_ids.size()}")
            print(f"input_ids = {input_ids}")
            print(f"segment_ids = {segment_ids}")
            print(f"label = {label}")

        # Segment ids are not used by RoBERTa

        transformer_outputs = self._transformer_model(input_ids=input_ids,
                                                      # token_type_ids=segment_ids,
                                                      attention_mask=question_mask)

        cls_output = transformer_outputs[0]

        if self._debug > 0:
            print(f"cls_output = {cls_output}")

        label_logits = self._classifier(cls_output)

        output_dict = {}
        output_dict['label_logits'] = label_logits

        output_dict['label_probs'] = torch.nn.functional.softmax(label_logits, dim=1)
        output_dict['label_predicted'] = label_logits.argmax(1)

        if label is not None:
            loss = self._loss(label_logits, label)
            self._accuracy(label_logits, label)
            output_dict["loss"] = loss

        if self._debug > 0:
            print(output_dict)
        return output_dict


    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            'accuracy': self._accuracy.get_metric(reset),
        }

    @classmethod
    def _load(cls,
              config: Params,
              serialization_dir: str,
              weights_file: str = None,
              cuda_device: int = -1,
              **kwargs) -> 'Model':
        model_params = config.get('model')
        model_params.update({"on_load": True})
        config.update({'model': model_params})
        return super()._load(config=config,
                             serialization_dir=serialization_dir,
                             weights_file=weights_file,
                             cuda_device=cuda_device,
                             **kwargs)


@Model.register("roberta_span_prediction")
class RobertaSpanPredictionModel(Model):
    """

    """
    def __init__(self,
                 vocab: Vocabulary,
                 pretrained_model: str = None,
                 requires_grad: bool = True,
                 transformer_weights_model: str = None,
                 layer_freeze_regexes: List[str] = None,
                 on_load: bool = False,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        if on_load:
            logging.info(f"Skipping loading of initial Transformer weights")
            transformer_config = RobertaConfig.from_pretrained(pretrained_model)
            self._transformer_model = RobertaModel(transformer_config)

        elif transformer_weights_model:
            logging.info(f"Loading Transformer weights model from {transformer_weights_model}")
            transformer_model_loaded = load_archive(transformer_weights_model)
            self._transformer_model = transformer_model_loaded.model._transformer_model
        else:
            self._transformer_model = RobertaModel.from_pretrained(pretrained_model)

        for name, param in self._transformer_model.named_parameters():
            grad = requires_grad
            if layer_freeze_regexes and grad:
                grad = not any([bool(re.search(r, name)) for r in layer_freeze_regexes])
            param.requires_grad = grad

        transformer_config = self._transformer_model.config
        num_labels = 2  # For start/end
        self.qa_outputs = Linear(transformer_config.hidden_size, num_labels)

        # Import GTP2 machinery to get from tokens to actual text
        self.byte_decoder = {v: k for k, v in bytes_to_unicode().items()}

        self._span_start_accuracy = CategoricalAccuracy()
        self._span_end_accuracy = CategoricalAccuracy()
        self._span_accuracy = BooleanAccuracy()
        self._squad_metrics = SquadEmAndF1()
        self._debug = 0
        self._padding_value = 1  # The index of the RoBERTa padding token


    def forward(self,
                tokens: Dict[str, torch.LongTensor],
                segment_ids: torch.LongTensor = None,
                start_positions: torch.LongTensor = None,
                end_positions: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> torch.Tensor:

        self._debug -= 1
        input_ids = tokens['tokens']

        batch_size = input_ids.size(0)
        num_choices = input_ids.size(1)

        tokens_mask = (input_ids != self._padding_value).long()

        if self._debug > 0:
            print(f"batch_size = {batch_size}")
            print(f"num_choices = {num_choices}")
            print(f"tokens_mask = {tokens_mask}")
            print(f"input_ids.size() = {input_ids.size()}")
            print(f"input_ids = {input_ids}")
            print(f"segment_ids = {segment_ids}")
            print(f"start_positions = {start_positions}")
            print(f"end_positions = {end_positions}")

        # Segment ids are not used by RoBERTa

        transformer_outputs = self._transformer_model(input_ids=input_ids,
                                                      # token_type_ids=segment_ids,
                                                      attention_mask=tokens_mask)
        sequence_output = transformer_outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        span_start_logits = util.replace_masked_values(start_logits, tokens_mask, -1e7)
        span_end_logits = util.replace_masked_values(end_logits, tokens_mask, -1e7)
        best_span = get_best_span(span_start_logits, span_end_logits)
        span_start_probs = util.masked_softmax(span_start_logits, tokens_mask)
        span_end_probs = util.masked_softmax(span_end_logits, tokens_mask)
        output_dict = {"start_logits": start_logits, "end_logits": end_logits, "best_span": best_span}
        output_dict["start_probs"] = span_start_probs
        output_dict["end_probs"] = span_end_probs

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            self._span_start_accuracy(span_start_logits, start_positions)
            self._span_end_accuracy(span_end_logits, end_positions)
            self._span_accuracy(best_span, torch.cat([start_positions.unsqueeze(-1), end_positions.unsqueeze(-1)], -1))

            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignored_index)
            # Should we mask out invalid positions here?
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            output_dict["loss"] = total_loss

        # Compute the EM and F1 on SQuAD and add the tokenized input to the output.
        if metadata is not None:
            output_dict['best_span_str'] = []
            output_dict['exact_match'] = []
            output_dict['f1_score'] = []
            output_dict['qid'] = []
            tokens_texts = []
            for i in range(batch_size):
                tokens_text = metadata[i]['tokens']
                tokens_texts.append(tokens_text)
                predicted_span = tuple(best_span[i].detach().cpu().numpy())
                predicted_start = predicted_span[0]
                predicted_end = predicted_span[1]
                predicted_tokens = tokens_text[predicted_start:(predicted_end + 1)]
                best_span_string = self.convert_tokens_to_string(predicted_tokens)
                output_dict['best_span_str'].append(best_span_string)
                answer_texts = metadata[i].get('answer_texts', [])
                exact_match = 0
                f1_score = 0
                if answer_texts:
                    exact_match, f1_score = self._squad_metrics(best_span_string, answer_texts)
                output_dict['exact_match'].append(exact_match)
                output_dict['f1_score'].append(f1_score)
                output_dict['qid'].append(metadata[i]['id'])
            output_dict['tokens_texts'] = tokens_texts

        if self._debug > 0:
            print(f"output_dict = {output_dict}")

        return output_dict

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        text = ''.join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors='replace')
        return text

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        exact_match, f1_score = self._squad_metrics.get_metric(reset)
        return {
            'start_acc': self._span_start_accuracy.get_metric(reset),
            'end_acc': self._span_end_accuracy.get_metric(reset),
            'span_acc': self._span_accuracy.get_metric(reset),
            'em': exact_match,
            'f1': f1_score,
        }

    @classmethod
    def _load(cls,
              config: Params,
              serialization_dir: str,
              weights_file: str = None,
              cuda_device: int = -1,
              **kwargs) -> 'Model':
        model_params = config.get('model')
        model_params.update({"on_load": True})
        config.update({'model': model_params})
        return super()._load(config=config,
                             serialization_dir=serialization_dir,
                             weights_file=weights_file,
                             cuda_device=cuda_device,
                             **kwargs)

@Model.register("roberta_span_reasoning")
class RobertaSpanReasoningModel(Model):
    """

    """
    def __init__(self,
                 vocab: Vocabulary,
                 pretrained_model: str = None,
                 requires_grad: bool = True,
                 transformer_weights_model: str = None,
                 layer_freeze_regexes: List[str] = None,
                 on_load: bool = False,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        if on_load:
            logging.info(f"Skipping loading of initial Transformer weights")
            transformer_config = RobertaConfig.from_pretrained(pretrained_model)
            self._transformer_model = RobertaModel(transformer_config)

        elif transformer_weights_model:
            logging.info(f"Loading Transformer weights model from {transformer_weights_model}")
            transformer_model_loaded = load_archive(transformer_weights_model)
            self._transformer_model = transformer_model_loaded.model._transformer_model
        else:
            self._transformer_model = RobertaModel.from_pretrained(pretrained_model)

        for name, param in self._transformer_model.named_parameters():
            grad = requires_grad
            if layer_freeze_regexes and grad:
                grad = not any([bool(re.search(r, name)) for r in layer_freeze_regexes])
            param.requires_grad = grad

        transformer_config = self._transformer_model.config

        self.embedder = Embedding(
                            num_embeddings=vocab.get_vocab_size('edges'),
                            embedding_dim=transformer_config.hidden_size,
                            padding_index=0)

        self.qa_outputs = Linear(transformer_config.hidden_size, 2)

        # Import GTP2 machinery to get from tokens to actual text
        self.byte_decoder = {v: k for k, v in bytes_to_unicode().items()}

        self._span_start_accuracy = CategoricalAccuracy()
        self._span_end_accuracy = CategoricalAccuracy()
        self._span_accuracy = BooleanAccuracy()
        self._squad_metrics = SquadEmAndF1()
        self._debug = 0
        self._padding_value = 1  # The index of the RoBERTa padding token


    def forward(self,
                tokens: Dict[str, torch.LongTensor],
                segment_ids: torch.LongTensor = None,
                sentence_graph: torch.LongTensor = None,
                chunks: torch.LongTensor = None,
                corefs: torch.LongTensor = None,
                cands: torch.LongTensor = None,
                cands_start: torch.LongTensor = None,
                cands_end: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> torch.Tensor:

        # print(f"chunks: {chunks}")
        # print(f"sentence_graph: {sentence_graph[0][0]}")
        # print(f"corefs: {corefs[0][0]}")
        # print(f"cands: {cands}")
        # print(f"cands_start: {cands_start}")
        # print(f"cands_end: {cands_end}")
        # print(sentence_graph.size())
        # print(corefs.size())
        # print(cands.size())
        # print(metadata[0]["qas_id"])
        # exit()
        self._debug -= 1
        input_ids = tokens['tokens']

        batch_size = input_ids.size(0)
        num_choices = input_ids.size(1)

        tokens_mask = (input_ids != self._padding_value).long()

        # if self._debug > 0:
        #     print(f"batch_size = {batch_size}")
        #     print(f"num_choices = {num_choices}")
        #     print(f"tokens_mask = {tokens_mask}")
        #     print(f"input_ids.size() = {input_ids.size()}")
        #     print(f"input_ids = {input_ids}")
        #     print(f"segment_ids = {segment_ids}")
        #     print(f"start_positions = {start_positions}")
        #     print(f"end_positions = {end_positions}")

        # Segment ids are not used by RoBERTa

        transformer_outputs = self._transformer_model(input_ids=input_ids,
                                                      # token_type_ids=segment_ids,
                                                      attention_mask=tokens_mask)
        sequence_output = transformer_outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        span_start_logits = util.replace_masked_values(start_logits, tokens_mask, -1e7)
        span_end_logits = util.replace_masked_values(end_logits, tokens_mask, -1e7)
        best_span = get_best_span(span_start_logits, span_end_logits)
        span_start_probs = util.masked_softmax(span_start_logits, tokens_mask)
        span_end_probs = util.masked_softmax(span_end_logits, tokens_mask)
        output_dict = {"start_logits": start_logits, "end_logits": end_logits, "best_span": best_span}
        output_dict["start_probs"] = span_start_probs
        output_dict["end_probs"] = span_end_probs

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            self._span_start_accuracy(span_start_logits, start_positions)
            self._span_end_accuracy(span_end_logits, end_positions)
            self._span_accuracy(best_span, torch.cat([start_positions.unsqueeze(-1), end_positions.unsqueeze(-1)], -1))

            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignored_index)
            # Should we mask out invalid positions here?
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            output_dict["loss"] = total_loss

        # Compute the EM and F1 on SQuAD and add the tokenized input to the output.
        if metadata is not None:
            output_dict['best_span_str'] = []
            output_dict['exact_match'] = []
            output_dict['f1_score'] = []
            output_dict['qid'] = []
            tokens_texts = []
            for i in range(batch_size):
                tokens_text = metadata[i]['tokens']
                tokens_texts.append(tokens_text)
                predicted_span = tuple(best_span[i].detach().cpu().numpy())
                predicted_start = predicted_span[0]
                predicted_end = predicted_span[1]
                predicted_tokens = tokens_text[predicted_start:(predicted_end + 1)]
                best_span_string = self.convert_tokens_to_string(predicted_tokens)
                output_dict['best_span_str'].append(best_span_string)
                answer_texts = metadata[i].get('answer_texts', [])
                exact_match = 0
                f1_score = 0
                if answer_texts:
                    exact_match, f1_score = self._squad_metrics(best_span_string, answer_texts)
                output_dict['exact_match'].append(exact_match)
                output_dict['f1_score'].append(f1_score)
                output_dict['qid'].append(metadata[i]['id'])
            output_dict['tokens_texts'] = tokens_texts

        if self._debug > 0:
            print(f"output_dict = {output_dict}")

        return output_dict

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        text = ''.join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors='replace')
        return text

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        exact_match, f1_score = self._squad_metrics.get_metric(reset)
        return {
            'start_acc': self._span_start_accuracy.get_metric(reset),
            'end_acc': self._span_end_accuracy.get_metric(reset),
            'span_acc': self._span_accuracy.get_metric(reset),
            'em': exact_match,
            'f1': f1_score,
        }

    @classmethod
    def _load(cls,
              config: Params,
              serialization_dir: str,
              weights_file: str = None,
              cuda_device: int = -1,
              **kwargs) -> 'Model':
        model_params = config.get('model')
        model_params.update({"on_load": True})
        config.update({'model': model_params})
        return super()._load(config=config,
                             serialization_dir=serialization_dir,
                             weights_file=weights_file,
                             cuda_device=cuda_device,
                             **kwargs)


@Model.register("roberta_sequence_labelling")
class RobertaSequenceLabelingModel(Model):
    """

    """
    def __init__(self,
                 vocab: Vocabulary,
                 pretrained_model: str = None,
                 requires_grad: bool = True,
                 transformer_weights_model: str = None,
                 layer_freeze_regexes: List[str] = None,
                 on_load: bool = False,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        if on_load:
            logging.info(f"Skipping loading of initial Transformer weights")
            transformer_config = RobertaConfig.from_pretrained(pretrained_model)
            self._transformer_model = RobertaModel(transformer_config)

        elif transformer_weights_model:
            logging.info(f"Loading Transformer weights model from {transformer_weights_model}")
            transformer_model_loaded = load_archive(transformer_weights_model)
            self._transformer_model = transformer_model_loaded.model._transformer_model
        else:
            self._transformer_model = RobertaModel.from_pretrained(pretrained_model)

        for name, param in self._transformer_model.named_parameters():
            grad = requires_grad
            if layer_freeze_regexes and grad:
                grad = not any([bool(re.search(r, name)) for r in layer_freeze_regexes])
            param.requires_grad = grad

        transformer_config = self._transformer_model.config
        
        self.UR_outputs = Linear(transformer_config.hidden_size, self.vocab.get_vocab_size(namespace="ur_tags")) #U, R, O
        self.M_outputs = Linear(transformer_config.hidden_size, self.vocab.get_vocab_size(namespace="m_tags")) #M, O
        # Import GTP2 machinery to get from tokens to actual text
        self.byte_decoder = {v: k for k, v in bytes_to_unicode().items()}

        self._UR_accuracy = CategoricalAccuracy()
        self._M_accuracy = CategoricalAccuracy()

        self._U_idx = self.vocab.add_tokens_to_namespace("U", namespace="ur_tags")

        self._R_idx = []
        for i in range(self.vocab.get_vocab_size(namespace="ur_tags")):
            if i == self._U_idx or i == self.vocab.add_tokens_to_namespace("O", namespace="ur_tags"):
                continue
            self._R_idx.append(i)

        self._M_idx = []
        for i in range(self.vocab.get_vocab_size(namespace="m_tags")):
            if i == self.vocab.add_tokens_to_namespace("O", namespace="m_tags"):
                continue
            self._M_idx.append(i)


        self._U_F1 = F1Measure(self._U_idx)

        self._R_F1 = FBetaMeasure(labels=self._R_idx)
        # self._R_F1_micro = FBetaMeasure(labels=self._R_idx, average="micro")
        # self._R_F1_macro = FBetaMeasure(labels=self._R_idx, average="macro")

        self._M_F1 = FBetaMeasure(labels=self._M_idx)
        # self._M_F1_micro = FBetaMeasure(labels=self._M_idx, average="micro")
        # self._M_F1_macro = FBetaMeasure(labels=self._M_idx, average="macro")

        self._debug = 0
        self._padding_value = 1  # The index of the RoBERTa padding token


    def forward(self,
                tokens: Dict[str, torch.LongTensor],
                segment_ids: torch.LongTensor = None,
                UR_tags: torch.LongTensor = None,
                M_tags: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> torch.Tensor:

        self._debug -= 1
        input_ids = tokens['tokens']

        batch_size = input_ids.size(0)
        sequence_length = input_ids.size(1)

        tokens_mask = (input_ids != self._padding_value).long()

        if self._debug > 0:
            print(f"batch_size = {batch_size}")
            print(f"sequence_length = {sequence_length}")
            print(f"tokens_mask = {tokens_mask}")
            print(f"input_ids.size() = {input_ids.size()}")
            print(f"input_ids = {input_ids}")
            print(f"segment_ids = {segment_ids}")
            print(f"UR_tags = {UR_tags}")
            print(f"M_tags = {M_tags}")
        # Segment ids are not used by RoBERTa

        transformer_outputs = self._transformer_model(input_ids=input_ids,
                                                      # token_type_ids=segment_ids,
                                                      attention_mask=tokens_mask)
        sequence_output = transformer_outputs[0]

        UR = self.UR_outputs(sequence_output)
        M = self.M_outputs(sequence_output)

        UR_probs = util.masked_softmax(UR, mask=None)
        M_probs = util.masked_softmax(M, mask=None)

        best_UR = UR_probs.argmax(-1)
        best_M = M_probs.argmax(-1)

        output_dict = {}

        if UR_tags is not None and M_tags is not None:
            # If we are on multi-GPU, split add a dimension

            self._UR_accuracy(UR_probs, UR_tags, tokens_mask)
            self._M_accuracy(M_probs, M_tags, tokens_mask)

            self._U_F1(UR_probs, UR_tags, tokens_mask)
            self._R_F1(UR_probs, UR_tags, tokens_mask)
            # self._R_F1_micro(UR_probs, UR_tags, tokens_mask)
            # self._R_F1_macro(UR_probs, UR_tags, tokens_mask)

            self._M_F1(M_probs, M_tags, tokens_mask)
            # self._M_F1_micro(M_probs, M_tags, tokens_mask)
            # self._M_F1_macro(M_probs, M_tags, tokens_mask)

            
            UR_loss = util.sequence_cross_entropy_with_logits(UR, UR_tags, tokens_mask, average="token")
            M_loss = util.sequence_cross_entropy_with_logits(M, M_tags, tokens_mask, average="token")

            total_loss = (UR_loss + M_loss) / 2
            output_dict["loss"] = total_loss

        # Compute the EM and F1 on SQuAD and add the tokenized input to the output.
        if metadata is not None:
            best_UR = best_UR.cpu().data.numpy()
            best_M = best_M.cpu().data.numpy()
            output_dict['best_UR'] = []
            output_dict['best_M'] = []
            output_dict['qid'] = []
            tokens_texts = []
            for i in range(batch_size):
                tokens_text = metadata[i]['tokens']
                tokens_texts.append(tokens_text)
                output_dict['best_UR'].append(
                    [self.vocab.get_token_from_index(x, namespace="ur_tags") for x in best_UR[i][:len(tokens_text)]]
                    )
                output_dict['best_M'].append(
                    [self.vocab.get_token_from_index(x, namespace="m_tags") for x in best_M[i][:len(tokens_text)]]
                    )
                output_dict['qid'].append(metadata[i]['id'])
            output_dict['tokens_texts'] = tokens_texts

        if self._debug > 0:
            print(f"output_dict = {output_dict}")

        return output_dict

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        text = ''.join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors='replace')
        return text

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:

        results = {'_ur_acc': self._UR_accuracy.get_metric(reset),'_m_acc': self._M_accuracy.get_metric(reset)}

        results['_u_pre'], results['_u_rec'], results['u_f1']  = self._U_F1.get_metric(reset)
        # subresults = self._R_F1_micro.get_metric(reset)
        # results['_r_micro_pre'] = subresults['precision']
        # results['_r_micro_rec'] = subresults['recall']
        # results['r_micro_f1'] = subresults['fscore']

        # subresults = self._R_F1_macro.get_metric(reset)
        # results['_r_macro_pre'] = subresults['precision']
        # results['_r_macro_rec'] = subresults['recall']
        # results['r_macro_f1'] = subresults['fscore']

        # subresults = self._M_F1_micro.get_metric(reset)
        # results['_m_micro_pre'] = subresults['precision']
        # results['_m_micro_rec'] = subresults['recall']
        # results['m_micro_f1'] = subresults['fscore']

        # subresults = self._M_F1_macro.get_metric(reset)
        # results['_m_macro_pre'] = subresults['precision']
        # results['_m_macro_rec'] = subresults['recall']
        # results['m_macro_f1'] = subresults['fscore']     

        subresults = self._R_F1.get_metric(reset)
        p = r = f = 0.0
        for i in self._R_idx:
            results['_'+self.vocab.get_token_from_index(i, namespace="ur_tags")+"_pre"] = subresults["precision"][i]
            results['_'+self.vocab.get_token_from_index(i, namespace="ur_tags")+"_rec"] = subresults["recall"][i]
            results['_'+self.vocab.get_token_from_index(i, namespace="ur_tags")+"_f1"] = subresults["fscore"][i]
            p += subresults["precision"][i]
            r += subresults["recall"][i]
            f += subresults["fscore"][i]
        results['_r_macro_pre'] = p / len(self._R_idx)
        results['_r_macro_rec'] = r / len(self._R_idx)
        results['r_macro_f1'] = f / len(self._R_idx)

        subresults = self._M_F1.get_metric(reset)
        p = r = f = 0.0
        for i in self._M_idx:
            results['_'+self.vocab.get_token_from_index(i, namespace="m_tags")+"_pre"] = subresults["precision"][i]
            results['_'+self.vocab.get_token_from_index(i, namespace="m_tags")+"_rec"] = subresults["recall"][i]
            results['_'+self.vocab.get_token_from_index(i, namespace="m_tags")+"_f1"] = subresults["fscore"][i]
            p += subresults["precision"][i]
            r += subresults["recall"][i]
            f += subresults["fscore"][i]
        results['_m_macro_pre'] = p / len(self._M_idx)
        results['_m_macro_rec'] = r / len(self._M_idx)
        results['m_macro_f1'] = f / len(self._M_idx)
        
        return results

    @classmethod
    def _load(cls,
              config: Params,
              serialization_dir: str,
              weights_file: str = None,
              cuda_device: int = -1,
              **kwargs) -> 'Model':
        model_params = config.get('model')
        model_params.update({"on_load": True})
        config.update({'model': model_params})
        return super()._load(config=config,
                             serialization_dir=serialization_dir,
                             weights_file=weights_file,
                             cuda_device=cuda_device,
                             **kwargs)