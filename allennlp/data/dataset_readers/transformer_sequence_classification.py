from typing import Dict, List, Any
import itertools
import json
import logging
import numpy
import os
import re

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.common.tqdm import Tqdm
from allennlp.common.util import JsonDict
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.document_retriever import combine_sentences, list_sentences, DocumentRetriever
from allennlp.data.fields import ArrayField, Field, TextField, LabelField
from allennlp.data.fields import ListField, MetadataField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import Token, PretrainedTransformerTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
# TagSpanType = ((int, int), str)

@DatasetReader.register("transformer_sequence_classification")
class TransformerSequenceClassificationReader(DatasetReader):
    """

    Parameters
    ----------
    """

    def __init__(self,
                 pretrained_model: str,
                 max_pieces: int = 512,
                 skip_id_regex: str = None,
                 add_prefix: Dict[str, str] = None,
                 model_type: str = None,
                 do_lowercase: bool = None,
                 sample: int = -1) -> None:
        super().__init__()
        if do_lowercase is None:
            do_lowercase = '-uncased' in pretrained_model

        self._tokenizer = PretrainedTransformerTokenizer(pretrained_model,
                                                         do_lowercase=do_lowercase,
                                                         start_tokens = [],
                                                         end_tokens = [])
        self._tokenizer_internal = self._tokenizer._tokenizer
        token_indexer = PretrainedTransformerIndexer(pretrained_model, do_lowercase=do_lowercase)
        self._token_indexers = {'tokens': token_indexer}

        self._max_pieces = max_pieces
        self._sample = sample
        self._skip_id_regex = skip_id_regex
        self._model_type = model_type
        self._add_prefix = add_prefix or {}
        if model_type is None:
            for model in ["roberta", "bert", "openai-gpt", "gpt2", "transfo-xl", "xlnet", "xlm"]:
                if model in pretrained_model:
                    self._model_type = model
                    break

    @overrides
    def _read(self, file_path: str):
        instances = self._read_internal(file_path)
        return instances

    def _read_internal(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        counter = self._sample + 1
        debug = 5

        with open(file_path, 'r') as data_file:
            logger.info("Reading instances from jsonl dataset at: %s", file_path)
            for line in data_file:
                item_json = json.loads(line.strip())
                item_id = item_json["id"]
                if self._skip_id_regex and re.match(self._skip_id_regex, item_id):
                    continue

                counter -= 1
                debug -= 1
                if counter == 0:
                    break

                if debug > 0:
                    logger.info(item_json)

                context = item_json.get("sequence")
                label = item_json.get("label")

                yield self.text_to_instance(
                    item_id=item_id,
                    label=label,
                    context=context,
                    debug=debug)

    @overrides
    def text_to_instance(self,  # type: ignore
                         item_id: str,
                         label: str,
                         context: str,
                         debug: int = -1) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}

        tokens, segment_ids = self.transformer_features(context)
        tokens_field = TextField(tokens, self._token_indexers)
        segment_ids_field = SequenceLabelField(segment_ids, tokens_field)

        fields['tokens'] = tokens_field
        fields['segment_ids'] = segment_ids_field
        fields['label'] = LabelField(label)

        metadata = {
            "id": item_id,
            "label": label,
            "context": context
        }

        if debug > 0:
            logger.info(f"tokens = {tokens}")
            logger.info(f"segment_ids = {segment_ids}")
            logger.info(f"context = {context}")
            logger.info(f"label = {label}")

        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)

    @staticmethod
    def _truncate_tokens(context_tokens, max_length):
        """
        Truncate context_tokens first, from the left, then question_tokens and choice_tokens
        """
        max_context_len = max_length - len(question_tokens) - len(choice_tokens)
        if max_context_len > 0:
            if len(context_tokens) > max_context_len:
                context_tokens = context_tokens[-max_context_len:]
        else:
            context_tokens = []
            while len(question_tokens) + len(choice_tokens) > max_length:
                if len(question_tokens) > len(choice_tokens):
                    question_tokens.pop(0)
                else:
                    choice_tokens.pop()
        return context_tokens, question_tokens, choice_tokens


    def transformer_features(self, context: str):
        cls_token = Token(self._tokenizer_internal.cls_token)
        sep_token = Token(self._tokenizer_internal.sep_token)
        #pad_token = self._tokenizer_internal.pad_token
        cls_token_segment_id = 0 
        #pad_on_left = bool(self._model_type in ['xlnet'])
        #pad_token_segment_id = 4 if self._model_type in ['xlnet'] else 0
        #pad_token_val=self._tokenizer.encoder[pad_token] if self._model_type in ['roberta'] else self._tokenizer.vocab[pad_token]
        
        context_tokens = []
        for token in context.split():
            sub_tokens = self._tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                context_tokens.append(sub_token)

        max_length = self._max_pieces - 2
        if len(context_tokens) > max_length:
            context_tokens = context_tokens[-max_length:]

        tokens = []
        segment_ids = []

        tokens = [cls_token] + context_tokens + [sep_token]
        segment_ids = [cls_token_segment_id] * len(tokens)

        return tokens, segment_ids
