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
from allennlp.data.fields import ArrayField, Field, TextField, SequenceLabelField
from allennlp.data.fields import ListField, MetadataField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import Token, PretrainedTransformerTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
# TagSpanType = ((int, int), str)

@DatasetReader.register("transformer_sequence_labelling")
class TransformerSequenceLabellingReader(DatasetReader):
    """

    Parameters
    ----------
    """

    def __init__(self,
                 pretrained_model: str,
                 max_pieces: int = 512,
                 syntax: str = "gec",
                 dataset_dir_out: str = None,
                 model_type: str = None,
                 is_training: bool = True,
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

        self._sample = sample
        self._syntax = syntax
        self._is_training = is_training
        self._dataset_dir_out = dataset_dir_out
        self._model_type = model_type
        if model_type is None:
            for model in ["roberta", "bert", "openai-gpt", "gpt2", "transfo-xl", "xlnet", "xlm"]:
                if model in pretrained_model:
                    self._model_type = model
                    break

    @overrides
    def _read(self, file_path: str):
        self._dataset_cache = None
        if self._dataset_dir_out is not None:
            self._dataset_cache = []
        instances = self._read_internal(file_path)
        if self._dataset_cache is not None:
            if not isinstance(instances, list):
                instances = [instance for instance in Tqdm.tqdm(instances)]
            if not os.path.exists(self._dataset_dir_out):
                os.mkdir(self._dataset_dir_out)
            output_file = os.path.join(self._dataset_dir_out, os.path.basename(file_path))
            logger.info(f"Saving contextualized dataset to {output_file}.")
            with open(output_file, 'w') as file:
                for d in self._dataset_cache:
                    file.write(json.dumps(d))
                    file.write("\n")
        return instances

    def _read_internal(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        counter = self._sample + 1
        debug = 5

        with open(file_path,"r") as reader:
            data = json.load(reader)

        for item in data["data"]:
            item_id = item["id"]
            sent = item["tokens"]
            UR = item["UR"]
            M = item["M"]

            if debug > 0:
                logger.info(item)
                debug -= 1

            yield self.text_to_instance(
                    item_id=item_id,
                    sent=sent,
                    UR=UR,
                    M=M,
                    debug=debug)

    @overrides
    def text_to_instance(self,  # type: ignore
                         item_id: int,
                         sent: str,
                         UR: str = None,
                         M: str = None,
                         debug: int = -1) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        
        tokens, segment_ids, token_to_orig_map = self.transformer_features_from_sequence(sent)
        tokens_field = TextField(tokens, self._token_indexers)
        #segment_ids_field = SequenceLabelField(segment_ids, tokens_field)
        if debug > 0:
            logger.info(f"tokens = {tokens}")
            logger.info(f"segment_ids = {segment_ids}")

        fields['tokens'] = tokens_field
        #fields['segment_ids'] = segment_ids_field

        metadata = {
            "id": item_id,
            "sent": sent,
            "tokens": [x.text for x in tokens],
            "token_to_orig_map": token_to_orig_map
        }

        if UR is not None and M is not None:
            UR = UR.split()
            M = M.split()
            nUR = ["O" for x in tokens]
            nM = ["O" for x in tokens]
            nUR[0] = UR[0]
            nM[0] = M[0]
            for k, v in token_to_orig_map.items():
                nUR[k] = UR[v+1]
                nM[k] = M[v+1]

            fields['UR_tags'] = SequenceLabelField(nUR, tokens_field, label_namespace="ur_tags")
            fields['M_tags'] = SequenceLabelField(nM, tokens_field, label_namespace="m_tags")
            metadata['UR_tags'] = nUR
            metadata['M_tags'] = nM

        if debug > 0:
            logger.info(f"tokens = {tokens}")
            #logger.info(f"segment_ids = {segment_ids}")
            logger.info(f"UR_tags = {nUR}")
            logger.info(f"M_tags = {nM}")

        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)

    @staticmethod
    def _truncate_tokens(context_tokens, question_tokens, choice_tokens, max_length):
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


    def transformer_features_from_sequence(self, sent: str):
        cls_token = Token(self._tokenizer_internal.cls_token)
        sep_token = Token(self._tokenizer_internal.sep_token)
        #pad_token = self._tokenizer_internal.pad_token
        sep_token_extra = bool(self._model_type in ['roberta'])
        cls_token_at_end = bool(self._model_type in ['xlnet'])
        cls_token_segment_id = 2 if self._model_type in ['xlnet'] else 0
        sequence_a_segment_id = 0
        sequence_b_segment_id = 1
        #pad_on_left = bool(self._model_type in ['xlnet'])
        #pad_token_segment_id = 4 if self._model_type in ['xlnet'] else 0
        #pad_token_val=self._tokenizer.encoder[pad_token] if self._model_type in ['roberta'] else self._tokenizer.vocab[pad_token]
        
        sent = sent.split()

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(sent):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = self._tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)


        tokens = []
        token_to_orig_map = {}
        token_is_max_context = {}
        segment_ids = []

        if not cls_token_at_end:
            tokens.append(cls_token)
            segment_ids.append(cls_token_segment_id)
            cls_index = 0

        for i in range(len(all_doc_tokens)):
            split_token_index = i
            token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]
            tokens.append(all_doc_tokens[split_token_index])
            segment_ids.append(sequence_a_segment_id)

        # SEP token - won't worry about two tokens for Roberta here
        tokens.append(sep_token)
        segment_ids.append(sequence_b_segment_id)

            # CLS token at the end
        if cls_token_at_end:
            tokens.append(cls_token)
            segment_ids.append(cls_token_segment_id)
            cls_index = len(tokens) - 1  # Index of classification token


        return tokens, segment_ids, token_to_orig_map
