from typing import Dict, List, Any
import collections
import itertools
import json
import logging
import numpy
import os
import re
import torch

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.common.tqdm import Tqdm
from allennlp.common.util import JsonDict
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField, SpanField, AdjacencyField, ListField, IndexField
from allennlp.data.fields import MetadataField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import PretrainedTransformerIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, PretrainedTransformerTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# WIP: Note this dataset reader has partially implemented parameters, inherited from earlier readers
# Much of the code is modified from pytorch-transformer examples for SQuAD
# The treatment of tokens and their string positions is a bit of a mess

@DatasetReader.register("transformer_span_reasoning_single")
class TransformerSpanReasoningSingleReader(DatasetReader):
    """

    """

    def __init__(self,
                 pretrained_model: str,
                 max_pieces: int = 512,
                 syntax: str = "squad",
                 skip_id_regex: str = None,
                 add_prefix: Dict[str, str] = None,
                 ignore_main_context: bool = False,
                 ignore_situation_context: bool = False,
                 dataset_dir_out: str = None,
                 model_type: str = None,
                 doc_stride: int = 100,
                 is_training: bool = True,
                 context_selection: str = "first",
                 answer_can_be_in_question: bool = None,
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
        self._edge_indexers = {'edges': SingleIdTokenIndexer(namespace="edges")}

        self._max_pieces = max_pieces
        self._sample = sample
        self._syntax = syntax
        self._skip_id_regex = skip_id_regex
        self._ignore_main_context = ignore_main_context
        self._ignore_situation_context = ignore_situation_context
        self._dataset_dir_out = dataset_dir_out
        self._model_type = model_type
        self._add_prefix = add_prefix or {}
        self._doc_stride = doc_stride
        self._answer_can_be_in_question = answer_can_be_in_question
        if self._answer_can_be_in_question is None:
            self._answer_can_be_in_question = syntax == "ropes"
        self._allow_no_answer = None
        self._is_training = is_training
        self._context_selection = context_selection
        self._global_debug_counters = {"best_window": 5}
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
        logger.info("Reading instances from jsonl dataset at: %s", file_path)
        examples = self._read_squad_examples(file_path)
        debug = 5

        for example in examples:
            debug -= 1
            if debug > 0:
                logger.info(example)
            yield self._example_to_instance(
                example=example,
                debug=debug)

    @overrides
    def text_to_instance(self,  # type: ignore
                         question: str,
                         background: str,
                         situation: str = None,
                         item_id: str = "NA",
                         debug: int = -1) -> Instance:
        # For use by predictor, does not support answer input atm
        paragraph_text = self._add_prefix.get("c", "") + background
        if self._ignore_main_context:
            paragraph_text = ""
        if situation and not self._ignore_situation_context:
            situation_context = self._add_prefix.get("s", "") + situation
            paragraph_text = paragraph_text + " " + situation_context
        question_text = self._add_prefix.get("q", "") + question
        # We're splitting into subtokens later anyway
        doc_tokens = [paragraph_text]
        question_tokens = [question_text]

        example = SpanPredictionExample(
            qas_id=item_id,
            doc_text=paragraph_text,
            question_text=question_text,
            doc_tokens=doc_tokens,
            question_tokens=question_tokens)
        return self._example_to_instance(example, debug)


    def _example_to_instance(self, example, debug):
        fields: Dict[str, Field] = {}
        features = self._transformer_features_from_example(example, debug)

        tokens_field = TextField(features.tokens, self._token_indexers)
        segment_ids_field = SequenceLabelField(features.segment_ids, tokens_field)
        fields['tokens'] = tokens_field
        fields['segment_ids'] = segment_ids_field
        
        chunks_field = ListField([SpanField(chunk[0], chunk[1], tokens_field) for chunk in features.chunks])
        sentence_graph_nodes_field = ListField([ListField([IndexField(n, chunks_field) for n in nodes]) for nodes in features.sentence_graph_nodes])
        # sentence_graph_edges_field = ListField([TextField([Token(edge) for edge in edges], self._edge_indexers) for edges in features.sentence_graph_edges])
        fs = []
        for edges in features.sentence_graph_edges:
            if len(edges) == 0:
                fs.append(TextField([], self._edge_indexers))
            else:
                fs.append(TextField([Token(edge) for edge in edges], self._edge_indexers))
        sentence_graph_edges_field = ListField(fs)

        fields['chunks'] = chunks_field
        fields['sentence_graph_nodes'] = sentence_graph_nodes_field
        fields['sentence_graph_edges'] = sentence_graph_edges_field

        fields['cands'] = SequenceLabelField(features.cands, chunks_field)
        fields['best'] = IndexField(features.best[0], chunks_field) 

        metadata = {}
        metadata['qas_id'] = example.qas_id
        metadata['cands'] = features.cands
        metadata['best'] = features.best[0]
        if debug > 0:
            logger.info(f"tokens = {features.tokens}")
            logger.info(f"segment_ids = {features.segment_ids}")
            logger.info(f"chunks = {features.chunks}")
            logger.info(f"sentence_graph_nodes = {features.sentence_graph_nodes}")
            logger.info(f"sentence_graph_edges = {features.sentence_graph_edges}")
            
        fields["metadata"] = MetadataField(metadata)
        return Instance(fields)

    def _read_squad_examples(self, input_file):
        """Read a SQuAD-format json file into a list of SpanPredictionExample."""
        with open(input_file, "r", encoding='utf-8') as reader:
            input_data = json.load(reader)["data"]

        version_2_with_negative = self._allow_no_answer

        def is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
                return True
            return False

        examples = []
        for entry in input_data:
            for paragraph in entry["paragraphs"]:
                if self._syntax == "squad":
                    paragraph_text = paragraph["context"]
                elif self._syntax == "ropes":
                    paragraph_text = " ".join(paragraph["background_segs"])
                else:
                    raise ValueError(f"Invalid dataset syntax {self._syntax}!")

                paragraph_text = paragraph_text
                if self._ignore_main_context:
                    paragraph_text = ""
                if self._syntax == "ropes" and not self._ignore_situation_context:
                        situation_text = " ".join(paragraph["situation_segs"])
                        situation_text = self._add_prefix.get("s", "") + situation_text
                        paragraph_text = paragraph_text + " " + situation_text

                doc_tokens = []
                prev_is_whitespace = True
                for c in paragraph_text:
                    if is_whitespace(c):
                        prev_is_whitespace = True
                    else:
                        if prev_is_whitespace:
                            doc_tokens.append(c)
                        else:
                            doc_tokens[-1] += c
                        prev_is_whitespace = False


                for qa in paragraph["qas"]:
                    if qa["skip"]:
                        continue
                    qas_id = qa["id"]
                    if self._syntax == "ropes":
                        question_text = qa["question_segs"]
                    else:
                        question_text = qa["question"]
                    question_text = self._add_prefix.get("q", "") + question_text
                    question_tokens = []
                    prev_is_whitespace = True
                    for c in question_text:
                        if is_whitespace(c):
                            prev_is_whitespace = True
                        else:
                            if prev_is_whitespace:
                                question_tokens.append(c)
                            else:
                                question_tokens[-1] += c
                            prev_is_whitespace = False

                    
                    chunks = [[int(x.strip().split()[0]), int(x.strip().split()[1])] for x in qa["offset_spans"].split("|||")]
                    sents, sentq = qa["bsq_sentence_offsets"].split()[1:]
                    sents, sentq = int(sents), int(sentq)
                    sentence_chunk_offsets = qa["sentence_chunk_offsets"].split()
                    chunkss = int(sentence_chunk_offsets[sents])
                    chunksq = int(sentence_chunk_offsets[sentq])

                    # for item in chunks: #because [CLS] is added in the head of the sentence
                    #     item[0] += 1
                    #     item[1] += 1

                    for item in chunks[chunkss:]: #because S: is inserted at the begin of the situation
                        item[0] += 1
                        item[1] += 1

                    for item in chunks[chunksq:]: #because Q: is inserted at the begin of the question
                        item[0] += 1
                        item[1] += 1

                    cands = [ [int(a) for a in qa["positive_nodes"].split()], [int(a) for a in qa["candidate_nodes"].split()]]

                    graph = [ [int(x.strip().split()[0]), int(x.strip().split()[1]), x.strip().split()[2]] for x in qa["deps"].split("|||")]


                    example = SpanPredictionExample(
                        qas_id=qas_id,
                        doc_text=paragraph_text,
                        doc_tokens=doc_tokens,
                        doc_chunks=chunks[:chunksq],
                        q_text = question_text,
                        q_tokens=question_tokens,
                        q_chunks=chunks[chunksq:],
                        cands=cands[1],
                        sentence_graph=graph,
                        best=cands[0],
                        is_impossible=qa["skip"])
                    examples.append(example)
                    if self._sample > 0 and len(examples) > self._sample:
                        return examples
        return examples

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

    def _improve_answer_span(self, doc_tokens, input_start, input_end,
                             orig_answer_text):
        """Returns tokenized answer spans that better match the annotated answer."""

        # The SQuAD annotations are character based. We first project them to
        # whitespace-tokenized words. But then after WordPiece/etc tokenization, we can
        # often find a "better match". For example:
        #
        #   Question: What year was John Smith born?
        #   Context: The leader was John Smith (1895-1943).
        #   Answer: 1895
        #
        # The original whitespace-tokenized answer will be "(1895-1943).". However
        # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
        # the exact answer, 1895.
        #
        # However, this is not always possible. Consider the following:
        #
        #   Question: What country is the top exporter of electornics?
        #   Context: The Japanese electronics industry is the lagest in the world.
        #   Answer: Japan
        #
        # In this case, the annotator chose "Japan" as a character sub-span of
        # the word "Japanese". Since our WordPiece tokenizer does not split
        # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
        # in SQuAD, but does happen.
        tok_answer_text = self._string_from_tokens(self._tokenizer.tokenize(orig_answer_text))

        for new_start in range(input_start, input_end + 1):
            for new_end in range(input_end, new_start - 1, -1):
                text_span = self._string_from_tokens(doc_tokens[new_start:(new_end + 1)])
                if text_span == tok_answer_text:
                    return (new_start, new_end)

        return (input_start, input_end)

    def _string_from_tokens(self, tokens):
        tokens_text = [x.text for x in tokens]
        if hasattr(self._tokenizer_internal, "convert_tokens_to_string"):
            return self._tokenizer_internal.convert_tokens_to_string(tokens_text)
        else:
            return " ".join(tokens_text)

    def _transformer_features_from_example(self, example, debug):

        cls_token = Token(self._tokenizer_internal.cls_token)
        sep_token = Token(self._tokenizer_internal.sep_token)
        cls_token_at_end = bool(self._model_type in ['xlnet'])
        cls_token_segment_id = 2 if self._model_type in ['xlnet'] else 0
        sequence_a_segment_id = 0
        sequence_b_segment_id = 1

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = self._tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        q_tok_to_orig_index = []
        q_orig_to_tok_index = []
        all_query_tokens = []
        for (i, token) in enumerate(example.q_tokens):
            q_orig_to_tok_index.append(len(all_query_tokens))
            sub_tokens = self._tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                q_tok_to_orig_index.append(i)
                all_query_tokens.append(sub_token)

        # print(len(all_doc_tokens))
        # print(all_doc_tokens)
        # print(all_doc_tokens[450:])
        # print(len(all_query_tokens))
        # print(all_query_tokens)



        # print(orig_to_tok_index)
        # print(len(orig_to_tok_index))
        # print(f"before example.doc_chunks : {example.doc_chunks}")
        for chunk in example.doc_chunks:
            s, e = chunk
            chunk[0] = orig_to_tok_index[s]
            chunk[1] = orig_to_tok_index[e + 1] - 1 if e + 1 < len(orig_to_tok_index) else len(all_doc_tokens) - 1

        # print(f"example.doc_chunks : {example.doc_chunks}")
        # print(len(all_doc_tokens))

        # print(f"before example.q_chunks : {example.q_chunks}")
        for chunk in example.q_chunks:
            s, e = chunk
            chunk[0] = q_orig_to_tok_index[s - len(orig_to_tok_index)] + len(all_doc_tokens)
            chunk[1] = q_orig_to_tok_index[e + 1 - len(orig_to_tok_index)] - 1 + len(all_doc_tokens) if  e + 1 - len(orig_to_tok_index) < len(q_orig_to_tok_index) else len(all_doc_tokens) + len(all_query_tokens) - 1

        # print(f"example.q_chunks : {example.q_chunks}")
        if len(all_query_tokens) > self._max_pieces:
            assert False, "not allowed"
            # all_query_tokens = all_query_tokens[0:self._max_pieces]

        tokens = all_doc_tokens + all_query_tokens
        # The -3 accounts for [CLS], [SEP] and [SEP]

        # print(f"max_pieces : {self._max_pieces}")
        max_tokens_for_doc = self._max_pieces - len(all_query_tokens) - 3

        # here we need to recover the spans true index of candidates
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, self._doc_stride)



        start_offset = 0
        if len(all_doc_tokens) > max_tokens_for_doc:
            start_offset = len(all_doc_tokens) - max_tokens_for_doc
            all_doc_tokens = all_doc_tokens[-max_tokens_for_doc:]
            
        ndelchunk = 0
        for chunk in example.doc_chunks:
            if chunk[0] >= start_offset:
                break
            ndelchunk += 1

        example.doc_chunks = example.doc_chunks[ndelchunk:]
        for chunk in example.doc_chunks:
            chunk[0] -= start_offset
            chunk[1] -= start_offset
        for chunk in example.q_chunks:
            chunk[0] -= start_offset
            chunk[1] -= start_offset

        if ndelchunk != 0:
            edges = []
            for edge in example.sentence_graph:
                if edge[0] < ndelchunk or edge[1] < ndelchunk:
                    pass
                else:
                    edge[0] -= ndelchunk
                    edge[1] -= ndelchunk
                    edges.append(edge)
            example.sentence_graph = edges

            for i in range(len(example.cands)): # candidates
                assert example.cands[i] < ndelchunk
                example.cands[i] -= ndelchunk
            for i in range(len(example.best)): #  positives
                assert example.best[i] < ndelchunk
                example.best[i] -= ndelchunk

        # print(f"window example.doc_chunks : {example.doc_chunks}")
        # print(f"window example.q_chunks : {example.q_chunks}")

        # We can have documents that are longer than the maximum sequence length.
        # # To deal with this we do a sliding window approach, where we take chunks
        # # of the up to our max length with a stride of `doc_stride`.
        # _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
        #     "DocSpan", ["start", "length"])
        # doc_spans = []
        # start_offset = 0
        # while start_offset < len(all_doc_tokens):
        #     length = len(all_doc_tokens) - start_offset
        #     if length > max_tokens_for_doc:
        #         length = max_tokens_for_doc
        #     doc_spans.append(_DocSpan(start=start_offset, length=length))
        #     if start_offset + length == len(all_doc_tokens):
        #         break
        #     start_offset += min(length, self._doc_stride)

        # features_list = []
        tokens = []
        segment_ids = []
        if not cls_token_at_end:
            tokens.append(cls_token)
            segment_ids.append(cls_token_segment_id)
            cls_index = 0
            for chunk in example.doc_chunks:
                chunk[0] += 1
                chunk[1] += 1
            for chunk in example.q_chunks:
                chunk[0] += 2
                chunk[1] += 2

        tokens += all_doc_tokens
        segment_ids += [sequence_a_segment_id] * len(all_doc_tokens)

        tokens.append(sep_token)
        segment_ids.append(sequence_a_segment_id)

        tokens += all_query_tokens
        segment_ids += [sequence_b_segment_id] * len(all_query_tokens)

        if cls_token_at_end:
            tokens.append(cls_token)
            segment_ids.append(cls_token_segment_id)
            cls_index = len(tokens) - 1  # Inde
            for chunk in example.q_chunks:
                chunk[0] += 1
                chunk[1] += 1

        # corefs
        # corefs = []
        # for coref in example.corefs:
        #     corefs.append(tuple(coref))
        # corefs_label = ["coref"] * len(example.corefs)

        # chunks
        chunks = []
        for chunk in example.doc_chunks:
            chunks.append(tuple(chunk))
        for chunk in example.q_chunks:
            chunks.append(tuple(chunk))

            

        #setence graph
        sentence_graph_nodes = [ [] for _ in range(len(chunks))]
        sentence_graph_edges = [ [] for _ in range(len(chunks))]
        for item in example.sentence_graph:
            sentence_graph_nodes[item[0]].append(item[1])
            sentence_graph_edges[item[0]].append(item[2])

        for item in sentence_graph_nodes:
            if len(item) == 0:
                item.append(-1)
        # for item in sentence_graph_edges:
        #     if len(item) == 0:
        #         item.append("None")

        
        cands = [ 0 for _  in range(len(chunks))]
        for cand in example.cands:
            cands[cand] = 1
        # for chunk in example.doc_chunks:
        #     print(tokens[chunk[0]:chunk[1]+1])
        # for chunk in example.q_chunks:
        #     print(tokens[chunk[0]:chunk[1]+1])
        # for i in example.cands:
        #     print(tokens[chunks[i][0]:chunks[i][1]+1])
        # for i in example.best:
        #     print(tokens[chunks[i][0]:chunks[i][1]+1])

        if debug > 0:
            logger.info("*** Features ***")
            logger.info(f"unique_id: {example.qas_id}")
            logger.info(f"tokens: {tokens}")
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info(f"chunks: {chunks}" )
            logger.info(f"sentence_graph_nodes: {sentence_graph_nodes}" )
            logger.info(f"sentence_graph_edges: {sentence_graph_edges}" )
            logger.info(f"cands: {cands}" )
            logger.info(f"cands_best: {example.best}")

        return InputFeatures(
                    unique_id=example.qas_id,
                    tokens=tokens,
                    segment_ids=segment_ids,
                    cls_index=cls_index,
                    chunks=chunks,
                    sentence_graph_nodes=sentence_graph_nodes,
                    sentence_graph_edges=sentence_graph_edges,
                    cands=cands,
                    best=example.best
                    )
        # Just filter away impossible/missing spans for now (this uses labels, so not fair on dev/test):
        # if example.orig_answer_text and self._is_training:
        #     features_list = list(filter(lambda x: not x.is_impossible and x.start_position is not None, features_list))
        # if not self._is_training and len(features_list) > 1:
        #     # If we're not creating training data, just pick the first/last context window
        #     if self._context_selection == "last":
        #         features_list = features_list[-1:]
        #     else:
        #         features_list = features_list[:1]
        # if len(features_list) > 1:
        #     top_score = -1
        #     # For now we'll just keep the first/last context which has the most answer tokens
        #     selected = None
        #     for f in features_list:
        #         if f.start_position is not None and f.end_position is not None:
        #             score = sum(10 + 0 * int(f.token_is_max_context.get(i, False)) for i in range(f.start_position, f.end_position) )
        #             if score > top_score or (self._context_selection == "last" and score >= top_score):
        #                 top_score = score
        #                 selected = f
        #     if self._global_debug_counters["best_window"] > 0:
        #         self._global_debug_counters["best_window"] -= 1
        #         logger.info(f"For answer '{example.orig_answer_text}', picked \n{self._string_from_tokens(selected.tokens)} "+
        #         f"\nagainst \n{[self._string_from_tokens(x.tokens) for x in features_list]} ")
        # elif len(features_list) == 1:
        #     selected = features_list[0]
        # else:
        #     selected = None
        # return selected


class SpanPredictionExample(object):
    """
    A single training/test example for a span prediction dataset.
    For examples without an answer, the start and end position are -1.
    """
    def __init__(self,
                qas_id,
                doc_text,
                doc_tokens,
                doc_chunks,
                q_text,
                q_tokens,
                q_chunks,
                cands,
                sentence_graph,
                best,
                is_impossible=None):
        self.qas_id = qas_id
        self.doc_text = doc_text
        self.doc_tokens = doc_tokens
        self.doc_chunks = doc_chunks
        self.q_text = q_text
        self.q_tokens = q_tokens
        self.q_chunks = q_chunks
        self.cands = cands
        self.sentence_graph = sentence_graph
        self.best = best
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "  qas_id: %s" % (self.qas_id)
        s += "\n  doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        s += f"\n  doc_chunks: {self.doc_chunks}"
        s += "\n  q_tokens: [%s]" % (" ".join(self.q_tokens))
        s += f"\n  q_chunks: {self.q_chunks}"
        s += f"\n. sentence_graph: {self.sentence_graph}"
        s += f"\n. cands: {self.cands}"
        if self.is_impossible:
            s += "\n  is_impossible: %r" % (self.is_impossible)
        return s

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                unique_id,
                tokens,
                segment_ids,
                cls_index,
                chunks,
                sentence_graph_nodes,
                sentence_graph_edges,
                cands,
                best):
    # def __init__(self,
    #              unique_id,
    #              example_index,
    #              doc_span_index,
    #              tokens,
    #              token_to_orig_map,
    #              token_is_max_context,
    #              input_ids,
    #              input_mask,
    #              segment_ids,
    #              cls_index,
    #              p_mask,
    #              paragraph_len,
    #              start_position=None,
    #              end_position=None,
    #              is_impossible=None):
        self.unique_id = unique_id
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.cls_index = cls_index
        self.chunks = chunks
        self.sentence_graph_nodes=sentence_graph_nodes
        self.sentence_graph_edges = sentence_graph_edges
        self.cands = cands
        self.best = best


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens

def _find_last_substring_index(pattern_string, string):
    if not pattern_string:
        return None
    regex = re.escape(pattern_string)
    if pattern_string[0].isalpha() or pattern_string[0].isdigit():
        regex = "\\b" + regex
    if pattern_string[-1].isalpha() or pattern_string[-1].isdigit():
        regex = regex + "\\b"
    res = [match.start() for match in re.finditer(regex, string)]
    if len(res) == 0:
        regex_uncased = "(?i)" + regex
        res = [match.start() for match in re.finditer(regex_uncased, string.lower())]
    return res[-1] if res else None


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index
