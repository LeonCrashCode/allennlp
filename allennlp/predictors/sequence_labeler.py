# pylint: disable=protected-access
from copy import deepcopy
from typing import Dict, List

from overrides import overrides
import numpy

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from allennlp.data.fields import (IndexField, ListField, LabelField, SpanField, SequenceLabelField,
                                  SequenceField)


@Predictor.register('sequence-labelling-gec')
class SequenceLabellingGECPredictor(Predictor):

    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence": sentence})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"question": "...", "passage": "..."}``.
        """

        sentence = json_dict["tokens"]
        item_id = json_dict["id"]
        return self._dataset_reader.text_to_instance(item_id=item_id, sent=sentence)

    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instance = self._json_to_instance(inputs)
        output = self.predict_instance(instance)
        # outputs = self._model.forward_on_instance(instance)
        return output
