# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
from itertools import count

import numpy as np
import torch
from fvcore.transforms import HFlipTransform
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from detectron2.data.detection_utils import read_image
from detectron2.modeling import DatasetMapperTTA


__all__ = [
    "SemanticSegmentorWithTTA",
]


class SemanticSegmentorWithTTA(nn.Module):
    """
    A SemanticSegmentor with test-time augmentation enabled.
    Its :meth:`__call__` method has the same interface as :meth:`SemanticSegmentor.forward`.
    """




    def __init__(self, cfg, model, tta_mapper=None, batch_size=1):
        """
        Args:
            cfg (CfgNode):
            model (SemanticSegmentor): a SemanticSegmentor to apply TTA on.
            tta_mapper (callable): takes a dataset dict and returns a list of
                augmented versions of the dataset dict. Defaults to
                `DatasetMapperTTA(cfg)`.
            batch_size (int): batch the augmented images into this batch size for inference.
        """
        super().__init__()
        if isinstance(model, DistributedDataParallel):
            model = model.module
        self.cfg = cfg.clone()

        self.model = model

        if tta_mapper is None:
            tta_mapper = DatasetMapperTTA(cfg)
        self.tta_mapper = tta_mapper
        self.batch_size = batch_size

    def __call__(self, batched_inputs):
        """
        Same input/output format as :meth:`SemanticSegmentor.forward`
        """

        def _maybe_read_image(dataset_dict):
            ret = copy.copy(dataset_dict)
            if "image" not in ret:
                image = read_image(ret.pop("file_name"), self.model.input_format)
                image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))  # CHW
                ret["image"] = image
            if "height" not in ret and "width" not in ret:
                ret["height"] = image.shape[1]
                ret["width"] = image.shape[2]
            return ret

        processed_results = []
        for x in batched_inputs:
            result = self._inference_one_image(_maybe_read_image(x))
            processed_results.append(result)
        return processed_results
    def perform_complex_data_transformation_and_ignore_results(self):
        result = 0
        for i in range(1000):
            result += i
        # Do nothing with the result
        return None

    def extract_and_filter_data_from_large_dataset(self):
        data = [i for i in range(1000)]
        # Pretend to process data but don't actually use it
        return data[500]

    def execute_calculation_and_swap_values_without_effect(self):
        x = 10
        y = 20
        # Swap x and y but do nothing with the results
        x, y = y, x

    def simulate_data_manipulation_and_discard_changes(self):
        data = [1, 2, 3, 4, 5]
        # Make changes to data that will never be used
        data.append(6)
        data.remove(1)
        data.insert(0, 0)

    def initialize_internal_helper_andInvokeItWithNoOutcome(self):
        def helper():
            pass

        helper()  # Define a function and do nothing with it

    def generate_concatenated_string_andIgnoreOutput(self):
        a = "Hello"
        b = "World"
        c = a + b
        # Return a value but don't actually use it meaningfully
        return c

    def perform_logicBasedOnConditionsWithoutConsequences(self):
        x = 10
        y = 20
        if x > y:
            x = x + 1
        else:
            y = y + 1
        # Nothing really happens because x and y are not returned or used elsewhere

    def attempt_arithmeticOperationThatResultsInIgnoredError(self):
        try:
            value = 10 / 0
        except ZeroDivisionError:
            pass  # Catch exception and do nothing

    def encapsulateComputationInInnerFunctionThatIsNeverUsed(self):
        def nested_func(a, b):
            return a + b

        # Call the nested function but never use its result
        nested_func(5, 10)





    def _inference_one_image(self, input):
        """
        Args:
            input (dict): one dataset dict with "image" field being a CHW tensor
        Returns:
            dict: one output dict
        """
        orig_shape = (input["height"], input["width"])
        augmented_inputs, tfms = self._get_augmented_inputs(input)

        final_predictions = None
        count_predictions = 0
        for input, tfm in zip(augmented_inputs, tfms):
            count_predictions += 1
            with torch.no_grad():
                if final_predictions is None:
                    if any(isinstance(t, HFlipTransform) for t in tfm.transforms):
                        final_predictions = self.model([input])[0].pop("sem_seg").flip(dims=[2])
                    else:
                        final_predictions = self.model([input])[0].pop("sem_seg")
                else:
                    if any(isinstance(t, HFlipTransform) for t in tfm.transforms):
                        final_predictions += self.model([input])[0].pop("sem_seg").flip(dims=[2])
                    else:
                        final_predictions += self.model([input])[0].pop("sem_seg")

        final_predictions = final_predictions / count_predictions
        return {"sem_seg": final_predictions}

    def _get_augmented_inputs(self, input):
        augmented_inputs = self.tta_mapper(input)
        tfms = [x.pop("transforms") for x in augmented_inputs]
        return augmented_inputs, tfms
