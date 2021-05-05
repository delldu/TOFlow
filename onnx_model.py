"""Onnx Model Tools."""# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 03月 23日 星期二 12:42:57 CST
# ***
# ************************************************************************************/
#

import argparse
import os
import pdb  # For debug
import time
import sys

import numpy as np
import onnx
import onnxruntime
import torch
import torchvision.transforms as transforms
from PIL import Image

#
# /************************************************************************************
# ***
# ***    MS: Import Model Method
# ***
# ************************************************************************************/
#
from model import get_model, model_load

def onnx_load(onnx_file):
    session_options = onnxruntime.SessionOptions()
    # session_options.log_severity_level = 0

    # Set graph optimization level
    session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

    onnx_model = onnxruntime.InferenceSession(onnx_file, session_options)
    # onnx_model.set_providers(['CUDAExecutionProvider'])
    print("Onnx Model Engine: ", onnx_model.get_providers(),
          "Device: ", onnxruntime.get_device())

    return onnx_model


def onnx_forward(onnx_model, input):
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    onnxruntime_inputs = {onnx_model.get_inputs()[0].name: to_numpy(input)}
    onnxruntime_outputs = onnx_model.run(None, onnxruntime_inputs)
    return torch.from_numpy(onnxruntime_outputs[0])


if __name__ == '__main__':
    """Onnx tools ..."""

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--export', help="export onnx model", action='store_true')
    parser.add_argument('-v', '--verify', help="verify onnx model", action='store_true')
    parser.add_argument('-o', '--output', type=str, default="output", help="output folder")

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    #
    # /************************************************************************************
    # ***
    # ***    MS: Define Global Names
    # ***
    # ************************************************************************************/
    #

    checkpoint = "models/VideoRIFE.pth"

    def export_clean_onnx():
        """Export onnx model."""

        dummy_input = torch.randn(1, 7, 3, 256, 448)
        onnx_file_name = "{}/toflow_clean.onnx".format(args.output)

        # 1. Create and load model.
        torch_model = get_model("clean")
        torch_model.eval()

        # 2. Model export
        print("Exporting onnx model to {}...".format(onnx_file_name))

        input_names = ["input"]
        output_names = ["output"]
        dynamic_axes = {'input': {3: "height", 4: "width"},
                        'output': {4: "height", 4: "width"}}

        torch.onnx.export(torch_model, dummy_input, onnx_file_name,
                          input_names=input_names,
                          output_names=output_names,
                          verbose=True,
                          opset_version=11,
                          keep_initializers_as_inputs=False,
                          dynamic_axes=dynamic_axes,
                          export_params=True)

        # 3. Optimize model
        # print('Checking model ...')
        # onnx_model = onnx.load(onnx_file_name)
        # onnx.checker.check_model(onnx_model)
        # https://github.com/onnx/optimizer

        # 4. Visual model
        # python -c "import netron; netron.start('output/toflow_clean.onnx')"

    def export_slow_onnx():
        """Export onnx model."""

        dummy_input = torch.randn(1, 2, 3, 256, 448)
        onnx_file_name = "{}/toflow_slow.onnx".format(args.output)

        # 1. Create and load model.
        torch_model = get_model("slow")
        torch_model.eval()

        # 2. Model export
        print("Exporting onnx model to {}...".format(onnx_file_name))

        input_names = ["input"]
        output_names = ["output"]
        dynamic_axes = {'input': {3: "height", 4: "width"},
                        'output': {4: "height", 4: "width"}}

        torch.onnx.export(torch_model, dummy_input, onnx_file_name,
                          input_names=input_names,
                          output_names=output_names,
                          verbose=True,
                          opset_version=11,
                          keep_initializers_as_inputs=False,
                          dynamic_axes=dynamic_axes,
                          export_params=True)

        # 3. Optimize model
        # print('Checking model ...')
        # onnx_model = onnx.load(onnx_file_name)
        # onnx.checker.check_model(onnx_model)
        # https://github.com/onnx/optimizer

        # 4. Visual model
        # python -c "import netron; netron.start('output/toflow_slow.onnx')"

    def export_zoom_onnx():
        """Export onnx model."""

        dummy_input = torch.randn(1, 7, 3, 256, 448)
        onnx_file_name = "{}/toflow_zoom.onnx".format(args.output)

        # 1. Create and load model.
        torch_model = get_model("zoom")
        torch_model.eval()

        # 2. Model export
        print("Exporting onnx model to {}...".format(onnx_file_name))

        input_names = ["input"]
        output_names = ["output"]
        dynamic_axes = {'input': {3: "height", 4: "width"},
                        'output': {4: "height", 4: "width"}}

        torch.onnx.export(torch_model, dummy_input, onnx_file_name,
                          input_names=input_names,
                          output_names=output_names,
                          verbose=True,
                          opset_version=11,
                          keep_initializers_as_inputs=False,
                          dynamic_axes=dynamic_axes,
                          export_params=True)

        # 3. Optimize model
        # print('Checking model ...')
        # onnx_model = onnx.load(onnx_file_name)
        # onnx.checker.check_model(onnx_model)
        # https://github.com/onnx/optimizer

        # 4. Visual model
        # python -c "import netron; netron.start('output/toflow_zoom.onnx')"


    def verify_onnx():
        """Verify onnx model."""

        sys.exit("Sorry, this function NOT work for grid_sampler, please use onnxservice to test.")
        dummy_input = torch.randn(2, 3, 256, 256)
        onnx_file_name = "{}/toflow_clean.onnx".format(args.output)

        torch_model = get_model(checkpoint)
        torch_model.eval()

        onnxruntime_engine = onnx_load(onnx_file_name)

        def to_numpy(tensor):
            return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

        with torch.no_grad():
            torch_output = torch_model(dummy_input)

        onnxruntime_inputs = {onnxruntime_engine.get_inputs()[0].name: to_numpy(dummy_input)}
        onnxruntime_outputs = onnxruntime_engine.run(None, onnxruntime_inputs)

        np.testing.assert_allclose(to_numpy(torch_output), onnxruntime_outputs[0], rtol=1e-03, atol=1e-03)
        print("Onnx model {} has been tested with ONNXRuntime, result sounds good !".format(onnx_file_name))


    if args.export:
        export_clean_onnx()
        export_slow_onnx()
        export_zoom_onnx()

    if args.verify:
        verify_onnx()
