"""Create model."""# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2020, All Rights Reserved.
# ***
# ***    File Author: Dell, 2020年 12月 20日 星期日 08:42:09 CST
# ***
# ************************************************************************************/
#

import os
import torch
from Network import CleanFlow, SlowFlow, ZoomFlow
import pdb

def model_load(model, path):
    """Load model."""

    state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    target_state_dict = model.state_dict()

    for n, p in state_dict.items():
        # n = n.replace("SpyNet.", "spynet.")
        # n = n.replace("ResNet.", "resnet.")
        if n in target_state_dict.keys():
            target_state_dict[n].copy_(p)
        else:
            raise KeyError(n)

def get_model(task):
    """Create model."""

    model_setenv()

    if task == "clean":
        checkpoint = "models/TOFlowClean.pth"
        model = CleanFlow()
    elif task == "slow":
        checkpoint = "models/TOFlowSlow.pth"
        model = SlowFlow()
    elif task == "zoom":
        checkpoint = "models/TOFlowZoom.pth"
        model = ZoomFlow()

    model_load(model, checkpoint)

    return model

def model_save(model, path):
    """Save model."""

    torch.save(model.state_dict(), path)

def model_device():
    """Please call after model_setenv. """
    return torch.device(os.environ["DEVICE"])

def model_setenv():
    """Setup environ  ..."""

    # random init ...
    import random
    random.seed(42)
    torch.manual_seed(42)

    # Set default device to avoid exceptions
    if os.environ.get("DEVICE") != "cuda" and os.environ.get("DEVICE") != "cpu":
        os.environ["DEVICE"] = 'cuda' if torch.cuda.is_available() else 'cpu'

    if os.environ["DEVICE"] == 'cuda':
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    print("Running Environment:")
    print("----------------------------------------------")
    print("  PWD: ", os.environ["PWD"])
    print("  DEVICE: ", os.environ["DEVICE"])


if __name__ == '__main__':
    """Test model ..."""

    model = get_model("models/TOFlowZoom.pth")
    print(model)

