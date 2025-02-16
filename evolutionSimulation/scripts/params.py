#conda_env: evolution

from evolutionSimulation.python.neuralnetworks.nn import Brain
from evolutionSimulation.python.neuralnetworks.params import *
import random
import torch
import torch.nn
import time
from evolutionSimulation.scripts.timer import timed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use CUDA if available

def randomize(tensor, random: float = 0):
    if random == 0:  # Adjusted the comparison to random instead of float
        return tensor
    minVal = 1 - random * 0.5
    maxVal = 1 + random * 0.5
    
    randTensor = minVal + (maxVal - minVal) * torch.rand(tensor.shape, device=DEVICE)  # Ensure it's on the right device
    return randTensor * tensor

def modify(first, second, shouldSwap: bool = False, shouldMerge: bool = False, random: float = 0, layers: list = [], shouldPrint = False):
    """
    Modify parameters of model either by swapping them or merging them.
    Either swap the params or take the average of the two.
    
    Args:
        first (Brain): Custom CNN
        second (Brain): Custom CNN 
        swap (bool): Whether or not we should swap the layers 
        merge (bool): Whether or not we should merge the layers 
        random (float): How much randomness should be added to the model 
        layers (list): List of layers to swap layers 

    Returns: 
        first (Brain): Modified CNN
    """

    # Swapping all layers
    if shouldSwap and layers == []:
        if shouldPrint:
            print(f"Swapping all layers between {first.name} with {second.name}")
        # Iterate through layers 
        for name, module in first.named_modules():
            if hasattr(module, "weight"):
                if shouldPrint:
                    print(f"Swapping {name}\n")
                swap = getattr(second, name)
                swapTensor = swap.weight.clone().to(DEVICE)
                module.weight.data = swapTensor
                if random != 0:
                    module.weight.data = randomize(module.weight.data, random)

    # Swapping specified layers
    elif shouldSwap and layers != []:
        if shouldPrint:
            print(f"Swapping {layers} between {first.name} with {second.name}")
        for name, module in first.named_modules():
            if name in layers:
                if shouldPrint:
                    print(f"Swapping {name}\n")
                swap = getattr(second, name)
                swapTensor = swap.weight.clone().to(DEVICE)
                module.weight.data = swapTensor
                if random != 0:
                    module.weight.data = randomize(module.weight.data, random)

    # Merging all layers
    elif shouldMerge and layers == []:
        if shouldPrint:
            print(f"Merging all layers between {first.name} with {second.name}")
        for name, module in first.named_modules():
            if hasattr(module, "weight"):
                if shouldPrint:
                    print(f"Merging {name}\n")
                # Get weights of both layers, add them, and divide by 2
                firstW = getattr(first, name).weight.clone().to(DEVICE)
                secondW = getattr(second, name).weight.clone().to(DEVICE)
                
                module.weight.data = torch.div(firstW + secondW, 2)
                if random != 0:
                    module.weight.data = randomize(module.weight.data, random)

    # Merging specified layers
    else:
        if shouldPrint:
            print(f"Merging {layers} between {first.name} with {second.name}")
        for name, module in first.named_modules():
            if name in layers:
                if shouldPrint:
                    print(f"Merging {name}\n")
                # Get weights of both layers, add them, and divide by 2
                firstW = getattr(first, name).weight.clone().to(DEVICE)
                secondW = getattr(second, name).weight.clone().to(DEVICE)
                module.weight.data = torch.div(firstW + secondW, 2)
                if random != 0:
                    module.weight.data = randomize(module.weight.data, random)
    return first



"""
allan = Brain("allan")
brandon = Brain("brandon")
base = Brain("base")

base.load_state_dict(allan.state_dict())
base.to(DEVICE)  # Move to GPU

modify(allan, brandon, shouldSwap=True, shouldMerge=False, random=0.5, shouldPrint=True)

compareParams = timed(compareParams)
compareParams(allan, brandon, False)
"""
