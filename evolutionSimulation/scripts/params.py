#conda_env: evolution
from evolutionSimulation.python.neuralnetworks.nn import Brain
from evolutionSimulation.python.neuralnetworks.params import *
import random
import torch
import torch.nn
import time
from evolutionSimulation.scripts.timer import timed

@timed
def modify(first, second, shouldSwap:bool = False, shouldMerge: bool = False, random: float = 0, layers: list = []):
    """
    Modify parameters of model either by swapping them or merging them
    Either swap the params or take the average of the two
    
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
    if shouldSwap == True and layers == []:
        print(f"Swapping all layers between {first.name} with {second.name}")
        # Iterate through layers 
        for name, module in first.named_modules():
            if hasattr(module, "weight"):
                print(f"Swapping {name}\n")
                swap = getattr(second, name)
                # Clone second model's tensors and swap with first
                swapTensor = swap.weight.clone().detach()
                module.weight.data = swapTensor
    
    # Swapping specified layers
    elif shouldSwap == True and layers != []:
        print(f"Swapping {layers} between {first.name} with {second.name}")
        for name, module in first.named_modules():
            if name in layers:
                print(f"Swapping {name}\n")
                swap = getattr(second, name)
                swapTensor = swap.weight.clone().detach()
                module.weight.data = swapTensor
    
    # Merging all layers
    elif shouldMerge == True and layers == []:
        print(f"Merging all layers between {first.name} with {second.name}")
        for name, module in first.named_modules():
            if hasattr(module, "weight"):
                print(f"Merging {name}\n")
                # Get weights of both layers, add them, and divide by 2
                firstW = getattr(first, name)
                secondW = getattr(second, name)
                module.weight.data = torch.div(firstW.weight.clone().detach() + secondW.weight.clone().detach(), 2)
    
    # Merging specified layers
    else: 
        print(f"Merging {layers} between {first.name} with {second.name}")
        for name, module in first.named_modules():
            if name in layers:
                print(f"Merging {name}\n")
                # Get weights of both layers, add them, and divide by 2
                firstW = getattr(first, name)
                secondW = getattr(second, name)
                module.weight.data = torch.div(firstW.weight.clone().detach() + secondW.weight.clone().detach(), 2)
    
allan = Brain("allan")
brandon = Brain("brandon")
base = Brain("base")
base.load_state_dict(allan.state_dict())


compareParams = timed(compareParams)

layers = ["conv1", "conv2"]
modify(allan, brandon, shouldSwap=False, shouldMerge=True)

compareParams(allan, base)
compareParams(brandon, base)
compareParams(brandon, allan)
