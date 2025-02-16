#conda_env: evolution
from nn import Brain
import torch
import torch.nn as nn 


def brainStudy(model, see_all = False):
    """
    Prints all of the layers (name + size) in the model
    
    Args:
        model (Brain): CNN to observe layers of
        see_all (bool): Whether print the actual tensors

    Returns: 
        None
    """
    print("\n")
    for name, module in model.named_modules():
        if name == "":
            continue
        print(f"Name: {name} || Module: {module}")
        if hasattr(module, 'weight'):
            layer = getattr(model, name)
            print(f"Weight Shape: {layer.weight.shape}")
            if see_all == True:
                test = layer.weight.clone().detach()
                print(test)
        print("---------------------------")

def paramSwap(model, swap = None):
    """
    If params 
    """
    if swap == None:
        for name, module in model.named_modules():
            if name == "":
                continue
            if hasattr(module, 'weight'):
                layer = getattr(model, name)
                test = layer.weight.clone().detach() 
                zeros = torch.zeros(layer.weight.shape)
                module.weight.data = zeros
    return model 

def compareParams(base, comparison):
    difference = 0
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    for name, module in base.named_modules():
        if name == "":
            continue
        print(f"Name: {name} || Module: {module}")
        if hasattr(module, 'weight'):
            default = getattr(base, name)
            compare = getattr(comparison, name)
            if torch.equal(default.weight.clone().detach(), compare.weight.clone().detach()):
                print(f"{name} is the same") 
            else:
                print(f"{name} is different")
                diff = cos(default.weight.clone().detach(), compare.weight.clone().detach())
                difference += diff.mean().item()
                print(f"Cosine similarity for {name}: {diff.mean().item()}")
        print("---------------------------")
    if difference == 0:
        print("Identical Model")
    else:
        print(difference)
    return difference 

