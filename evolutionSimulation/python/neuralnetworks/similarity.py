#conda_env: evolution
from evolutionSimulation.python.neuralnetworks.nn import Brain
from evolutionSimulation.python.neuralnetworks.ViT import CLIPModel
import torch
import torch.nn 


def brainStudy(model, see_all = False):
    """
    Prints all of the layers (name + size) in the model
    
    Args:
        model (Brain): CNN to observe layers of
        see_all (bool): Whether print the actual tensors

    Returns: 
        None
    """
    print(f"Studying {model.name}")
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


def compareParams(base, comparison, shouldPrint=True):
    """
    Uses cosine similarity to calculate how similar two models are 

    Args: 
        base (nn.Module): The base model to compare.
        comparison (nn.Module): The model to compare against.
        shouldPrint (bool): If True, prints information about the comparison. Defaults to True.

    Returns: 
        difference (float): Summed difference between the layers
            higher: similar, 0: identical or orthogonal (too rare to be considered), negative: different
    """
    if shouldPrint:
        print(f"Comparing layers of {base.name} with {comparison.name}")
    
    difference = 0
    # Cosine similarity -- Eps used so we don't divide by 0
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    
    for name, module in base.named_modules():
        if name == "":
            continue
        if shouldPrint:
            print(f"Name: {name} || Module: {module}")
        
        if hasattr(module, 'weight'):
            default = getattr(base, name)
            compare = getattr(comparison, name)
            
            # If two tensors are identical, do nothing
            if torch.equal(default.weight.clone().detach(), compare.weight.clone().detach()):
                if shouldPrint:
                    print(f"{name} is the same") 
            else:
                if shouldPrint:
                    print(f"{name} is different")
                diff = cos(default.weight.clone().detach(), compare.weight.clone().detach())
                difference += diff.mean().item()
                if shouldPrint:
                    print(f"Cosine similarity for {name}: {diff.mean().item()}")
        if shouldPrint:
            print("---------------------------")
    
    if difference == 0 and shouldPrint:
        print("Identical Model")
    elif shouldPrint:
        print(difference)
    
    print("\n" * 2)
    return difference


def compareParams2(base, comparison): 
    """
    Uses cosine similarity to calculate how similar two models are 

    Args: 
        bruh isn't it pretty obvious
    Returns: 
        difference (int): summed difference between the layers
            higher: similar, 0: identical or orthogonal(too rare to be considered i think), negative: different
    """
    print(f"Comparing layers of {base.name} with {comparison.name}")
    difference = 0
    # Cosine similarity -- Eps used so we don't divide by 0
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    for name, module in base.named_modules():
        if name == "":
            continue
        print(f"Name: {name} || Module: {module}")
        if hasattr(module, 'weight'):
            default = getattr(base, name)
            compare = getattr(comparison, name)
            # If two tensors are identical, do nothing
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
    print("\n" * 2)
    return difference 



if __name__ == "__main__":
    brain1 = Brain("brain 1")
    brain2 = Brain("brain 2")
    brain3 = Brain("brain 3")
    brain3.load_state_dict(brain1.state_dict())

    compareParams(brain1, brain3)
    compareParams(brain2, brain3)
    
    normal = Brain("normal")
    evolution = Brain("evolution")
    test = Brain("normal2")
    test2 = Brain("evolution2")
    normal.load_state_dict(torch.load(r"C:/Users/allan/nvim/projects/evolutionSimulation/evolutionSimulation/weights/simpleModelWeights/8000/model2.pt"))
    test.load_state_dict(torch.load(r"C:/Users/allan/nvim/projects/evolutionSimulation/evolutionSimulation/weights/simpleModelWeights/5000/model2.pt"))
    evolution.load_state_dict(torch.load(r"C:/Users/allan/nvim/projects/evolutionSimulation/evolutionSimulation/weights/evolvedWeights/img2000/generation10/population100sheep0.pt"))
    test2.load_state_dict(torch.load(r"C:/Users/allan/nvim/projects/evolutionSimulation/evolutionSimulation/weights/evolvedWeights/img400/generation10/population100sheep0.pt"))
    brainStudy(evolution, see_all = False)
    brainStudy(normal, see_all = False)

    compareParams(normal, evolution)
    compareParams(normal, test)
    compareParams(test2, test)
