#conda_env: evolution
from nn import Brain
import torch
import torch.nn as nn 


def brainStudy(model, see_all = False):
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
    return model

def paramSwap(model, swap = None):
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


brain1 = Brain()
brains = ["C:/Users/allan/nvim/projects/evolutionSimulation/evolutionSimulation/modelWeights/10000/model2.pt", "C:/Users/allan/nvim/projects/evolutionSimulation/evolutionSimulation/modelWeights/5000/model2.pt", "C:/Users/allan/nvim/projects/evolutionSimulation/evolutionSimulation/modelWeights/100/model2.pt"]

brainWeights = []

#brain1.load_state_dict(torch.load(r"C:/Users/allan/nvim/projects/evolutionSimulation/evolutionSimulation/modelWeights/10000/model2.pt"))

for i in brains:
    temp_brain = Brain()
    temp_brain.load_state_dict(torch.load(r"{}".format(i)))
    brainWeights.append(temp_brain)

brainWeights.append(brain1)


print("Compare 10000, 10000")
compareParams(brainWeights[0], brainWeights[0])
print("--------------------------------")

print("Compare 10000, 5000")
compareParams(brainWeights[0], brainWeights[1])
print("--------------------------------")

print("Compare 10000, 100")
compareParams(brainWeights[0], brainWeights[2])
print("--------------------------------")

print("Compare 10000, base")
compareParams(brainWeights[0], brainWeights[3])
print("--------------------------------")

print("Compare 5000, 100")
compareParams(brainWeights[1], brainWeights[2])
print("--------------------------------")

print("Compare 5000, base")
compareParams(brainWeights[1], brainWeights[3])
print("--------------------------------")

print("Compare 100, base")
compareParams(brainWeights[2], brainWeights[3])
print("--------------------------------")

print("Compare base, base")
compareParams(brainWeights[3], brainWeights[3])
print("--------------------------------")
