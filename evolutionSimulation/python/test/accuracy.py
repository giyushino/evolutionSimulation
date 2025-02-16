#conda_env: evolution

import torch 
from evolutionSimulation.python.neuralnetworks.nn import Brain
from datasets import load_dataset
from evolutionSimulation.python.train.train_script import * 


brain = Brain()
print("Loaded CNN")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = load_dataset("json", data_files=r"C:/Users/allan/nvim/projects/evolutionSimulation/evolutionSimulation/python/dataset/simple_dataset.json")
print("Loaded Dataset")
shuffled_dataset = data.shuffle() 

def accuracy(model,  dataset, num_img, batch_size, weight_path):
    """
    Computes accracy of model
    Args: 
        Models: Brain class object
        dataset: datasset we want to use 

        num_img
    Returns: 
        Accuracy
    """
    total = 0
    correct = 0
    if weight_path: 
        model.load_state_dict(torch.load(weight_path))
    model.to(device) 
    for i in range(0, num_img, batch_size):
        temp_batch, truth = batch(batch_size, i, dataset)
        predictions = model(temp_batch.to(device))
        output = torch.softmax(predictions, dim=1)
        for i in range(len(output)):
            total += 1
            if torch.argmax(output[i]) == truth[i]:
                correct += 1
    
    print(f"Accuracy: {correct/total * 100}%")
    return correct/total
    

    
print("Accuracy for default: Expected 50%")
accuracy(brain, shuffled_dataset, 500, 10, weight_path=None)
print("----------------------------------")

print("Accuracy for model trained on 100:")
accuracy(brain, shuffled_dataset, 500, 10, weight_path=r'C:/Users/allan/nvim/projects/evolutionSimulation/evolutionSimulation/modelWeights/100/model2.pt')
print("----------------------------------")

print("Accuracy for model trained on 5000:")
accuracy(brain, shuffled_dataset, 500, 10, weight_path=r'C:/Users/allan/nvim/projects/evolutionSimulation/evolutionSimulation/modelWeights/5000/model2.pt')
print("----------------------------------")

print("Accuracy for model trained on 10000:")
accuracy(brain, shuffled_dataset, 500, 10, weight_path=r'C:/Users/allan/nvim/projects/evolutionSimulation/evolutionSimulation/modelWeights/10000/model2.pt')
print("----------------------------------")

print("Accuracy for model trained on 30000:")
accuracy(brain, shuffled_dataset, 500, 10, weight_path=r'C:/Users/allan/nvim/projects/evolutionSimulation/evolutionSimulation/modelWeights/30000/model2.pt')
print("----------------------------------")

print("Accuracy for model trained on 50000:")
accuracy(brain, shuffled_dataset, 500, 10, weight_path=r'C:/Users/allan/nvim/projects/evolutionSimulation/evolutionSimulation/modelWeights/50000/model2.pt')
print("----------------------------------")
