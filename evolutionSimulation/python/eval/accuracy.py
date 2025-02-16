#conda_env: evolution

import torch 
from evolutionSimulation.python.neuralnetworks.nn import Brain
from datasets import load_dataset
from evolutionSimulation.python.train.train_script import * 



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = load_dataset("json", data_files=r"C:/Users/allan/nvim/projects/evolutionSimulation/evolutionSimulation/python/dataset/simple_dataset.json")
shuffled_dataset = data.shuffle() 

def accuracy(dataset, num_img, batch_size, weight_path):
    """
    Computes accuracy of model
    Args: 
        Models: Brain class object
        dataset: datasset we want to use 

        num_img
    Returns: 
        Accuracy
    """
    total = 0
    correct = 0
    model = Brain()
    
    # If weight path specified, load it with state dict
    if weight_path: 
        model.load_state_dict(torch.load(weight_path))
    model.to(device) 

    # Iterate through dataset in batches
    for i in range(0, num_img, batch_size):
        temp_batch, truth = batch(batch_size, i, dataset)
        # Model inference, prediction size is (batch_size, num classes)
        predictions = model(temp_batch.to(device))
        output = torch.softmax(predictions, dim=1)
        for i in range(len(output)):
            total += 1
            # Compare the max of each row in the output to the ground truth. If same, count as correct
            if torch.argmax(output[i]) == truth[i]:
                correct += 1
    
    print(f"Accuracy: {correct/total * 100}%")
    return correct/total

