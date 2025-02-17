#conda_env: evolution
import torch 
from evolutionSimulation.python.neuralnetworks.nn import Brain
from datasets import load_dataset
from evolutionSimulation.python.train.training import * 
from evolutionSimulation.scripts.timer import timed
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def accuracy(dataset, num_img, batch_size, model = None, weight_path = None, shouldPrint = False):
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
    if model == None:
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
    if shouldPrint: 
        print(f"Accuracy: {correct/total * 100}%")
    return correct/total

# Make more modular i guess for future use 
#accuracyTimed = timed(accuracy)
def multipleModelInference(dataset, num_img, batch_size, repeat):
    t0 = time.perf_counter()
    for i in range(repeat):
        accuracy(dataset, num_img, batch_size, model = None, weight_path=r"C:/Users/allan/nvim/projects/evolutionSimulation/evolutionSimulation/weights/simpleModelWeights/10000/model2.pt")
    t1 = time.perf_counter()
    print(f"took {(t1 - t0):.4f} seconds")
data = load_dataset("json", data_files=r"C:/Users/allan/nvim/projects/evolutionSimulation/evolutionSimulation/python/dataset/simple_dataset.json")
shuffled_dataset = data.shuffle() 

accuracy = timed(accuracy)
test = accuracy(shuffled_dataset, num_img = 4000, batch_size = 200, model = None, weight_path = r"C:/Users/allan/nvim/projects/evolutionSimulation/evolutionSimulation/weights/evolvedWeights/400/generation10sheep1.pt", shouldPrint = True)
print(test)
