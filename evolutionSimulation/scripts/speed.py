#conda_env: evolution

from evolutionSimulation.python.neuralnetworks.nn import Brain
from evolutionSimulation.python.neuralnetworks.ViT import CLIPModel
from evolutionSimulation.python.train.training import * 
from datasets import load_dataset
import time

def creationSpeed(numModels, model = Brain()):
    """
    Shows how long it takes to create certain number of models
    By default uses Brain cerated in nn.py
    Args: 
        numModels (int): Number of models to create 
        model (torch.nn): Which neural network to use, defaults to sheep brain

    Return: 
        None (Prints time taken to complete task)
    """
    store = []
    t0 = time.perf_counter()
    for i in range(numModels):
        store.append(Brain())
    t1 = time.perf_counter()
    
    length = len(f"|| Models: {numModels} || Time Taken: {(t1 - t0):.4f} || Time Per Model: {(t1 - t0)/numModels:.4f} ||")
    print("+" + ("-" * (length - 2)) + "+")
    print(f"|| Models: {numModels} || Time Taken: {(t1 - t0):.4f} || Time Per Model: {(t1 - t0)/numModels:.4f} ||")
    print("+" + ("-" * (length - 2)) + "+")

#creationSpeed(100, Brain())

def creationSpeedCLIP(numModels):
    """
    Shows how long it takes to create certain number of models
    By default uses Brain cerated in nn.py
    Args: 
        numModels (int): Number of models to create 
        model (torch.nn): Which neural network to use, defaults to sheep brain

    Return: 
        None (Prints time taken to complete task)
    """
    store = []
    t0 = time.perf_counter()
    for i in range(numModels):
        model = CLIPModel()
        store.append(model)
    t1 = time.perf_counter()
    
    length = len(f"|| Models: {numModels} || Time Taken: {(t1 - t0):.4f} || Time Per Model: {(t1 - t0)/numModels:.4f} ||")
    print("+" + ("-" * (length - 2)) + "+")
    print(f"|| Models: {numModels} || Time Taken: {(t1 - t0):.4f} || Time Per Model: {(t1 - t0)/numModels:.4f} ||")
    print("+" + ("-" * (length - 2)) + "+")
#creationSpeedCLIP(100)



def inferenceSpeed(deviceType, batchSizes, datasetPath, model = Brain()):
    """
    Checks the speed of different aspects of things in my project. 
    Times how long it takes for the program to load datasets, create 
    a Brain, and model inference 

    Args:
        deviceType (str): "cpu" or "cuda"
        batchSizes (list): List of batchSize integers
        model (Torch.nn): model to use, defaults to Brain
    Return:
        None
    """
    device = torch.device(deviceType)
    model.to(device) 
    t0 = time.perf_counter()
    dataset = load_dataset("json", data_files = datasetPath) 
    dataset.shuffle()
    t1 = time.perf_counter()
    print(f"Loaded Dataset in {(t1 - t0):4f}")
    print(f"Using {device}") 
    times = []
    length = len(f"|| Batch Size || Time Per Image || Total Time ||")
    print("+" + ("-" * (length-2)) + "+")
    print(f"|| Batch Size || Time Per Image || Total Time ||")
    print("||" + ("-" * (length-4)) + "||")
    for batchSize in batchSizes:
        t2 = time.perf_counter()
        tempBatch = batch(batchSize, 0, dataset)
        model(tempBatch[0].to(device))
        t3 = time.perf_counter()
        #print(f"||    {batchSize:<3}     ||    {((t3-t2)/batchSize):<6.6f}    ||   {t3-t2:<6.4f}   ||")
        times.append([batchSize, (t3-t2) / batchSize, t3 - t2])


    times = sorted(times, key = lambda x:x[1])
    for entry in times: 
        print(f"||    {entry[0]:<3}     ||    {entry[1]:<6.6f}    ||   {entry[2]:<6.4f}   ||")
    print("+" + ("-" * (length-2)) + "+")
inferenceSpeed("cuda", [1, 2, 4, 8, 16, 32, 64, 128, 256], r"C:/Users/allan/nvim/projects/evolutionSimulation/evolutionSimulation/python/dataset/dataset.json")
#inferenceSpeed("cpu", [1, 2, 4, 8, 16, 32, 64, 128, 256], r"C:/Users/allan/nvim/projects/evolutionSimulation/evolutionSimulation/python/dataset/dataset.json")



def checkSpeed(numModels, batchSizes, dataset = False):
    device = torch.device("cuda")
    brains = []
    t0 = time.perf_counter()
    
    # Create new CNN #numModels of time
    for i in range(numModels):
        temp = Brain()
        brains.append(temp.to(device))
    t1 = time.perf_counter()
    print(f"Took {t1 - t0:.4f} to create {numModels} CNNs. {(t1-t0)/numModels:.4f} per model")
    
    if dataset: 
        t2 = time.perf_counter()
        six = load_dataset("json", data_files = r"C:/Users/allan/nvim/projects/evolutionSimulation/evolutionSimulation/python/dataset/dataset.json")
        t3 = time.perf_counter()
        print(f"{t3 - t2:.4f} to load complex dataset with {len(six["train"])} entries")

    t4 = time.perf_counter()
    two = load_dataset("json", data_files = r"C:/Users/allan/nvim/projects/evolutionSimulation/evolutionSimulation/python/dataset/simple_dataset.json")
    two.shuffle()
    t5 = time.perf_counter()
    print(f"{t5 - t4:.4f} to load simple dataset with {len(two["train"])} entries") 
    
    times = []
    # Time model inference
    for batchSize in batchSizes:
        t6 = time.perf_counter()
        tempBatch = batch(batchSize, 0, two)
        brains[0](tempBatch[0].to(device))
        t7 = time.perf_counter()
        times.append([batchSize, (t7 - t6) / batchSize])
        #print(f"Took {t7 - t6:.4f} seconds to compute batch size {batchSize}. {(t7 - t6) / batchSize:.4f} seconds per image")
    
    times = sorted(times, key=lambda x: x[1])
    print("Fastest Batch Sizes")

    max_index_width = len(str(len(times)))
    max_batch_width = max(len(str(time[0])) for time in times)
    time_width = 16  
    label_width = len(" images || ")  
    total_width = max_index_width + max_batch_width + time_width + label_width + 18

    print("+" + "-" * total_width + "+")

    for i, (batch_size, foo) in enumerate(times, start=1):
        print(f"| {i:>{max_index_width}}. {batch_size:<{max_batch_width}} images || {foo:.10f} seconds per image |")

    print("+" + "-" * total_width + "+")

#checkSpeed(100, [x for x in range(1, 51, 5)], dataset = False)

