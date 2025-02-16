#conda_env: evolution
from evolutionSimulation.python.neuralnetworks.nn import Brain
from evolutionSimulation.python.train.training import * 
from evolutionSimulation.python.eval.accuracy import * 
from evolutionSimulation.scripts.timer import timed
from evolutionSimulation.scripts.params import *
from datasets import load_dataset
import random

# Global vars
DATASET_PATH = r"C:/Users/allan/nvim/projects/evolutionSimulation/evolutionSimulation/python/dataset/simple_dataset.json" 
DEVICE = torch.device("cuda")

def generation(numMembers):
    """
    Creates a generation of the species, kept in list
    Args: 
        numMembers (int): Number of members in generation 

    Returns: 
        population (list): 🐑 🧠 (sheep brains)
    """
    population = []
    for i in range(numMembers):
        sheep = Brain(f"Sheep {i}")
        population.append([sheep.to(DEVICE), 0])
    return population

def dataset(dataset_path):
    """
    Loads dataset 

    Args: 
        dataset_path (str): self explanatory

    Returns: 
        dataset (dataset): Huggingface formatted dataset
    """
    return load_dataset("json", data_files = dataset_path)


def sheepPredation(generation, dataset, numImg, batchSize, treshhold, shouldPrint = False):
    """
    Compute accuracy of each member in the generation 

    Args: 
        generation (list): The list of all living members in the generation
        dataset (dataset): Huggingface formatted dataset
        numImg (int): Number of images to compute member on 
        batchSize (int): Number of images in a batch 

    Returns: 
        surivors (list): The list of all surviving members and their accuracy
    """
    # Shuffle dataset again for good measure
    survivors = []
    for i in range(len(generation)):
        dataset.shuffle()
        result = accuracy(dataset, numImg, batchSize, model = generation[i][0], weight_path=None) * 100
        #print(f"{result} || {(i/len(generation) * 100)}%")
        if i % 10 == 0:
            print("🐑", end = " ", flush=True)
        if result >= treshhold: 
            generation[i][1] = result
            survivors.append(generation[i])
    return survivors 

def addMembers(generation, numMembers):
    """
    Add members to a generation if too many are dead to increase gene pool  
    
    Args: 
        generation (list): The list of all living members in the generation
        numMembers (int): Number of members desired  

    Returns: 
        generation (list): The list of all surviving members  
    """

    desired = numMembers - len(generation)
    for i in range(desired):
        outsider = Brain(f"Outside Sheep {i}")
        generation.append([outsider, 0])
    return generation 

def procreate(father, mother, shouldSwap, shouldMerge, randomInt, layers = []):
    """
    Simulates procreation between 2 sheep 
   
   Args:
        father (Brain): 🐑 🧠 Custom CNN
        mother (Brain): 🐑 🧠 Custom CNN 
        shouldSwap (bool): Whether or not we should swap the layers 
        shouldMerge (bool): Whether or not we should merge the layers 
        random (float): How much randomness should be added to the model 
        layers (list): List of layers to swap layers 
        
    Returns: 
        child (Brain): Modified CNN
    """
    return modify(father, mother, shouldSwap, shouldMerge, randomInt, layers)

def newGeneration(oldGeneration, swap, merge, randomInt, skew = 20):
    """
    Creating the new generation by picking members of the old to breed, 
    skews to the left towards the members who have higher percentages; 
    Outsiders are less likely to breed, but still there for genetic variety 
    
    Args: 
        oldGeneration (list): Old generation of sheep 
        skew (int): How far we should skew the selection of sheep
    Returns: 
        newGenreation (list): New generation of sheep
    """
    newGeneration = []
    oldSort = sorted(oldGeneration, key=lambda x: x[1], reverse=True)
    print("Current best 10") 
    for i in range(10):
        print(oldSort[i][1])
    for i in range(len(oldGeneration)):
        x = random.randint(0, len(oldGeneration) - 1) - skew
        y = random.randint(0, len(oldGeneration) - 1) - skew
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        # really should just access father
        child = procreate(oldSort[x][0], oldSort[y][0], swap, merge, randomInt)
        newGeneration.append([child, 0])
    return newGeneration
    
def evolve(numMembers: int = 100, startingAccuracy : int = 50, geneticVariability : float = 0.5, shouldSwap: bool = False, shouldMerge: bool = True, numGenerations = 5, skew = 20, numImg = 40, batchSize = 40):

    """
    Finally! Our very simple evolution simulation! I'm very exicted to see if this will work
    """

    dataset = datasetTimed(DATASET_PATH) 
    firstgeneration = generationTimed(numMembers)
    
    # Initialize these variables for future use
    current_dataset = dataset.shuffle()
    current_generation = firstgeneration
    for i in range(numGenerations):
        current_treshold = startingAccuracy + (i * 3)
        survivors = sheepPredationTimed(current_generation, current_dataset, numImg, batchSize, current_treshold)
        survivors = addMembers(survivors, numMembers)
        print(f"Survivor length {len(survivors)}")
        current_generation = newGeneration(survivors, shouldSwap, shouldMerge, geneticVariability, skew)
        print(f"Completed Generation {i + 1}")

generationTimed = timed(generation)
datasetTimed = timed(dataset)
sheepPredationTimed = timed(sheepPredation)
accuracyTimed = timed(accuracy)

evolve()


"""
generationTimed = timed(generation)
datasetTimed = timed(dataset)
multipleModelInferenceTimed = timed(multipleModelInference)

test = generationTimed(100)
testset = datasetTimed(DATASET_PATH)
testset.shuffle()

sheepPredationTimed = timed(sheepPredation)

survivors = sheepPredationTimed(test, testset, 20, 20, 60)
print(len(survivors))
"""
