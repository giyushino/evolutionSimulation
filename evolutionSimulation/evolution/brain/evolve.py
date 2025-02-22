#conda_env: evolution
from evolutionSimulation.python.neuralnetworks.nn import Brain
from evolutionSimulation.python.train.training import * 
from evolutionSimulation.python.eval.accuracy import * 
from evolutionSimulation.scripts.timer import timed
from evolutionSimulation.scripts.params import *
from datasets import load_dataset
import random
import math


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


def sheepPredation(generation, dataset, numImg, batchSize, treshold, shouldPrint = False):
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
    dataset.shuffle()
    for i in range(len(generation)):
        result = accuracy(dataset, numImg, batchSize, model = generation[i][0], weight_path=None) * 100
        print(f"\r🐑 {i + 1}/{len(generation)} || Acccuracy: {result:.2f}% {'| survived' if result >= treshold else '| died    '}", end="", flush=True)
        if result >= treshold: 
            generation[i][1] = result
            survivors.append(generation[i])
    print(f"\n{len(survivors)} sheep survived!")
    return survivors 

def addMembers(generation, numMembers, dataset, numImg, batchSize, threshold):
    """
    Add members to a generation if too many are dead to increase gene pool  
    
    Args: 
        generation (list): The list of all living members in the generation
        numMembers (int): Number of members desired  

    Returns: 
        generation (list): The list of all surviving members  
    """
    
    desired = numMembers - len(generation)
    print(f"{desired} 🐑 are migrating in! Threshold || {threshold}")
    count = 0
    used = 0
    while used < desired:
        outsider = Brain(f"Outside Sheep {count}")
        result = accuracy(dataset, numImg, batchSize, model = outsider, weight_path=None) * 100
        if result >= threshold:
            used += 1
            print(f"\r🐑 {count + 1} || Accuracy: {result:.2f}% {'| can mate! '} || {used} 🐑 have entered", end="", flush=True)
            generation.append([outsider.to(DEVICE), result])
        else:
            print(f"\r🐑 {count + 1} || Accuracy: {result:.2f}% {'| forced out'} || {used} 🐑 have entered", end="", flush=True)

        count += 1
    print("\n")
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

def newGeneration(oldGeneration, swap, merge, randomInt, numMembers, skew = 5):
    """
    Creating the new generation by picking members of the old to breed, 
    skews to the left towards the members who have higher percentages; 
    Outsiders are less likely to breed, but still there for genetic variety 
    
    Args: 
        oldGeneration (list): Old generation of sheep 
        skew (int): How far we should skew the selection of sheep
        numMembers (int): How many members to include in the population 
    Returns: 
        newGeneration (list): New generation of sheep
    """
    newGeneration = []
    skew = int(len(oldGeneration)/skew)
    oldSort = sorted(oldGeneration, key=lambda x: x[1], reverse=True)
    print("Current smartest 10 🐑")
    print("+--------------------+")
    for i in range(10):
        print(f"|| Accuracy || {(oldSort[i][1]):.1f} ||")
    print("+--------------------+")
    print("\n")
    for i in range(len(oldGeneration)):
        x = random.randint(0, len(oldGeneration) - 1) - skew
        y = random.randint(0, len(oldGeneration) - 1) - skew
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x == y: 
            y = random.randint(int(len(oldGeneration)/2), len(oldGeneration) - 1)
        # really should just access father
        child = procreate(oldSort[y][0], oldSort[x][0], swap, merge, randomInt)
        newGeneration.append([child, 0])
    return newGeneration


def newGenerationAdd(oldGeneration, dataset, swap, merge, randomInt, numMembers, numImg, batchSize, skew = 5):
    """
    Creating the new generation by picking members of the old to breed, 
    skews to the left towards the members who have higher percentages; 
    Outsiders are less likely to breed, but still there for genetic variety 
    
    Args: 
        oldGeneration (list): Old generation of sheep 
        skew (int): How far we should skew the selection of sheep
        numMembers (int): How many members to include in the population 
    Returns: 
        newGeneration (list): New generation of sheep
    """
    newGeneration = []
    oldSort = sorted(oldGeneration, key=lambda x: x[1], reverse=True)
    print("Current best 10")
    for i in range(min(10, len(oldSort))):
        print(oldSort[i][1])
    
    y = 0
    while len(newGeneration) <= numMembers: 
        x = random.randint(0, len(oldGeneration) - 1) - skew
        if x < 0:
            x = 0
        while y == x: 
            y = random.randint(0, len(oldGeneration) - 1)
        child = procreate(oldSort[y][0], oldSort[x][0], swap, merge, randomInt)
        newGeneration.append([child, accuracy(dataset.shuffle(), numImg, batchSize, model = child, weight_path=None) * 100])
    
    return newGeneration

def calculateThreshold(startingAccuracy: int , generation: int, spread: int = 5) -> float:
    """
    Accuracy threshold necessary to survive
    
    Args: 
        startingAccuracy (int): The accuracy to first weed sheep out 
        generation (int): Which generation the sheep are on 
        spread (int): Really should be dependent on generation but 🤷
        
    Returns: 
        coefficient * ln (float): The accuracy the current generation needs to pass 
    """
    ln = spread * math.log(generation + 1) + startingAccuracy
    return ln


def evolve(numMembers: int = 20, startingAccuracy : int = 55, geneticVariability : float = 0.5, shouldSwap: bool = False, shouldMerge: bool = True, numGenerations = 50, skew = 18, numImg = 1000, batchSize = 100):
    """    
    Simulates the evolution of a population over a specified number of generations
    
    Args:
        numMembers (int): Number of members in the initial generation
        startingAccuracy (int): Initial accuracy threshold for survival
        geneticVariability (float): Variability in genetic traits
        shouldSwap (bool): Whether to allow swapping of genetic material
        shouldMerge (bool): Whether to allow merging of genetic material
        numGenerations (int): Number of generations to simulate
        skew (int): Skew factor for genetic variability
        numImg (int): Number of images to use in the dataset
        batchSize (int): Size of the batch for processing images
    
    Returns:
        None
    """
    label_width = 25
    value_width = 10

    # Store the lines of information
    lines = [
        f"|| {'population size:'.ljust(label_width)} {str(numMembers).rjust(value_width)} || {'generations:'.ljust(label_width)} {str(numGenerations).rjust(value_width)} ||",
        f"|| {'merge:'.ljust(label_width)} {str(shouldMerge).rjust(value_width)} || {'swap:'.ljust(label_width)} {str(shouldSwap).rjust(value_width)} ||",
        f"|| {'starting accuracy:'.ljust(label_width)} {str(startingAccuracy).rjust(value_width)} || {'images:'.ljust(label_width)} {str(numImg).rjust(value_width)} ||",
        f"|| {'genetic variability:'.ljust(label_width)} {str(geneticVariability).rjust(value_width)} || {'batch size:'.ljust(label_width)} {str(batchSize).rjust(value_width)} ||"
    ]

    # Determine the box width based on the longest line
    box_width = max(len(line) for line in lines)

    # Print the top border
    print("+" + "-" * (box_width - 2) + "+")

    # Print the lines inside the box
    for line in lines:
        print(line)

    # Print the bottom border
    print("+" + "-" * (box_width - 2) + "+")
    dataset = datasetTimed(DATASET_PATH) 
    firstgeneration = generationTimed(numMembers)
    
    # Initialize these variables for future use
    current_dataset = dataset.shuffle()
    current_generation = firstgeneration
    for i in range(numGenerations):
        text = f" Generation {i} "
        border = "+" + "-" * (len(text) + 2) + "+"
        print(border)
        print(f"||{text}||")
        print(border) 
       
        # rewrite so that it slows down as we approach 90 
        current_treshold = calculateThreshold(startingAccuracy, i, 6)
        print(f"Current Threshold || {current_treshold} %")
        if current_treshold > 80:
            # no need to shoot super high yet
            current_treshold = 80

        survivors = sheepPredationTimed(current_generation, current_dataset, numImg, batchSize, current_treshold)
        survivors = addMembers(survivors, numMembers, current_dataset, numImg, batchSize, current_treshold)
        current_generation = newGeneration(survivors, shouldSwap, shouldMerge, geneticVariability, numMembers, skew)
     
    best = sorted(current_generation, key=lambda x: x[1], reverse=True)
    for i in range(3):
        try: 
            os.makedirs(r'C:\Users\allan\nvim\projects\evolutionSimulation\evolutionSimulation\weights\evolvedWeights\img{}\generation{}'.format(numImg, numGenerations))
        except FileExistsError:
            pass
        torch.save(best[i][0].state_dict(), r'C:\Users\allan\nvim\projects\evolutionSimulation\evolutionSimulation\weights\evolvedWeights\img{}\generation{}\population{}sheep{}.pt'.format(numImg, numGenerations, numMembers, i))

generationTimed = timed(generation)
sheepPredationTimed = timed(sheepPredation)
datasetTimed = timed(dataset)
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

