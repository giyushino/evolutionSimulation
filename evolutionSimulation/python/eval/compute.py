#conda_env: evolution

import torch 
from evolutionSimulation.python.neuralnetworks.nn import Brain
from datasets import load_dataset
from evolutionSimulation.python.train.training import * 
from evolutionSimulation.scripts.timer import timed
from evolutionSimulation.python.eval.accuracy import *

DEVICE = torch.device("cuda")

@timed
def computeNormal(img, sheep, shuffled_dataset, numImg = 400, batchSize = 20,  shouldPrint = True):
    """
    Computes accuracy of specified brain 
    """ 
    test = accuracy(shuffled_dataset, num_img = numImg, batch_size = batchSize, model = None, weight_path = r"C:/Users/allan/nvim/projects/evolutionSimulation/evolutionSimulation/weights/simpleModelWeights/{}/model{}.pt".format(img, sheep), shouldPrint = shouldPrint)
    return test
@timed
def computeEvolution(img, generation, population, sheep, dataset, numImg = 400, batchSize = 20,  shouldPrint = True):
    """
    Computes accuracy of specified brain 
    """
    test = accuracy(dataset, num_img = numImg, batch_size = batchSize, model = None, weight_path = r"C:/Users/allan/nvim/projects/evolutionSimulation/evolutionSimulation/weights/evolvedWeights/img{}/generation{}/population{}sheep{}.pt".format(img, generation, population, sheep), shouldPrint = shouldPrint)
    return test


if __name__ == "__main__":
    dataset_path = r"C:/Users/allan/nvim/projects/evolutionSimulation/evolutionSimulation/python/dataset/simple_dataset.json" 
    data = load_dataset("json", data_files = dataset_path)
    while True:
        shuffled_dataset = data.shuffle()
        computeEvolution(100, 5, 10, 0, shuffled_dataset)
    #computeNormal(8000, 2, shuffled_dataset)
    #computeEvolution(200, 100, 50, 0, shuffled_dataset)
    #computeNormal(5000, 2, shuffled_dataset)
    #computeEvolution(400, 10, 100, 0, shuffled_dataset)
