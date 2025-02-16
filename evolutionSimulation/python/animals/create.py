from evolutionSimulation.python.neuralnetworks.nn import * 
from datasets import load_dataset
import torch
import transformers
import time
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import random


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Animals():
    def __init__(self, species, name, brain_path, num, stats, status, mating, speed, eyesight, fov):
            brain = Brain()
            self.species = species
            self.name = name    
            if brain_path: 
                self.brain = brain.load_state_dict(torch.load(brain_path)).to(DEVICE)
            else:
                self.brain = brain
            brain.num = num
            brain.stats = stats
            brain.status = status
            brain.mating = mating
            brain.speed = speed 
            brain.eyesight = eyesight 
            brain.fov = fov
