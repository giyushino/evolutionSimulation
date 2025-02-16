from datasets import load_dataset
import numpy as np
import json
import os 

def create_dataset(directory = r"C:/Users/allan/nvim/projects/evolutionSimulation/evolutionSimulation/data", save_path = r"dataset.json"):
    """
    Function to create dataset; Has 5 classes

    Args: 
        directory (str): Path where data .npy files are saved
        save_path (str): Path to save dataset 

    Returns:
        dataset (dataset): Loaded Huggingface dataset
    """

    temp = []
    names = ["crocodile", "dragon", "duck", "lion", "sheep"]

    for name in os.listdir(directory):
        temp.append(np.load(os.path.join(directory, name)))

    reshaped = []
    for animal in temp:
        bruh = []
        for i in range(len(animal)):
            bruh.append(animal[i].reshape(28,28))
        reshaped.append(bruh)
    
    count = 0
    
    with open(save_path, "w") as file:
        for i in range(len(reshaped)):
            for j in range(len(reshaped[i])):
                line = {"name": names[i], "image": reshaped[i][j].tolist()}
                file.write(json.dumps(line) + "\n")
                count += 1
                print(f"Line {count} created")
    
    dataset = load_dataset("json", data_files=save_path)
    return dataset

#create_dataset()

def create_dataset_simple(save_path, directory = r"C:\Users\allan\nvim\evolutionSimulation\evolutionSimulation\data"):
    """
    Simpler version of create_dataset, only 2 classes: lion and sheep
    
    Args: 
        directory (str): Path where data .npy files are saved
        save_path (str): Path to save dataset 

    Returns:
        dataset (dataset): Loaded Huggingface dataset
    """
    temp = []
    names = ["lion", "sheep"]
    """change for later"""
    temp.append(np.load(r"C:\Users\allan\Downloads\full_numpy_bitmap_lion.npy"))

    temp.append(np.load(r"C:\Users\allan\Downloads\full_numpy_bitmap_sheep.npy"))
    
    reshaped = []
    for animal in temp:
        bruh = []
        for i in range(len(animal)):
            bruh.append(animal[i].reshape(28,28))
        reshaped.append(bruh)
    
    count = 0
    
    with open(save_path, "w") as file:
        for i in range(len(reshaped)):
            for j in range(len(reshaped[i])):
                line = {"name": names[i], "image": reshaped[i][j].tolist()}
                file.write(json.dumps(line) + "\n")
                count += 1
                print(f"Line {count} created")
    
    dataset = load_dataset("json", data_files=save_path)
    return dataset

#create_dataset_simple(save_path = "simple_dataset.json")
