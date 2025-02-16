#conda_env: evolution
from evolutionSimulation.python.neuralnetworks.nn import * 
from datasets import load_dataset
import torch
import transformers
import time
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import os

brain = Brain()
data = load_dataset("json", data_files=r"C:/Users/allan/nvim/projects/evolutionSimulation/evolutionSimulation/python/dataset/simple_dataset.json")
shuffled_dataset = data.shuffle()

animals = {
    "lion": 1,
    "crocodile": 1,
    "dragon": 1,
    "duck": 0,
    "sheep": 0,
}

def batch(batch_size, start_index, dataset):
    """
    Batch image into tensors

    Args: 
        batch_size (int): Number of examples to pass through model at once 
        start_index (int): Where to start in the dataset
        dataset (dataset): Dataset we want to train model on

    Returns:
        tensor.float() (tensor): Batched tensor that's now a float 
        truth (list): Ground truth labels, 0 or 1 for now
    """
    truth = []
    images = [sample for sample in dataset["train"][start_index:start_index + batch_size]["image"]]
    tensor = torch.tensor(images)
    tensor = tensor.view(batch_size, 1, 28, 28)
    for animal in dataset["train"][start_index:start_index + batch_size]["name"]:
        truth.append(animals[animal])
    return tensor.float(), truth


# Currently only works for 2 classes, 0/1. Could be be predator/prey, should change later if we want to do multiple classes
def train(num_img, batch_size, num_epoch, model, dataset):
    """
    Trains brain 🧠
    
    Arg: 
        num_img (int): Number of images to train CNN on 
        batch_size (int): Number of examples to pass through model at once 
        num_epoch (int): Number of epochs to train model on 
        model (Brain): Custom CNN 
        dataset (dataset): Dataset we want to train model on

    Returns:
        None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    model.to(device)
    model.train()
    best_loss = float("inf")
    total_loss = 0

    for epoch in range(num_epoch):
        epoch_loss = 0
        t0 = time.perf_counter()

        for i in range(0, num_img, batch_size):
            temp_batch = batch(batch_size, i, dataset)
            predictions = model(temp_batch[0].to(device))
            ground_truth = torch.tensor(temp_batch[1]).to(device, dtype = torch.long)
            loss_fn = nn.CrossEntropyLoss()

            loss = loss_fn(predictions, ground_truth)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if (i / num_img * 100) % 10 == 0:
                print(f"{i / num_img * 100}% | Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / (num_img // batch_size)
        total_loss += epoch_loss
        t1 = time.perf_counter()
        print(f"Finished Epoch {epoch} in {t1 - t0} seconds, Loss: {avg_loss:.4f}")
        try: 
            os.mkdir(r'C:\Users\allan\nvim\projects\evolutionSimulation\evolutionSimulation\modelWeights\{}'.format(num_img))
        except FileExistsError:
            pass
        torch.save(model.state_dict(), r'C:\Users\allan\nvim\projects\evolutionSimulation\evolutionSimulation\modelWeights\{}\model{}.pt'.format(num_img, epoch))
