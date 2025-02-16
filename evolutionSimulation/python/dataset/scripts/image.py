import matplotlib.pyplot as plt
from datasets import load_dataset
from PIL import Image
import numpy as np

data = load_dataset("json", data_files=r"C:\Users\allan\nvim\projects\evolutionSimulation\evolutionSimulation\python\dataset\dataset.json")

def view(dataset, dataset_type, image_index):
    """
    View an image from custom created hugginface dataset 

    Args:
        dataset (dataset): Custom dataset, hugginface format
        dataset_type (string): "train" or "test"
        image_index (int): which image to open
    Returns:
        image (Numpy array): Self explanatory
    """
    image = np.array(dataset[dataset_type][image_index]["image"], dtype=np.uint8)
    print(data[dataset_type][image_index]["name"])
    display = Image.fromarray(image)
    plt.imshow(display, cmap="gray") 
    plt.axis("off")  
    plt.show()

    return image

view(data, "train", 100000)

