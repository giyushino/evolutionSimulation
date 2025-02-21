# evolutionSimulation
Testing out if I can replicate evolutionary adaptations seen in nature, as well as the natural improvements of neural networks. Currently testing if a binary classification CNN trained to differentiate between lions and sheep can achieve relatively high accuracy simply through a rudimentary implementation of evolution.

## Current Features 
Evolution is currently simulated by iterating through generations of sheep. An initial population of size n is created, where 

## Setting up with Anaconda  
To set up this project, clone the project and create a new Anaconda environment

```sh
conda env create -f environment.yml
```

After this, install [the proper version of Pytorch with GPU support for your device.](https://pytorch.org/get-started/locally/)
I'm using CUDA 12.6
```sh
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

To allow imports across the project, run 
```sh
pip install -e .
```

## To Do
- [x] Create class for animals 
- [x] Decide what the task for NN to solve is. MNIST? Binary classification?
- [x] Test smallest number of training images to achieve good accuracy ~ 90% (5k seems fine)
- [x] GET THE EVOLUTION ACTUALLY WORKING
- [ ] Test on CPU (normal + evolve)
- [ ] Speed up evolution (lower outsider sheep)
- [ ] Test with more classes 
- [ ] Add venv + uv accessibility 


