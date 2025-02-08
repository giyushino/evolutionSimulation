# evolutionSimulation
Testing out if I can replicate evolutionary adaptations seen in nature, as well as the natural improvements of neural networks

## Setting up with Anaconda  
To set up this project, clone the project and create a new Anaconda environment

```sh
conda create -n <my-env>
pip install -e .
```

## Setting up with venv 
```sh
python -m venv .venv
.venv\Scripts\activate 
pip install -e .
```


## To Do
- [x] Create class for animals 
- [x] Decide what the task for NN to solve is. MNIST? Binary classification?
- [ ] Test smallest number of training images to achieve good accuracy ~ 90% (4k seems fine)
- [ ] Test if one pooling layer is enough
- [ ] Decide what game engine to use ~ Godot seems promising due to C++ integration with GDExtension
- [ ] Decide how to assign neural networks to animals
- [ ] Something probably 
- [ ] When predators get close to pray, run an image through their nn, maybe do like a google doodle. If the prey can accurately do the binary classification, it has a 50% of escaping. if it doesn't, it has 0% chance of escaping. 

