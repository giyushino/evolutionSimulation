#conda_env: evolution

from evolutionSimulation.python.neuralnetworks.nn import Brain
from evolutionSimulation.python.train.train_script import * 
from datasets import load_dataset
import time

temp = []

device = torch.device("cuda")

brains = ["C:/Users/allan/nvim/projects/evolutionSimulation/evolutionSimulation/simpleModelWeights/10000/model2.pt", "C:/Users/allan/nvim/projects/evolutionSimulation/evolutionSimulation/simpleModelWeights/5000/model2.pt", "C:/Users/allan/nvim/projects/evolutionSimulation/evolutionSimulation/simpleModelWeights/100/model2.pt"]

brainWeights = []
for i in brains:
    temp_brain = Brain()
    temp_brain.load_state_dict(torch.load(r"{}".format(i)))
    brainWeights.append(temp_brain.to(device))

t0 = time.perf_counter()
for i in range(100):
    temp_brain = Brain()
    temp.append(temp_brain.to(device))
t1 = time.perf_counter()

t2 = time.perf_counter()
data = load_dataset("json", data_files=r"C:/Users/allan/nvim/projects/evolutionSimulation/evolutionSimulation/python/dataset/simple_dataset.json")
t3 = time.perf_counter()

t4 = time.perf_counter()
temp_batch = batch(1, 0, data)
print(temp_batch)

prediction = brainWeights[0](temp_batch[0].to(device))
t5 = time.perf_counter()
print(prediction, temp_batch[1])

t6 = time.perf_counter()
temp_batch = batch(10, 0, data)
print(temp_batch)

prediction = brainWeights[0](temp_batch[0].to(device))
t7 = time.perf_counter()
print(prediction, temp_batch[1])

t8 = time.perf_counter()
temp_batch = batch(100, 0, data)
print(temp_batch)

prediction = brainWeights[0](temp_batch[0].to(device))
t9 = time.perf_counter()
print(prediction, temp_batch[1])

t10 = time.perf_counter()
temp_batch = batch(20, 0, data)
print(temp_batch)

prediction = brainWeights[0](temp_batch[0].to(device))
t11 = time.perf_counter()
print(prediction, temp_batch[1])


print(f"Time taken to load dataset: {t3 - t2}")
print(f"Time taken to create 100 brains: {t1 - t0}")
print(f"Time taken to create batch + inference of size 1: {t5 - t4}")
print(f"Time taken to create batch + inference of size 10: {t7 - t6}")
print(f"Time taken to create batch + inference of size 20: {t11 - t10}")
print(f"Time taken to create batch + inference of size 100: {t9 - t8}")
