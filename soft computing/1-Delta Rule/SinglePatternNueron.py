import random
import numpy as np

N = 0 # number of neuron inputs
M = 0 # number of training patterns
k = 100 # number of training epochs
alpha = 0.005 # learning rate
interval = [-1, 1] # weight interval

file_name = "patterns1.txt"

inputX = []
outputZ = []
weights = []
with open(file_name, 'r') as f:
    first_line = f.readline()
    M = first_line.split()[2]
    next(f)
    for line in f:
        l = [float(x) for x in line.split()]
        N = len(l)-1
        inputX.append(l[:-1])
        outputZ.append(l[-1])


# since it is single pattern we train it with only one set
inputX = inputX[0]
outputZ = outputZ[0]

y = 0
for i in range(0, N):
    weights.append(round(random.uniform(interval[0], interval[1]), 2))


for i in range(0, k+1):
    weights = np.asanyarray(weights)
    inputX = np.asanyarray(inputX)
    y= np.dot(weights, inputX)
    for j in range(0, len(weights)):
        weights[j] = weights[j] + alpha * (outputZ-y) * inputX[j]

print("output : ", y)
