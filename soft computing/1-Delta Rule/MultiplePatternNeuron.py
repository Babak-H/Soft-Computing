import random
import numpy as np

N = 0 # number of neuron inputs
M = 0 # number of training patterns
k = 200 # number of training epochs
alpha = 0.003 # learning rate
interval = [-1, 1] # weight interval

file_name = "patterns2.txt"

inputX = []
outputZ = []
outputY = []
weights = []

with open(file_name, 'r') as f:
    first_line = f.readline()
    M = int(first_line.split()[2])
    next(f)
    for line in f:
        l = [float(x) for x in line.split()]
        N = len(l)-1
        inputX.append(l[:-1])
        outputZ.append(l[-1])

y = 0
for i in range(0, N):
    weights.append(round(random.uniform(interval[0], interval[1]), 2))


for i in range(0, k+1):
    weights = np.asanyarray(weights)
    inputX = np.asanyarray(inputX)
    for b in range(0, M):
        y = np.dot(weights, inputX[b])

        for j in range(0, len(weights)):
            weights[j] = weights[j] + alpha * (outputZ[b]-y) * inputX[b][j]

        if i == k: # in the last iteration we won't have any changes to our weight therefore Y value will stay the same
            outputY.append(y)

print(outputY)

