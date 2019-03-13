import numpy as np
import random
from functools import reduce
import math

N = 0 # number of output neurons
M = 0 # number of input neurons

train_file = "train.txt"
test_file = "test.txt"

characters = dict()
weights = dict()

# Train Network
with open (train_file, 'r') as f:
    N = int(f.readline())
    M = reduce((lambda x, y : int(x) * int(y)), f.readline().split())

    curent_char = ""
    for line in f:
        line = line.replace('\n', '').replace('\r', '').strip()

        if len(line) == 1:
            characters[str(line)] = []
            curent_char = str(line)
        # this is the part of copying neuron :
        elif len(line) == 4:
            for x in line:
                if x == "#":
                    characters[curent_char].append(1.0)
                elif x == "-":
                    characters[curent_char].append(0.0)
# setting the weights (Feed Forward)
for k, v in characters.items():
    weights[k] = []
    v = np.asanyarray(v)
    sum = np.sum(v)
    weights[k] = v * (1 / math.sqrt(sum))

# Testing the network:
test_chars = {}
with open("test.txt", 'r') as f:
    i = 0
    test_chars[i] = []
    for line in f:
        line = line.replace('\n', '').replace('\r', '').strip()
        if len(line) == 4:
            for x in line:
                if x == "#":
                    test_chars[i].append(1.0)
                elif x == "-":
                    test_chars[i].append(0.0)
        if len(line) == 0:
            i += 1
            test_chars[i] = []



# flatten the array
# second neuron, use the weights and find which character resembles test instance the most
for k, v in test_chars.items():
    dist_list = []
    v = np.asanyarray(v)
    sum = np.sum(v)
    test_chars[k] = v * (1 / math.sqrt(sum))

    for k1, v1 in weights.items():
        distance = np.sum(abs(test_chars[k] - v1))
        dist_list.append(distance)
        print(k1, distance)
    index = dist_list.index(min(dist_list))
    if index == 0:
        print ("X is closest with distance of : ", dist_list[index])
    elif index == 1:
        print ("Y is closest with distance of : ", dist_list[index])
    elif index == 2:
        print ("Z is closest with distance of : ", dist_list[index])
    print('\n')
