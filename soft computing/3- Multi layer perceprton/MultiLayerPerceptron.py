import numpy as np
import random
from random import randrange
from functools import reduce
import math

# input layer 4 neurons
# hidden layer 2 neurons
# output layer 4 neurons

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

#df = f * (1 - f)
              # input   # output
patterns = [
            ([1,0,0,0],[1,0,0,0]),
            ([0,1,0,0],[0,1,0,0]),
            ([0,0,1,0],[0,0,1,0]),
            ([0,0,0,1],[0,0,0,1])
            ]

learn_rate = 0.06
final_outz = {}
# set the weights randomly for the neurons
weights_l1 = {}
for i in range(0, 2):
    weights_l1[i] = []
    for j in range(0,4):
        weights_l1[i].append(round(random.uniform(-0.5, 0.5), 2))

weights_l2 = {}
for i in range(0, 4):
    weights_l2[i] = []
    for j in range(0,2):
        weights_l2[i].append(round(random.uniform(-0.5, 0.5), 2))

bias_1 = round(random.uniform(-0.5, 0.5), 2)
bias_2 = round(random.uniform(-0.5, 0.5), 2)

epoch = 0

# FEEDING FORWARD
used_patterns_index = []
while len(used_patterns_index) < len(patterns):
    pattern_index = randrange(len(patterns))
    output_secondLayer = []

    if pattern_index not in used_patterns_index:
        pattern = patterns[pattern_index]
        used_patterns_index.append(pattern_index)
        sums_secondLayer = []

        for i in range(2):
            # we add bias*1 to sum of all values
            sigma = np.dot(np.asarray(weights_l1[i]) , np.asarray(pattern[0]))+bias_1
            sums_secondLayer.append(sigma)
            output_secondLayer.append(sigmoid(sigma))

        sums_thirdLayer = []
        final_out = []
        for i in range(4):
            # we add bias_2 to sum of all values
            sigma = np.dot(np.array(weights_l2[i]) , np.array(output_secondLayer))+bias_2
            sums_thirdLayer.append(sigma)
            final_out.append(sigmoid(sigma))

        final_outz[pattern_index] = final_out

        # BACK PROPAGATION:
        third_layer_error = []
        second_layer_error = []

        # f'(s) = f(s)(1-f(s))
        # for both weights in each neuron the delta value will be the same since they both have same sum and output value!!
        delta = 0
        delta_array = []
        for i in range(len(weights_l2)):
            delta = sigmoid(sums_thirdLayer[i]) * (1-sigmoid(sums_thirdLayer[i])) * (pattern[1][i] - final_out[i])
            delta_array.append(delta)
            inner_weight = weights_l2[i]
            for j in range(len(inner_weight)):
                weights_l2[i][j] = weights_l2[i][j] + learn_rate * delta * output_secondLayer[j]
        bias_2 = bias_2 + learn_rate * delta

        delta_1 = 0
        for i in range(len(weights_l1)):
            inner_weight = weights_l1[i]
            # weights w1i w2i w3i w4i
            tempt_W = []
            for k, v in weights_l2.items():
                tempt_W.append(v[i])

            delta_1 = sigmoid(sums_secondLayer[i]) * (1-sigmoid(sums_secondLayer[i])) * np.dot(np.array(tempt_W), np.array(delta_array))
            for j in range(len(inner_weight)):
                weights_l1[i][j] = weights_l1[i][j] + learn_rate * delta_1 * patterns[pattern_index][0][j]
        bias_1 = bias_1 + learn_rate * delta_1

        # checking the accuracy of data :
        if len(used_patterns_index) == len(patterns):
            epoch += 1
            if epoch < 1000000:
                used_patterns_index = [] # so it will repeat the process
            else:
                print(final_outz)
                print(output_secondLayer)
