import random
import numpy as np
import math

# optimize the function : f(x) = (e^x * sin(10px) + 1) for maximum
# members of each chromosome m = 22
chrom_length = 22
# precision
p = 4
# min and max
a = 0.5
b = 2.5
Pop_size = 50
Num_attempt = 10
alpha = 0.1
# Number of of times that we attemp whole SGA
Run = 0
# current generation(epoch) of the current SGA Run
Gen = 0
# the population array
pop = []
# averge of the fitness for pops in different generations
pops_avg = []
solution = 0

# initialize the population
for i in range(0, Pop_size):
    pop.append([])
    for j in range(0, chrom_length):
        pop[i].append(random.choice([0,1]))


# fitness function
def calc_fitness(arr):
    final_num = calc_realNum(arr)
    return (math.exp(final_num) * math.sin(10 * 3.14 * final_num)) / final_num + 5


def calc_realNum(arr):
    # calculate array x into a real Number
    under = (2 ** chrom_length) - 1
    i = 0
    sum = 0
    for num in arr[::-1]:
        if num != 0:
            sum += 2**i
        i+=1
    f_num = a + (b-a) * (sum / under)
    return f_num


# should we stop the test or not function
def should_stop():
    sum = 0
    for sol in pop:
        sum += calc_fitness(sol)
    avg = sum / len(pop)
    if (len(pops_avg) > 0) and (avg - pops_avg[-1] < alpha):
        return 1
    else:
        pops_avg.append(avg)
        return 0


while Run < Num_attempt:

    if should_stop() == 1:
        Run += 1
        # finding the best solution:
        resultz = []
        for arr in pop:
            resultz.append(calc_fitness(arr))

        print(max(resultz))
        index = resultz.index(max(resultz))
        solution = pop[index]
        print(calc_realNum(solution))

        pops_avg = []
        pop = []
        # initialize the population
        for i in range(0, Pop_size):
            pop.append([])
            for j in range(0, chrom_length):
                pop[i].append(random.choice([0, 1]))




    else:
        # fitness function for all elements of population
        pops_fitness = []
        next_pop = []
        sum = 0
        for arr in pop:
            sum += calc_fitness(arr)
            pops_fitness.append(calc_fitness(arr))
        selection_probs = [x/sum for x in pops_fitness]
        q = np.cumsum(np.array(selection_probs))

        # operations to transform into next generation:
        iters = Pop_size

        while iters > 0:
            # choose a solution from pop
            chosen = 0
            while chosen == 0:
                rand = random.uniform(0, 1)
                for i in range(1, len(q)):
                    if q[i-1] < rand < q[i]:
                        chosen = pop[i]
                        break

            # select which operation to do:
            r = random.uniform(0, 1)
            # crossover
            if r < 0.45:
                chosen_2 = 0
                while chosen_2 == 0:
                    rand = random.uniform(0, 1)
                    for i in range(1, len(q)):
                        if q[i-1] < rand < q[i]:
                            chosen_2 = pop[i]
                            break

                cross_point = random.randrange(len(chosen))
                next_pop.append(chosen[:cross_point]+chosen_2[cross_point:])
                iters -= 2
            # mutate
            elif 0.55 >= r >= 0.45:
                chosen[random.randrange(len(chosen))] = random.choice([0,1])
                next_pop.append(chosen)
                iters -= 1
            elif r > 0.55: # reproduce
                next_pop.append(chosen)
                iters -= 1

        if iters == 0:
            pop = next_pop.copy()
            next_pop = []

