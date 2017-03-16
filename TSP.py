from collections import defaultdict
import operator
import numpy as np
import random
import math
from random import randint
from random import shuffle


# to remove extra spaces while reading from the files provided
def space_remover(str):
    output = []
    for i in range(len(str)):
        if str[i] != '':
            output.append(str[i])
    return output


def choromosome_validity(distance_matrix, path_list):
    costMatrix = np.array(distance_matrix)
    for i in range(0, len(path_list) - 1):
        city_a = path_list[i]
        city_b = path_list[i + 1]
        if costMatrix[city_a - 1][city_b - 1] == 0:
            return False
    return True


# to return the names of all cities in a list
def get_cities(filename):
    file = open(filename)
    data = file.readlines()
    cities = []
    for i in data:
        i = i.strip('\n')
        for index in range(len(i)):
            if i[index] == ',':
                i = i[:index]
                break
        cities.append(i)
    return cities


# to return the distances of each city from others in a 312 x 312 matrix
def nxn_distance_matrix(filename, total_cities):
    file = open(filename)
    data = file.readlines()
    row_counter = 0
    iteration = 0
    distance_matrix = np.zeros(shape=(total_cities, total_cities))
    for str in data:
        str = str.strip("\n")
        str = str.split(" ")
        str = space_remover(str)
        for j in range(len(str)):
            distance_matrix[row_counter][j + iteration] = str[j]
        iteration += len(str)

        if len(str) == 2:
            row_counter += 1
            iteration = 0

    return distance_matrix


# returns starting and remaining city numbers
def get_starting_and_other_points(cities):
    initial = [1]
    other = []
    for i in range(2, len(cities) + 1):
        other.append(i)
    return (initial, other)


# returns randomly generated chromosome
def get_choromosome(initial, other, distance_matrix):
    chromosome = []
    while True:
        random_arrangement = list(np.random.permutation(other))
        chromosome = initial + random_arrangement
        if choromosome_validity(distance_matrix, chromosome) == True:
            break
    return chromosome


# generating chromosomes, equal to the size of population [inputted]
def get_initial_population(population_size, initial, other, distance_matrix):
    population = []
    for i in range(population_size):
        population.append(get_choromosome(initial, other, distance_matrix))
    return population


def calculateFitness(distances, choromosome):
    distance_matrix = np.array(distances)
    fitness = 0
    for i in range(len(choromosome) - 1):
        city_a = choromosome[i]
        city_b = choromosome[i + 1]
        fitness = fitness + distance_matrix[city_a - 1][city_b - 1]
    return -(fitness)


def swap_mutation(ch, mutation_rate):
    values_to_mutate = len(ch) * mutation_rate
    for i in range(math.ceil(values_to_mutate)):
        random_city_1 = randint(0, len(ch) - 1)
        random_city_2 = randint(0, len(ch) - 1)
        ch[random_city_1], ch[random_city_2] = ch[random_city_2], ch[random_city_1]
    return ch


def inversion_mutation(ch, mutation_rate):
    values_to_mutate = len(ch) * mutation_rate

    for i in range(math.ceil(values_to_mutate)):
        point1 = point2 = 0

        # generating two random numbers
        while point1 == point2:
            point1 = randint(1, len(ch) - 1)
            point2 = randint(1, len(ch) - 1)

        # smaller value always in point1
        if point1 > point2:
            point1, point2 = point2, point1

        while point1 < point2:
            ch[point1], ch[point2] = ch[point2], ch[point1]
            point1 += 1
            point2 -= 1

    return ch


def best_chromosome_after_mutation(ch, mutation_rate, distances):
    i = 0
    maxFitness = -9999999999              # minimum value
    parallelChromosome = []
    while i != 10:

        clone = ch
        if i%2 == 0:
            clone = swap_mutation(clone, mutation_rate)
        else:
            clone = inversion_mutation(clone, mutation_rate)

        currFitness = calculateFitness(distances, clone)
        if (currFitness > maxFitness):
            maxFitness = currFitness
            parallelChromosome = clone
        i += 1


    # return the one with the best fitness
    return parallelChromosome


def one_point_crossover(ch1, ch2, mutation_rate, distances):
    offspring1 = [None] * len(ch1)
    offspring2 = [None] * len(ch1)

    # Loop 100 times for unique cities, high probability of failure
    for i in range (100):

        # generating a random number
        i = 0
        cross_over_point = randint(1, len(ch1) - 1)

        while i != cross_over_point:
            offspring1[i] = ch2[i]
            offspring2[i] = ch1[i]
            i += 1

        while i != len(ch1):
            offspring1[i] = ch1[i]
            offspring2[i] = ch2[i]
            i += 1

        # to check if all cities are unique or not
        if not len(offspring1) > len(set(offspring1)):
            if not len(offspring2) > len(set(offspring2)):
                offspring1 = best_chromosome_after_mutation(offspring1, mutation_rate, distances)
                offspring2 = best_chromosome_after_mutation(offspring2, mutation_rate, distances)
                pop = []
                pop.append(offspring1)
                pop.append(offspring2)
                return pop

    # else return the original mutated chromosomes
    ch1 = best_chromosome_after_mutation(ch1, mutation_rate, distances)
    ch2 = best_chromosome_after_mutation(ch2, mutation_rate, distances)
    pop = []
    pop.append(ch1)
    pop.append(ch2)
    return pop


def two_point_crossover(ch1, ch2, mutation_rate, distances):
    offspring1 = [None] * len(ch1)
    offspring2 = [None] * len(ch1)

    # Loop 100 times for unique cities, high probability of failure
    for i in range(100):
        point1 = point2 = i = 0

        # generating two random numbers
        while point1 == point2:
            point1 = randint(1, len(ch1) - 1)
            point2 = randint(1, len(ch1) - 1)

        # smaller value always in point1
        if point1 > point2:
            point1, point2 = point2, point1

        while i != point1:
            offspring1[i] = ch1[i]
            offspring2[i] = ch2[i]
            i += 1

        while i != point2:
            offspring1[i] = ch2[i]
            offspring2[i] = ch1[i]
            i += 1

        while i != len(ch1):
            offspring1[i] = ch1[i]
            offspring2[i] = ch2[i]
            i += 1

        # to check if all cities are unique or not
        if not len(offspring1) > len(set(offspring1)):
            if not len(offspring2) > len(set(offspring2)):
                offspring1 = best_chromosome_after_mutation(offspring1, mutation_rate, distances)
                offspring2 = best_chromosome_after_mutation(offspring2, mutation_rate, distances)
                pop = []
                pop.append(offspring1)
                pop.append(offspring2)
                return pop

    # else return the original chromosomes as it is
    ch1 = best_chromosome_after_mutation(ch1, mutation_rate, distances)
    ch2 = best_chromosome_after_mutation(ch2, mutation_rate, distances)
    pop = []
    pop.append(ch1)
    pop.append(ch2)
    return pop


def uniform_crossover(ch1, ch2, mutation_rate, distances):
    offspring1 = []
    offspring2 = []

    p = [ch1, ch2]
    p1 = [ch2, ch1]
    for i in range(len(p[0])):
        if p[1][i] in p[0]:                 # cities are not unique
            offspring1.append(p[0][i])
        else:                               # unique otherwise
            offspring1.append(p[random.randint(0, 1)][i])

    for i in range(len(p1[0])):
        if p1[1][i] in p1[0]:
            offspring2.append(p1[0][i])
        else:
            offspring2.append(p1[random.randint(0, 1)][i])

    offspring1 = best_chromosome_after_mutation(offspring1, mutation_rate, distances)
    offspring2 = best_chromosome_after_mutation(offspring2, mutation_rate, distances)
    pop = []
    pop.append(offspring1)
    pop.append(offspring2)
    return pop


def myown_crossover(ch1, ch2, mutation_rate, distances):
    # first i apply one point crossover , then i created a list of number of cities and
    # then i shuffled it. after then i compared it with the offsprings and removed the repeated values in them

    point = len(ch1) - 3
    el = list(range(1, len(ch1) + 1))
    offspring1 = ch1[:point] + ch2[point:]
    offspring2 = ch2[:point] + ch1[point:]
    shuffle(el)
    for i in el:
        if i not in offspring1:
            offspring1.append(i)

    shuffle(el)
    for i in el:
        if i not in offspring2:
            offspring2.append(i)

    first = list(set(offspring1))
    second = list(set(offspring2))

    first = best_chromosome_after_mutation(first, mutation_rate, distances)
    second = best_chromosome_after_mutation(second, mutation_rate, distances)

    pop = []
    pop.append(first)
    pop.append(second)

    return pop


def stochastic_selection(population, population_size, distances):
    fitnessDict = defaultdict(list)
    for i in population:
        fitnessDict[calculateFitness(distances, i)] = i

    sortedFitness = list(reversed(sorted(fitnessDict.items(), key=operator.itemgetter(0))))
    population_new = []
    pop = 0
    for i in range(population_size):
        population_new.append(sortedFitness[pop][1])
        pop += 1

    return population_new


def tournament_selection(population, population_size, distances, tournament_size):
    fitnessDict = defaultdict(list)
    for i in population:
        fitnessDict[calculateFitness(distances, i)] = i

    sortedFitness = list(fitnessDict.items())
    new_list = random.sample(sortedFitness, tournament_size)
    population_new = []
    pop = 0
    for i in range(population_size):
        population_new.append(new_list[pop][1])
        pop += 1

    return population_new


def genetic_algorithm(initial_population_size, crossover, selection, mutation_rate, iterations, tournament_size):
    cities = get_cities("names.txt")
    (starting_point, remaining_points) = get_starting_and_other_points(cities)
    distance_matrix = nxn_distance_matrix("distances.txt", len(cities))

    population = get_initial_population(initial_population_size, starting_point, remaining_points, distance_matrix)

    fitnessDict = defaultdict(list)
    for i in range(initial_population_size):
        fitnessDict[calculateFitness(distance_matrix, population[i])] = population[i]

    for i in range(0, iterations):
        clone_for_population = population[:]
        for i in range(0, len(clone_for_population), 2):
            if crossover == "uniform":
                population_intermediate = uniform_crossover(clone_for_population[i], clone_for_population[i + 1], mutation_rate, distance_matrix)
            elif crossover == "one_point":
                population_intermediate = one_point_crossover(clone_for_population[i], clone_for_population[i + 1], mutation_rate, distance_matrix)
            elif crossover == "two_point":
                population_intermediate = two_point_crossover(clone_for_population[i], clone_for_population[i + 1], mutation_rate, distance_matrix)
            else:
                population_intermediate = myown_crossover(clone_for_population[i], clone_for_population[i + 1], mutation_rate, distance_matrix)
            population.extend(population_intermediate)

        clone_for_population[:] = []
        if selection == "stochastic":
            population = stochastic_selection(population, initial_population_size, distance_matrix)
        else:
            population = tournament_selection(population, initial_population_size, distance_matrix, tournament_size)

        for i in range(initial_population_size):
            fitnessDict[calculateFitness(distance_matrix, population[i])] = population[i]

    return fitnessDict, cities


def get_lat_long(filename):
    file = open(filename)
    data = file.readlines()
    row_counter = 0
    lat = []
    long = []
    for i in data:
        i = i.strip("\n")
        i = i.split(" ")
        lat.append(i[0])
        long.append(i[1])
        row_counter += 1

    return (lat, long)


(fitnessDict, cities) = genetic_algorithm(2, "myown", "stochastic", 0.1, 100, 2)
sortedFitness = list(reversed(sorted(fitnessDict.items(), key=operator.itemgetter(0))))
print ("Mutation Rate: 0.1")
print ("CrossOver Method: My Own")
print("Fitness for the Best Path: " + str(sortedFitness[0][0]))
path = sortedFitness[0][1]
city_wise_path = []
for i in path:
    city_wise_path.append(cities[i - 1])
print("Cities in Best Path: " + str(city_wise_path))


