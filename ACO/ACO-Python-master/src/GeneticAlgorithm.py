import os
import sys
import random
from TSPData import TSPData
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


# TSP problem solver using genetic algorithms.
class GeneticAlgorithm:

    # Constructs a new 'genetic algorithm' object.
    # @param generations the amount of generations.
    # @param popSize the population size.
    def __init__(self, generations, pop_size):
        self.generations = generations
        self.pop_size = pop_size

    # Knuth-Yates shuffle, reordering an array randomly
    # @param chromosome array to shuffle.
    @staticmethod
    def shuffle(chromosome):
        n = len(chromosome)
        for i in range(n):
            r = i + int(random.uniform(0, 1) * (n - i))
            swap = chromosome[r]
            chromosome[r] = chromosome[i]
            chromosome[i] = swap
        return chromosome

    # computes fitness for the given route
    @staticmethod
    def compute_fitness(data, route):
        cost = (1 / data.get_start_distances()[route[0]]) + (1 / data.get_end_distances()[route[len(route) - 1]])
        distances = data.get_distances()

        # add internal route costs
        for i in range(len(route) - 1):
            cost += (1 / distances[route[i]][route[i + 1]])

        return cost

    # computes fitness ratios
    @staticmethod
    def compute_fitness_ratio(fitnesses):
        fitness_sum = np.sum(fitnesses)

        # compute ratio for each fitness
        for i in range(len(fitnesses)):
            if i > 0:
                fitnesses[i] = ((fitnesses[i] / fitness_sum) * 100) + fitnesses[i - 1]
            else:
                fitnesses[i] = (fitnesses[i] / fitness_sum) * 100

        return fitnesses

    # search for the correct ratio
    @staticmethod
    def binary_search(arr, n):
        # check if ratio is 100%
        if n == 1:
            return len(arr) - 1

        i = 0
        j = len(arr)

        # binary search
        while j - i > 1:
            mid = int((j + i) / 2)

            # check middle
            if arr[mid] > n:
                j = mid
            else:
                i = mid

        return j

    # generates new chromosome based on two parent chromosomes
    @staticmethod
    def generate_offspring(ch1, ch2, crossover_position):
        crossover = ch1[crossover_position:]
        offspring = []

        # add genes from second chromosome into the offspring
        for i in range(len(ch2)):
            if not (ch2[i] in crossover):
                offspring.append(ch2[i])

        return offspring + crossover

    # mutates the chromosome with some probability
    @staticmethod
    def mutate(offspring, probability):
        if random.random() <= probability:
            pos1 = random.randint(0, len(offspring) - 1)
            pos2 = random.randint(0, len(offspring) - 1)

            aux = offspring[pos1]
            offspring[pos1] = offspring[pos2]
            offspring[pos2] = aux

        return offspring

    # This method should solve the TSP.
    # @param pd the TSP data.
    # @return the optimized product sequence.
    def solve_tsp(self, data):
        chromosome = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17]
        population = [chromosome]
        mutation_probability = 0.01

        # compute first generation
        for i in range(self.pop_size - 1):
            population.append(self.shuffle(chromosome.copy()))

        # compute fitnesses and create new generations
        for i in range(self.generations - 1):
            fitnesses = []
            offsprings = []

            # compute fitnesses
            for j in range(self.pop_size):
                fitnesses.append(self.compute_fitness(data, population[j]))

            fitnesses = self.compute_fitness_ratio(fitnesses)

            # generate offsprings
            for j in range(self.pop_size):
                ch1 = population[self.binary_search(fitnesses, random.random() * 100)]
                ch2 = population[self.binary_search(fitnesses, random.random() * 100)]

                offspring = self.generate_offspring(ch1, ch2, random.randint(0, len(ch1) - 1))
                offspring = self.mutate(offspring, mutation_probability)

                offsprings.append(offspring)

            population = offsprings.copy()

        fitnesses = []

        # compute fitnesses of last generation
        for i in range(self.pop_size):
            fitnesses.append(self.compute_fitness(data, population[i]))

        print(fitnesses)
        print(np.max(fitnesses))

        return population[np.argmax(fitnesses)]


# After receiving TSPData information, solve_tsp is called, which runs the genetic algorithm
# solve_tsp makes use of various methods like shuffle, compute_fitness, compute_fitness_ratio,
# binary_search, generate_offspring and mutate
if __name__ == "__main__":
    # parameters
    population_size = 100
    gen = 100000
    persistFile = "./../tmp/productMatrixDist"
        
    # setup optimization
    tsp_data = TSPData.read_from_file(persistFile)
    # print(tsp_data.get_end_distances())
    ga = GeneticAlgorithm(gen, population_size)

    # run optimization and write to file
    solution = ga.solve_tsp(tsp_data)
    print(solution)
    tsp_data.write_action_file(solution, "./../data/TSP solution.txt")
