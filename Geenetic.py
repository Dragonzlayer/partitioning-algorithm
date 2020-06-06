from Timer import timer
from copy import deepcopy
from math import ceil
import numpy as np
from heapq import nsmallest
import sys
import random

""""
Pseudocode version 1:

    Input: Randomized 2n jobs and m Machines
    
    Initialization: To start with: create 100 chromosomes 
        How to create chromosome: While jobs != empty:
                                    1: draw one job from the input
                                    2: draw another job (a partner for the first job) - both chosen randomly - uniform distribution
                                    3 draw a machine for this 2 jobs (treat them as a 'unit')
                                    4: back to stage 1
    
        

"""


class Genetic:

    def __init__(self, input, number_of_machines): # input is a list of 2n jobs
        self.number_of_genes = len(input) # length is number of jobs
        self.population_size - num_of_chromosomes
        self.number_of_machines = number_of_machines
        # Defining the population size
        self.num_of_chromosomes = 100

    # TODO: check if actually working, He
    def create_population(self, input):

        population = np.array([])

        # How many chromosomes to create
        for i in self.num_of_chromosomes:

            # each chromosome is in self.number_of_genes length
            chromosome_i = np.zeroes(self.number_of_genes)
            # Each time, generating 2 random jobs from the list and randomly assigns a machine for them
            for i in self.number_of_genes:
                # choose 2 random jobs - after choosing the first one, remove it from the
                #  list - so we don't choose it again and then choose the second job and return the job to the list
                # TODO: check uniforn distribution
                rand_job1 = random.choice(input)
                index_rand_job1 = input.index(rand_job1)
                input.remove(rand_job1)
                rand_job2 = random.choice(input)
                index_rand_job2 = input.index(rand_job2)
                input.append(rand_job1)

                # choosing a random machine for the 2 chosen jobs
                random_machine = random.choice(self.number_of_machines)

                # Assigning the random machine in the according indexes
                chromosome_i[index_rand_job1] = random_machine
                chromosome_i[index_rand_job2] = random_machine

            # Appending every chromosome to the population
            population.append(chromosome_i)

        population = np.asanyarray(population)

        # return the complete population
        return population