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

num_of_chromosomes = 10

class genetic:

    def __init__(self, input_data, number_of_machines): # input_data is a list of 2n jobs
        self.number_of_genes = len(input_data)  # length is number of jobs
        self.number_of_machines = number_of_machines
        # Defining the population size
        self.num_of_chromosomes = num_of_chromosomes
        self.input_data = input_data
    def action(self):
        """
        Performing Genetic algo approach
        Returns:
            # TODO: Add documentation

        """
        print("Starting Genetic")
        population_sample = self.create_population(self.input_data)
        print(population_sample)

    # TODO: check if actually working, He
    def create_population(self, input_data):

        # initializing the matrix with '-1' in every entry
        population = -1*np.ones((self.num_of_chromosomes, self.number_of_genes), dtype=int)

        # How many chromosomes to create
        for i in range(self.num_of_chromosomes):

            # each chromosome is in self.number_of_genes length
            chromosome_i = population[i]
            # input_data_copy = deepcopy(input_data)
            # list of indexes we'll use to randomly chose jobs from
            index_list = list(range(len(input_data)))
            # Each time, generating 2 random jobs from the list and randomly assigns a machine for them
            j = 0
            # At each iteration we deal with 2 jobs - so do half of the input's length iterations
            for j in range(int((self.number_of_genes/2))):
                # choose 2 random jobs indexes - after choosing the first one, remove it from the
                #  list - so we don't choose it again and then choose the second job index - and remove this index from the list as well
                # TODO: check uniforn distribution

                random_index1 = random.choice(index_list)
                index_list.remove((random_index1))
                random_index2 = random.choice(index_list)
                index_list.remove((random_index2))

                # choosing a random machine for the 2 chosen jobs
                random_machine = random.choice(range(self.number_of_machines))

                # Assigning the random machine in the according indexes
                chromosome_i[random_index1] = random_machine
                chromosome_i[random_index2] = random_machine

        # return the complete population
        return population

    if __name__ == '__main__':

        print(create_population(input_data))
