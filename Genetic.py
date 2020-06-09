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

num_of_chromosomes = 20

class genetic:

    def __init__(self, input_data, number_of_machines): # input_data is a list of 2n jobs
        self.number_of_genes = len(input_data)  # length is number of jobs
        self.number_of_machines = number_of_machines
        # Defining the population size
        self.num_of_chromosomes = num_of_chromosomes
        self.input_data = input_data
        self.fitness_func = 1 # just a random number - still need to work on it
        self.Mutation_percentage = 1 # just random - still need to work on it
        self.population_sample = np.array(([]))
        self.objective_function_value = np.zeros(self.num_of_chromosomes) # initializing objective function value
        # self.obj_func_value_per_chromosome = np.array()  # will store objective function value for each chromosome

    def action(self):
        """
        Performing Genetic algo approach
        Returns:
            # TODO: Add documentation

        """
        print("Starting Genetic")
        self.population_sample = self.create_population()
        print(self.population_sample)
        print(self.input_data)
        self.objective_function_value = self.objective_func_calc(self.population_sample)
        print("First objective function value: ", self.objective_function_value)
        #print("decoded:", self.decoding(self.population_sample[0]))
        print((self.calc_probabilities(self.population_sample)))

    def create_population(self):
        """
        The Method creates self.num_of_chromosomes chromosomes, each one in length(self.number_of_genes).
        It creates a list of indexes in length(self.number_of_genes) and for each chromosome it randomly chooses first
        index, removes it from the list, and then randomly chooses a second index (**).
        After that it randomly chooses a Machine from the given list s.t these two jobs indexes will be assigned to
        and adds the machine number according to the randomized indexes in the new chromosome
        That results in a Matrix shaped arrays that represents the new chromosomes and their assigned machines

        (**): we remove the indexes from the list so that we can't choose the same job twice - a job
              can be assigned to one Machine only.

        Returns: A Matrix in the size self.num_of_chromosomes X self.number_of_genes
                 s.t every row represents a chromosome in length(self.number_of_genes)

        """

        # initializing the matrix with '-1' in every entry
        population = -1*np.ones((self.num_of_chromosomes, self.number_of_genes), dtype=int)

        # How many chromosomes to create
        for i in range(self.num_of_chromosomes):

            # each chromosome is in self.number_of_genes length
            chromosome_i = population[i]
            # list of indexes we'll use to randomly chose jobs from
            index_list = list(range(self.number_of_genes))

            j = 0

            # At each iteration we deal with 2 jobs - so do half of the input's length iterations
            for j in range(int((self.number_of_genes/2))):
                # Each time, generate 2 random jobs from the list and randomly assign a machine for them

                # TODO: check uniforn distribution

                random_index1 = random.choice(index_list)
                index_list.remove(random_index1)
                random_index2 = random.choice(index_list)
                index_list.remove(random_index2)

                # choosing a random machine for the 2 chosen jobs
                random_machine = random.choice(range(self.number_of_machines))

                # Assigning the random machine in the according indexes
                chromosome_i[random_index1] = random_machine
                chromosome_i[random_index2] = random_machine

        # return the complete population
        return population

    def objective_func_calc(self, sample):
        """
        calculates objective function value by iterating every chromosome, and for each chromosome use decoding method
        to find the process time of each machine.
        Then -  stores the maximal process time for each chromosome as the objective function value in the according position

        Args:
            sample: The current population

        Returns: Objective function value - max process time of every chromosome

        """

        for i in range(self.num_of_chromosomes):
            self.objective_function_value[i] = np.max(self.decoding(sample[i]))

        return self.objective_function_value

    # TODO: calculate fitness function
    def fitness_func_calc(self):
        """
        calculates current fitness function for this generation
        Returns: None

        """
        pass

    def choose_parents_for_XO(self):
        """
        choosing parents (?) for XO
        Returns:

        """
        pass

    def perform_XO(self,  XO_position):
        """
        Actually doing the XO
        Returns:

        """
        pass

    def choose_Mutation(self):
        """
        TODO - check what to do with that
        Returns:

        """
        pass

    def perform_Mutation(self, Mutation_info):
        """
        Actually do Mutation
        Args:
            Mutation_info:

        Returns:

        """
        pass

    def choose_position_for_XO(self, low, high):
        """
        randomly choosing XO position for 2 chromosomes between low and high bounds
        Returns:

        """
        pass

    # TODO: fix representation
    def calc_probabilities(self, population):
        """
        calculating probabilities for current population-
        i.e - chromosome index, the chromosome itself, x_i (check again), choosing probability
        Returns: Matrix representation of current population's data

        """
        population_data =  np.zeros((self.num_of_chromosomes, 4), dtype=float)
        sum_obj_functions = np.sum(self.objective_function_value)
        for i in range(self.num_of_chromosomes):
            population_data[i] =['chromosome index: {0} chromosome: {1} function value: {2} choosing probability:{3}'.format(i, population[i],
            self.objective_function_value[i] , self.objective_function_value[i] / sum_obj_functions)]

        return population_data


    def elitism(self):
        """
        something should happen, right?
        Returns:

        """
        pass

    def decoding(self, chromosome):
        """
        Decoding each chromosome: creates an array in size len(self.number_of_machines)
        s.t each position in the array stores the sum process time for each machine in the chromosome
        Args:
            chromosome: the array we need to decode

        Returns: Array that stores the sum process time of each machine in the chromosome

        """
        sum_each_machine = np.zeros(self.number_of_machines)

        for i in range(len(chromosome)):
            sum_each_machine[chromosome[i]] += self.input_data[i]

        return sum_each_machine
    # TODO: do
    def correction(self, chromosome):
        """
        to correct invalid chromosomes from XO/mutation
        Args:
            chromosome:

        Returns:

        """


# TODO: when there's more than 2 machines - check what to do when some chromosomes don't assign jobs to a certain machine
# TODO decoding method to visually see the partition
