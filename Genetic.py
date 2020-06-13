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
XO_parameter = 1 # parameter that stores the numbers of chromosomes we'll perform XO over


class genetic:

    def __init__(self, input_data, number_of_machines):  # input_data is a list of 2n jobs
        self.number_of_genes = len(input_data)  # length is number of jobs
        self.number_of_machines = number_of_machines
        # Defining the population_sample size
        self.num_of_chromosomes = num_of_chromosomes
        self.input_data = input_data
        self.fitness_func = np.zeros(self.num_of_chromosomes)
        self.Mutation_percentage = 1  # just random - still need to work on it
        self.objective_function_value = np.zeros(self.num_of_chromosomes)  # initializing objective function value
        self.probabilities = np.zeros(self.num_of_chromosomes)  # initializing probabilities value for each chromosome
        # self.obj_func_value_per_chromosome = np.array()  # will store objective function value for each chromosome

        # the complete self.population_sample, initializing the matrix with '-1' in every entry
        self.population_sample = -1 * np.ones((self.num_of_chromosomes, self.number_of_genes), dtype=int)
        self.sum_obj_functions = 0  # initializing - check if it's the correct type

    def action(self):
        """
        Performing Genetic algo approach
        Returns:
            # TODO: Add documentation

        """
        print("Starting Genetic")
        self.create_population()
        print(self.population_sample)
        print(self.input_data)
        self.objective_function_value = self.objective_func_calc(self.population_sample)
        print("First objective function value: ", self.objective_function_value)
        # print("decoded:", self.decoding(self.population_sample[0]))
        print((self.calc_probabilities(self.population_sample)))
        print(self.sum_obj_functions)
        print(self.fitness_func)
        print(self.objective_function_value)

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

        # How many chromosomes to create
        for i in range(self.num_of_chromosomes):

            # each chromosome is in self.number_of_genes length
            chromosome_i = self.population_sample[i]
            # list of indexes we'll use to randomly chose jobs from
            index_list = list(range(self.number_of_genes))

            j = 0

            # At each iteration we deal with 2 jobs - so do half of the input's length iterations
            for j in range(int((self.number_of_genes / 2))):
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

    def objective_func_calc(self, sample):
        """
        calculates objective function value by iterating every chromosome, and for each chromosome use decoding method
        to find the process time of each machine.
        Then -  stores the maximal process time for each chromosome as the objective function value in the according position

        Args:
            sample: The current population_sample

        Returns: Objective function value - max process time of every chromosome

        """

        for i in range(self.num_of_chromosomes):
            self.objective_function_value[i] = np.max(self.decoding(sample[i]))

        return self.objective_function_value

    # TODO: not working - check why
    def fitness_func_calc(self):
        """
        calculates current fitness function for this generation
        * First version: sum of all objective functions - objective function value for a given chromosome

        Returns: None

        """
        for i in range(self.num_of_chromosomes):
            self.fitness_func[i] = self.sum_obj_functions - self.objective_function_value[i]

    # TODO: check if actually working, he
    def choose_parents_for_XO(self):
        """
        choosing parents (?) for XO
        Returns: list of indexes s.t every two adjacent indexes are the parent "chosen together"

        """
        # np array that contains the pairs of chromosomes for XO
        XO_partners = -1 * np.ones((1, XO_parameter), dtype=int)

        for i in range(XO_parameter): # TODO: check if it's actually working
            XO_partners[i] = random.choices(self.population_sample, weights=self.probabilities, k=2)

        return XO_partners

     # TODO: check if working, and if giving 2 chromosomes is neccessary
    # TODO: first keep working on that!
    def perform_XO(self, xo_position, index_1, index_2):
        """
        Actually doing the XO
        Returns:

        """
        # TODO makesure updates self.pop
        chrome1 = self.population_sample[index_1]
        chrome2 = self.population_sample[index_2]

        temp = deepcopy(chrome1[xo_position:])
        chrome1[xo_position:] = chrome2[xo_position:]
        chrome2[xo_position:] = temp

        # return chromosome_1, chromosome_2

    def choose_Mutation(self):
        """
        TODO - check what to do with that
        Returns:

        """
        return random.uniform(0.1, 5)  # TODO: merge it into mutation method, check how to use it

    def perform_Mutation(self, Mutation_info):  # TODD: what is mutation?
        """
        Actually do Mutation
        Args:
            Mutation_info:

        Returns:

        """
        pass

    # TODO: merge it into perform_XO
    def choose_position_for_XO(self):
        """
        randomly choosing XO position for 2 chromosomes
        Returns: number (int) represents the position in which we'll perform the XO

        """
        # returns random index in range(self.number_of_genes)
        return random.choice(self.number_of_genes)

    # TODO: fix representation
    def calc_probabilities(self, population):
        """
        calculating probabilities for current population_sample-
        i.e - chromosome index, the chromosome itself, x_i (check again), choosing probability
        Returns: Matrix representation of current population_sample's data

        """
        self.sum_obj_functions = np.sum(self.objective_function_value)
        for i in range(self.num_of_chromosomes):
            self.probabilities[i] = self.objective_function_value[i] / self.sum_obj_functions

            print(
                f'ID {i}: {population[i]} function value: {self.objective_function_value[i]} choosing prob:{round(self.probabilities[i], 6)}')
            # population_data[i] =[f'chromosome index: {i} chromosome: {population_sample[i]} function value: {self.objective_function_value[i]} choosing probability:{self.objective_function_value[i] / sum_obj_functions}']

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
