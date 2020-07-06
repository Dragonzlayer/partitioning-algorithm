import random
import sys
from copy import deepcopy
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax
# TODO: seed!!!! (for second part of the first run)
np.random.seed(4)
random.seed(5)
np.set_printoptions(threshold=sys.maxsize)
num_of_chromosomes = 100
XO_parameter = 49  # parameter that stores the numbers of chromosomes we'll perform XO over
num_generations = 200003 # just initialization - change when necessary
print_rate = 50000  # print to file and plot each this many generations
methods = ['6', '7', '8', '9', '10']   # ['1', '2', '3', '4', '5']  # ['fitness_sqrt']  #"fitness", "fitness_squared", "fitness_softmax"]  # , 'objective', 'objective_softmax', 'objective_squared']


class genetic:
    """
    Full Implementation of Genetic Algorithm as defined

    Example Usage: # TODO: example usage & fix & clean main
        local_searcher = LocalSearch(state)
        local_searcher.search()

    Results can be found at: # TODO: fix result can be found at
        local_searcher.curr_state
        local_searcher.sum_processing_times_per_machine
        local_searcher.sum_squared_processing_times

    Args:
        input_data: list of 2n jobs
        number_of_machines: int, representing number of machines as given in the input

    """

    def __init__(self, input_data, number_of_machines):
        self.number_of_genes = len(input_data)
        self.number_of_machines = number_of_machines
        # Defining the population_sample size
        self.num_of_chromosomes = num_of_chromosomes
        self.input_data = input_data
        self.fitness_func = np.zeros(self.num_of_chromosomes) # initializing fitness function value
        self.Mutation_percentage = 1  # just random - still need to work on it TODO: check what to do with it
        self.objective_function_value = np.zeros(self.num_of_chromosomes)  # initializing objective function value
        self.probabilities = np.zeros(self.num_of_chromosomes)  # initializing probabilities value for each chromosome
        # self.obj_func_value_per_chromosome = np.array()  # will store objective function value for each chromosome TODO: check what to do with it

        # the complete self.population_sample, initializing the matrix with '-1' in every entry
        self.population_sample = -1 * np.ones((self.num_of_chromosomes, self.number_of_genes), dtype=int)
        self.sum_fitness_functions = 0  # initializing
        self.mutated_indexes = []  # stores indexes of chromosomes who've been mutated
        self.sum_input_data = np.sum(input_data)
        self.XO_partners = -1 * np.ones((XO_parameter, 2), dtype=int)
        self.best_population = deepcopy(self.population_sample) # stores best solution so far
        self.best_objective_function_mean = float('inf')
        self.sum_to_machines_avg = np.sum(self.input_data)/self.number_of_machines

        the_time = datetime.now().strftime("%d%m%Y_%H%M%S")
        self.file_path = f"Genetic_output_final/genetic_output_{number_of_machines}machines_{self.number_of_genes}genes_generations{num_generations}_{the_time}.txt"
        self.img_path = f"Genetic_output_final/Obj_vs_Gen_{number_of_machines}machines_{self.number_of_genes}genes_generations{num_generations}_{the_time}"

    def action(self):
        """
        Performing Genetic algo approach
        For each fitness method, it creates population, then send it to a different function to perform
        the needed calculations.
        It also calculates the mean objective function value, and writes the results to a file.

        Returns: The last population after num_generations iterations, as well as data about the population.
        """

        full_res = []

        for method in methods:

            self.create_population()
            self.write_to_file("Initialize", method)

            res = []
            for generation in range(num_generations + 1):

                self.action_iteration(method)

                mean_obj_func = self.calc_mean_and_update_best()
                res.append(mean_obj_func)
                print(f"gen {generation} complete")

                if generation % print_rate < 2:

                    self.write_to_file(generation, method)

                    self.plot_results(res, method, is_full=False)

            print(self.best_population)
            print(self.best_objective_function_mean)

            full_res.append(res)
            self.write_to_file(num_generations, method)
        self.plot_results(full_res, methods, is_full=True)
        print(f'saving results to {self.file_path}')

    def calc_mean_and_update_best(self):
        """
        calculates mean objective function value, and if it's better than the current best objective function value -
        updates it as the new best objective function value.

        Returns: mean objective function value

        """
        mean_obj_func = self.objective_function_value.mean()

        if mean_obj_func < self.best_objective_function_mean:
            self.best_objective_function_mean = mean_obj_func
            self.best_population = deepcopy(self.population_sample)

        return mean_obj_func

    def action_iteration(self, method):
        """
        Performing the necessary calculations for the Algorithm.
        It first clears the list of the mutated indexes, then calculating objective function values, fitness function values,
        and probabilities for the current population.
        From this data it performs elitism, Mutation and choosing (and performing) XO.
        Args:
            method: The current fitness function method

        Returns:

        """
        self.mutated_indexes.clear()

        self.objective_func_calc()

        self.fitness_func_calc()

        self.calc_probabilities(method)

        self.elitism()

        self.perform_Mutation()

        self.choose_parents_for_XO()

        for indx in range(XO_parameter):
            # TODO: how to choose them using fitnees function?
            chrome1 = self.XO_partners[indx, 0]
            chrome2 = self.XO_partners[indx, 1]
            if chrome1 != chrome2:
                self.perform_XO(self.XO_partners[indx, 0], self.XO_partners[indx, 1])

    def write_to_file(self, generation, method):
        """
        Writing the necessary data to the file

        Args:
            generation: The current generation we're in
            method: The current fitness function method

        Returns: None

        """
        with open(self.file_path, "a") as f:
            print(f" ------------------------------- End of generation={generation}, method={method} -----------------------------", file=f)
            if generation =="Initialize":
                row_labels = ['# Genes', '# Machines', '# Chromosomes', '#XO', 'Sum Total jobs', '# Generations']
                table_vals = [[self.number_of_genes], [self.number_of_machines], [self.num_of_chromosomes],
                              [XO_parameter],
                              [sum(self.input_data)], [num_generations]]
                for label, val in zip(row_labels,table_vals):
                    print(f"{label}:{val}", file=f)

                print(f"Jobs:{self.input_data}", file=f)

            print("Population:", file=f)
            for i in range(self.num_of_chromosomes):
                print(f"Chrome {i}:\n", self.population_sample[i], file=f)

            print(f"Fitness:\n{self.fitness_func}", file=f)
            print(f"Probabilities:\n{self.probabilities}", file=f)
            try:
                print(f"Elitism:\n{self.mutated_indexes[0]}", file=f)
                print(f"Mutated:\n{self.mutated_indexes[1:]}", file=f)
            except:
                print(f"No Mutation", file=f)

            print(f"XO Partners:\n{self.XO_partners}", file=f)
            print(f"Objective Function Values:\n{self.objective_function_value}", file=f)
            print(f"Min object function value:\n{min(self.objective_function_value)}", file=f)
            print(f"Avg object function value:\n{self.objective_function_value.mean()}", file=f)


    def plot_results(self, res, methods, is_full):
        """
        Plots the results visually

        Args:
            res: list of methos # TODO: check about this args
            methods: The current fitness function method
            is_full:

        Returns:

        """
        plt.figure(figsize=(16, 12))
        if is_full:
            for inx, method in enumerate(methods):
                plt.plot(res[inx], label=method)
        else:
            plt.plot(res, label=methods)
        plt.xlabel("Generations")
        plt.ylabel("Mean objective function value")
        plt.title("Mean objective function value per Generation")
        plt.legend(loc='best')
        col_labels = ['Value']
        row_labels = ['# Genes', '# Machines', '# Chromosomes', '#XO', 'Sum Total jobs', 'Mean fitness']
        table_vals = [[self.number_of_genes], [self.number_of_machines], [self.num_of_chromosomes],
                      [XO_parameter],
                      [sum(self.input_data)], [self.fitness_func.sum() / self.num_of_chromosomes]]

        # the rectangle is where I want to place the table
        plt.table(cellText=table_vals,
                  colWidths=[0.1] * 3,
                  rowLabels=row_labels,
                  colLabels=col_labels,
                  loc='best')
        plt.savefig(self.img_path)
        plt.show()

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

    def objective_func_calc(self):
        """
        calculates objective function value by iterating every chromosome, and for each chromosome use decoding method
        to find the process time of each machine.
        Then - stores the maximal process time for each chromosome as the objective function value in the according position

        Args:
            sample: The current population_sample

        Updates Objective function value - max process time of every chromosome

        """

        for i in range(self.num_of_chromosomes):
            self.objective_function_value[i] = np.max(self.decoding(self.population_sample[i]))

    # TODO:  - check if OK or what/how to fix
    def fitness_func_calc(self):
        """
        calculates current fitness function for this generation
        * First version: sum of all objective functions - objective function value for a given chromosome



        Returns: None

        """
        # First fitness function calculation
        #self.fitness_func = self.sum_input_data - self.objective_function_value

        # Second fitness function calculation
        # I've added 1 so that no chromosome will receive the value '0'
        mean_obj_func = np.sum(self.input_data)/ self.number_of_machines
        self.fitness_func = 1/(self.objective_function_value - mean_obj_func + 1)


    def choose_parents_for_XO(self):
        """
        choosing parents (?) for XO
        Returns: list of indexes s.t every two adjacent indexes are the parent "chosen together"

        """
        # np array that contains the pairs of chromosomes for XO
        available_to_xo_indexes = list(range(num_of_chromosomes))

        for i in range(XO_parameter):  # TODO: check if it's actually working
            x = np.array(
                random.choices(available_to_xo_indexes, weights=self.probabilities[available_to_xo_indexes], k=2))
            self.XO_partners[i] = x

    def perform_XO(self, index_1, index_2):
        """
        Performin XO:
            1. Randomly chose XO_position
            2. Randomly chose 'side' s.t we'll switch the chromosome's tails to the right or left of the XO_position
            3. According to the XO_position and side, we'll perform the XO for the given chromosomes
            4. Send the 2 chromosomes for 'correction' - the XO might cause a case in which the partition
               won't be even.
        Args: 2 indexes of chromosomes to perform XO over them
        Returns: None

        """
        xo_position = random.choice(list(range(self.number_of_genes)))
        xo_side = random.choice(['right', 'left'])

        chrome1 = self.population_sample[index_1]
        chrome2 = self.population_sample[index_2]

        if xo_side =='right':
            temp = deepcopy(chrome1[xo_position:])
            chrome1[xo_position:] = chrome2[xo_position:]
            chrome2[xo_position:] = temp

        else:
            temp = deepcopy(chrome1[:xo_position])
            chrome1[:xo_position] = chrome2[:xo_position]
            chrome2[:xo_position] = temp

        # sending the chromosomes to self.correction() in case the XO caused for invalid partition
        self.correction(chrome1)
        self.correction(chrome2)


    def perform_Mutation(self):
        """
        Perform Mutation:
            uniformly choose probability for mutation between 0.1% and 5%.
            According to the chosen mutation probability - chose a chromosome to mutate:
                Then randomly draw 2 indexes and swap their machine's assignments

        Mark its position in the mutated chromosomes index list

        Args: Self
        Returns: None

        """
        mutation_prob = random.uniform(0.001, 0.05)
        available_to_mut_indexes = list(range(self.num_of_chromosomes))

        for i in range(self.num_of_chromosomes):
            mut_index = np.array(random.choices(available_to_mut_indexes, weights=self.probabilities[available_to_mut_indexes], k=1))[0]

            # draw random between 0 and 1
            x = random.random()
            if x < mutation_prob:
                position1 = random.randint(0, self.number_of_genes - 1)
                position2 = random.randint(0, self.number_of_genes - 1)
                temp = self.population_sample[mut_index][position2]
                self.population_sample[mut_index][position2] = self.population_sample[mut_index][position1]
                self.population_sample[mut_index][position1] = temp
                self.mutated_indexes.append(mut_index)

    def calc_probabilities(self, method):
        """
        calculating probabilities for current population_sample according to given fitness function method-
        i.e - chromosome index, the chromosome itself, x_i , choosing probability
        Returns: Matrix representation of current population_sample's data

        """

        if method == 'fitness':
            self.probabilities = self.fitness_func / np.sum(self.fitness_func)

        elif method == 'objective':
            self.probabilities = self.objective_function_value / np.sum(self.objective_function_value)

        elif method == 'objective_softmax':
            self.probabilities = softmax(self.objective_function_value)

        elif method == 'objective_squared':
            self.probabilities = self.objective_function_value ** 2 / np.sum(self.objective_function_value ** 2)

        elif method == 'fitness_squared':
            self.probabilities = self.fitness_func ** 2 / np.sum(self.fitness_func ** 2)

        elif method == 'fitness_softmax':
            self.probabilities = softmax(self.fitness_func)

        elif method == 'fitness_sqrt':
            self.probabilities = np.sqrt(self.fitness_func)

        elif method == '1':
            self.probabilities = 1/self.objective_function_value

        elif method == '2':
            self.probabilities = 1/self.objective_function_value**2

        elif method == '3':
            self.probabilities = 1 / np.sqrt(self.objective_function_value)

        elif method == '4':
            self.probabilities = 1 / self.objective_function_value ** 3

        elif method == '5':
            self.probabilities = 1 / (2*self.objective_function_value - self.sum_to_machines_avg)

        elif method == '6':
            self.probabilities = 1 / (self.objective_function_value - self.sum_to_machines_avg + 1)

        elif method == '7':
            self.probabilities = 1 / (3*self.objective_function_value - 2*self.sum_to_machines_avg)

        elif method == '8':
            self.probabilities = 1 / (self.objective_function_value - self.sum_to_machines_avg + 1)**2

        elif method == '9':
            self.probabilities = 1 / (self.objective_function_value - self.sum_to_machines_avg + 1) ** 3

        elif method == '10':
            self.probabilities = 1 / np.sqrt(self.objective_function_value - self.sum_to_machines_avg + 1)

        # Normalized probabilities
        self.probabilities /= sum(self.probabilities)

    def elitism(self):
        """
        Choose the chromosome with the highest fitness function value
        and transfer it as is to the next generation.
        * Marks its position in the mutated chromosomes index list

        Returns: None

        """
        index_for_elitism = np.argmax(self.fitness_func)
        self.mutated_indexes.append(index_for_elitism)

    def decoding(self, chromosome):
        """
        Decoding each chromosome: creates an array in size len(self.number_of_machines)
        s.t each position in the array stores the sum process time for each machine in the chromosome
        Args:
            chromosome: the array we need to decode

        Returns: Array that stores the sum process time of each machine in the chromosome

        """

        return np.bincount(chromosome, weights=self.input_data)

    def correction(self, chromosome):
        """
        Correct invalid chromosomes from XO:
        Doing it by counting how many machines has an odd number of jobs, pair those machines together, and
        for each pair moving jobs from the machine with the higher sum of jobs to the second machine
        s.t we'll get an even partition.

        Args:
            chromosome: Chromosome needed a correction

        Returns: None

        """
        decoded_chromosome = self.decoding(chromosome)

        # array that stores 0 if the machine with the current index has even number of jobs - and 1 if it's odd
        # odd_even = np.zeros(self.number_of_machines)

        # uniques stores uniques machine values
        # frequencies stores how many jobs assigned to each machine in the corresponding position in the array
        uniques, frequencies = np.unique(chromosome, return_counts=True)

        # Counting how many machines has odd number of jobs
        odd_machines = []
        for machine, count in zip(uniques, frequencies):
            if count % 2 == 1:
                odd_machines.append(machine)

        # pairs machines with odd number of jobs, so we can move jobs from one machine to the other
        # s.t after the moving each machine will have an even number of jobs
        pairs = []
        for i in range(0, len(odd_machines), 2):
            pairs.append((odd_machines[i], odd_machines[i + 1]))

        # For each pair of machines, move jobs from the machines with higher sum of jobs
        # so that afterwards each machine will have an even number of jobs, and make the partition a bit more balanced
        for i, j in pairs:

            if decoded_chromosome[j] > decoded_chromosome[i]:

                masked = np.ma.MaskedArray(self.input_data, chromosome != j)
                index_min_job = np.ma.argmin(masked)
                chromosome[index_min_job] = i

            else:
                masked = np.ma.MaskedArray(self.input_data, chromosome != i)
                index_min_job = np.ma.argmin(masked)
                chromosome[index_min_job] = j


