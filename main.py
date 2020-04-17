"""
Local search psuedocode:(greedy approach)

curr_state = put all jobs to machine #1
search_space = min_search_space

while search_space < max_search_space:

    new_state = transfer_jobs(curr_state, search_space)   
                
    if new_state == curr_state:
        break 
    else:
        search_space = search_space.expand()                
        curr_state = new_state
    
    

def transfer_jobs(curr_state, search_space):

    max_machine = the machine with maximal processing time
    jobs_to_move = all jobs from max_machine by search_space

    while jobs_to_move not empty:
    
        select a jobs_to_move from jobs_to_move  # TODO select how?
        
        available_machines = all_machines minus max_machine
        
        while available_machines not empty:
        
            select destination_machine from available_machines #TODO select how?
            
            temp_new_state = curr_state plus jobs_to_move was moved to destination_machine
            
            if obj_func(new_state) < obj_func(curr_state): # TODO implement more effienect look only at changes - look at delta)
            
                curr_state = temp_new_state
                max_machine = the machine with maximal processing time
                jobs_to_moves = all jobs from max_machine
                search_space = search_space.reset()
                break         
                
            else:
                available_machines = available_machines minus destination_machine
        
        jobs_to_move = jobs_to_move minus jobs_to_move
        
    return curr_state
"""
import random
import numpy as np
from copy import deepcopy
from init_parameters import get_parameters
import sys
from itertools import combinations
DEBUG = False  # Switch to True for prints during the run


class LocalSearch:
    """

    """
    def __init__(self, state):

        self.curr_state = state  # list of machine's indexes with their according jobs
        self.sum_processing_times_per_machine = np.array([sum(element.values()) for element in self.curr_state])
        self.sum_squared_processing_times = sum(self.sum_processing_times_per_machine ** 2)  # helper function value
        self.max_sum_processing_times = max(self.sum_processing_times_per_machine)   # objective function value
        self.temp_sum_processing_times = []
        self.max_search_space = len(self.curr_state[0]) - 1

    def search(self):
        """
        performing local search algorithm
        partitions the input by calling self._transfer_jobs,
        such that each machine (in self.curr_state) has an even number of jobs

        """
        min_search_space = 2
        search_space = min_search_space

        # print(self.curr_state)
        while search_space < self.max_search_space:

            is_changed = self._transfer_jobs(search_space)
            if not is_changed:
                break
            else:
                search_space = search_space + 2
            # print("State with search space {}: {}".format(search_space, self.curr_state))

    def _transfer_jobs(self, search_space):
        """
            Transfer jobs such that at the end of the run the partition cannot be improved given search_space

            Does this by calculating parameters for transferring jobs and
            updating current state if transferring jobs is necessary
        Args:
            search_space:int, 2 or bigger, always even. number of jobs to move in current run

        Returns: bool, is_changed - if the current run changes curr_state (if not - means that the local search can't improve)

        """
        is_changed = False

        # calculating index of source_machine and available_jobs_to_move from source_machine
        all_machines = [index for index, value in enumerate(list(self.sum_processing_times_per_machine))]

        while all_machines:
            #TODO: check if to pop from first position or shuffle?
            # TODO: calculate for every even combination of jobs!
            source_machine = all_machines.pop(0)
            available_jobs_to_move = deepcopy(self.curr_state[source_machine])

            self.max_search_space = len(available_jobs_to_move)

            # moving available_jobs_to_move from current source_machine to jobs_to_move according to possible search_space

            # creating list of indexes permutations of len(curr_search_space)
            if search_space < self.max_search_space:
                curr_search_space = search_space
            else:
                curr_search_space = self.max_search_space

            keys_combinations = list(combinations(available_jobs_to_move.keys(), curr_search_space))

            # with the given parameters, check every iteration if we can transfer jobs
            while keys_combinations:

                jobs_to_move = []
                key_comb = keys_combinations.pop()
                for key in key_comb:
                    value = available_jobs_to_move[key]
                    jobs_to_move.append((key, value))

                # calculating indexes of available_machines , and then removes the index of source_machine from this list
                available_machines = [i for i in range(number_of_machines)]

                available_machines.pop(source_machine)
                # random.shuffle(available_machines) # todo return if needed
                """
                As long as available_machines is not empty, calculates possible target_machine,
                checks whether jobs transfer improves the state and if so - 
                updates curr_state and resets search_space to 2 (minimum)
                
                if transferring jobs doesn't improve state - try the next target_machine
                """
                while available_machines:

                    #TODO: why in the first position and not shuffle?
                    target_machine = available_machines[0]

                    # TODO: check all possibilities of jobs to transfer
                    if self._can_improve(source_machine, target_machine, jobs_to_move):
                        # Updates State
                        for key, job in jobs_to_move:
                            self.curr_state[target_machine][key] = job
                            del self.curr_state[source_machine][key]

                        # updates objective / helper function values
                        self.sum_processing_times_per_machine = self.temp_sum_processing_times
                        self.max_sum_processing_times = max(self.sum_processing_times_per_machine)
                        self.sum_squared_processing_times = sum(self.sum_processing_times_per_machine ** 2)

                        # Update next step
                        all_machines = [index for index, value in enumerate(list(self.sum_processing_times_per_machine))]
                        source_machine = all_machines.pop(0)

                        available_jobs_to_move = deepcopy(self.curr_state[source_machine])

                        self.max_search_space = len(available_jobs_to_move)

                        # moving available_jobs_to_move from current source_machine to jobs_to_move according to possible search_space

                        # creating list of indexes permutations of len(curr_search_space)
                        curr_search_space = 2

                        keys_combinations = list(combinations(available_jobs_to_move.keys(), curr_search_space))

                        is_changed = True

                        break
                    else:
                        available_machines.remove(target_machine)

                    # print("current state: ", curr_state)

        return is_changed

    def _can_improve(self, max_machine, destination_machine, jobs_to_move):
        """
            deciding whether or not to transfer job according to objective and helper functions
        Args:
            max_machine: int, index of the machine with maximal jobs processing times
            jobs_to_move: NumPy array, processing times of jobs from max_machine
            destination_machine: index of machine to try to transfer jobs_to_move to

        Returns: bool, true if transferring the jobs improves objective *or* helper functions, false otherwise
        """

        sum_jobs_to_move = 0
        for key, value in jobs_to_move:
            sum_jobs_to_move += value


        # calculating objective function values given the jobs were transferred
        self.temp_sum_processing_times = deepcopy(self.sum_processing_times_per_machine)
        self.temp_sum_processing_times[max_machine] -= sum_jobs_to_move
        self.temp_sum_processing_times[destination_machine] += sum_jobs_to_move
        max_temp_sum_pt = max(self.temp_sum_processing_times)

        # calculating helper function values given the jobs were transferred
        temp_squared_sum_pt = sum(self.temp_sum_processing_times ** 2)

        # if DEBUG:
            # print("sums: ", self.temp_sum_processing_times, "Squared Sums: ", self.temp_squared_sum_processing_times)
            # print("Max sum: ", max_temp_sum_pt, "Max Squared Sum: ", max_temp_squared_sum_pt)

        return max_temp_sum_pt < self.max_sum_processing_times or temp_squared_sum_pt < self.sum_squared_processing_times


if __name__ == '__main__':
    sys.stdout = open(r'C:\Users\user1\PycharmProjects\partitioning-algorithm\local-search_output\output.txt', mode='a')
    print("--------------- New Run --------------")
    # receive input from user and draw process times
    jobs_process_time, number_of_machines = get_parameters()

    # initializing: putting all jobs in the first machine
    initial_state = [{} for machine in range(number_of_machines)]
    initial_state[0] = jobs_process_time

    # initialize LocalSearch class with initial state
    local_searcher = LocalSearch(initial_state)
    print("Performing Local search...")
    # perform local search algorithm
    local_searcher.search()

    print("Final state:")
    for i,k in enumerate(local_searcher.curr_state):
        print("machine nr. ", i, ":", k)

    print("Number of jobs for each machine: ")
    print([len(item) for item in local_searcher.curr_state])
    print("Sum of process times for each machine: ", local_searcher.sum_processing_times_per_machine)
    print("Squared sum of process times for each machine: ", local_searcher.sum_squared_processing_times)
