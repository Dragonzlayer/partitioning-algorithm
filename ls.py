from itertools import combinations

import numpy as np

DEBUG = False  # Switch to True for prints during the run
FULL_MODE = True  # If false caps run after 10M iterations

# from Timer import timer


class LocalSearch:

    def __init__(self, state):
        """
        Full Implementation of Local Search Algorithm as defined

        Example Usage:
            local_searcher = LocalSearch(state)
            local_searcher.search()

        Results can be found at:
            local_searcher.curr_state
            local_searcher.sum_processing_times_per_machine
            local_searcher.sum_squared_processing_times

        Args:
            state: list of dictionaries of machine's indexes with their according jobs (job_id : job_val)
        """

        self.curr_state = state  # list of machine's indexes with their according jobs
        self.sum_processing_times_per_machine = np.array([sum(element.values()) for element in self.curr_state])
        self.sum_squared_processing_times = sum(self.sum_processing_times_per_machine ** 2)  # helper function value
        self.max_sum_processing_times = max(self.sum_processing_times_per_machine)  # objective function value
        self.max_search_space = len(self.curr_state[0]) - 1
        self.number_of_machines = len(self.curr_state)
        self.counter = 0
        self.jobs_per_balance = [[1, 1], [1, 3], [1, 1], [2, 2], [1, 1]]
        self.cap = 1000000

    def search(self):
        """
        performing local search algorithm
        Moving even number of jobs (starting from 2) as long as objective/helper functions values improve
        After that trying to improve objective/helper functions values by switching jobs (first Trying to switch job-for job, when
        it exhaust these options - increasing to trying to switch 2 jobs - for 2 jobs, and back to 1 for 1, etc.. (shown below)
        """
        self._transfer_jobs(search_space=2)

        for jobs_from_source, jobs_from_target in self.jobs_per_balance:
            self._balance_jobs(key_comb_from_source=jobs_from_source, key_comb_from_target=jobs_from_target)

    def _balance_jobs(self, key_comb_from_source, key_comb_from_target):
        """
        Trying to improve objective/helper functions values by switching jobs from Machines.
        At each Iteration, it select a 'Source machine' - A machine in which we are trying to switch jobs from, and an
        available possible 'Target Machine' - A machine in which we will try to switch the job from the source machine.
        It calculates possible permutations of jobs (size of sub-group given as an input) from source (and Target)
        Machines and checking whether or not switching these jobs to permutation of jobs from target machine will
        improve objective/helper functions values. Stops when exhausts the possibilities to switch jobs from possible
        source and target machines (in the given size from the input)

        Args:
             key_comb_from_source: number of jobs we'll try to move from source_machine.
                                   i.e - the size of permutations of jobs from source_machine.
             key_comb_from_target: number of jobs we'll try to move from target_machine.
                                   i.e - the size of permutations of jobs from target_machine.

        Returns:
                None.

        """
        # available/possible source & destination machines
        all_machines = list(range(self.number_of_machines))
        # Exhaust switches possibilities as long as there are possible source and destination machines left
        while all_machines:

            # Takes available source_machine from the available machines list
            # and its corresponding jobs
            source_machine = all_machines.pop(0)
            available_jobs_to_move_from_source = self.curr_state[source_machine]

            # creating list of indexes permutations of len(key_comb_from_source)
            keys_combinations_from_source = list(
                combinations(available_jobs_to_move_from_source.keys(), key_comb_from_source))

            # with the given parameters, check every iteration if we can switch jobs
            while keys_combinations_from_source:

                jobs_to_move_from_source = []
                sum_jobs_to_move_from_source = 0
                key_comb = keys_combinations_from_source.pop()
                for key in key_comb:
                    value = available_jobs_to_move_from_source[key]
                    sum_jobs_to_move_from_source += value
                    jobs_to_move_from_source.append((key, value))

                # calculating indexes of available_machines ,
                # and then removes the index of source_machine from this list
                available_machines = list(range(self.number_of_machines))

                available_machines.pop(source_machine)
                """
                As long as available_machines is not empty, calculates possible target_machine,
                checks whether jobs switch improves the state and if so - 
                updates curr_state 

                if switching jobs doesn't improve state - try the next target_machine
                """
                while available_machines:
                    # bool that says if a permutation of jobs was switched
                    swapped = False
                    target_machine = available_machines[0]
                    available_jobs_to_move_from_target = self.curr_state[target_machine]

                    # creating list of indexes permutations of len(curr_search_space)
                    keys_combinations_from_target = list(
                        combinations(available_jobs_to_move_from_target.keys(), key_comb_from_target))

                    # with the given parameters, check every iteration if we can switch jobs
                    while keys_combinations_from_target:

                        jobs_to_move_from_target = []
                        sum_jobs_to_move_from_target = 0
                        key_comb = keys_combinations_from_target.pop()
                        for key in key_comb:
                            value = available_jobs_to_move_from_target[key]
                            sum_jobs_to_move_from_target += value
                            jobs_to_move_from_target.append((key, value))

                        sum_jobs_to_move = sum_jobs_to_move_from_target - sum_jobs_to_move_from_source

                        # for extremely large input, cap the number of iterations, as there is no further improvement.
                        if not FULL_MODE:
                            if self.counter > self.cap:
                              return

                        if self._can_swap(source_machine, target_machine, sum_jobs_to_move):
                            # Updates State
                            for key, job in jobs_to_move_from_source:
                                self.curr_state[target_machine][key] = job
                                del self.curr_state[source_machine][key]

                            for key, job in jobs_to_move_from_target:
                                self.curr_state[source_machine][key] = job
                                del self.curr_state[target_machine][key]

                            # Update next step
                            all_machines = list(range(self.number_of_machines))
                            source_machine = all_machines.pop(0)

                            available_jobs_to_move_from_source = self.curr_state[source_machine]
                            available_jobs_to_move_from_target = self.curr_state[target_machine]

                            # creating list of indexes permutations of jobs from source_machine and target_machine
                            keys_combinations_from_source = list(
                                combinations(available_jobs_to_move_from_source.keys(), key_comb_from_source))
                            keys_combinations_from_target = list(
                                combinations(available_jobs_to_move_from_target.keys(), key_comb_from_target))
                            swapped = True
                            break

                    available_machines.remove(target_machine)
                    if swapped:
                        break

    def _can_swap(self, source_machine, target_machine, sum_jobs_to_move):
        """
         utility function that calculates whether or not switching jobs
         improves objective/helper functions.

         Args:
             source_machine: the machine we're trying to switch jobs from
             target_machine: the machine we're trying to switch jobs to
             sum_jobs_to_move: sum_jobs_to_move_from_target - sum_jobs_to_move_from_source
             the difference that we'll need to add/subtract to source/target machines

         Returns:
             bool, true - if switching these jobs will improve objective/helper function, False - otherwise
         """
        # calculating objective function values given the jobs were switched
        self.counter += 1

        self.sum_processing_times_per_machine[source_machine] += sum_jobs_to_move
        self.sum_processing_times_per_machine[target_machine] -= sum_jobs_to_move

        max_temp_sum_pt = np.max(self.sum_processing_times_per_machine)

        # calculating helper function values given the jobs were transferred
        temp_squared_sum_pt = np.sum(self.sum_processing_times_per_machine ** 2)

        # If not means that swap does not improve objective function or helper function
        # So revert state to state before the swap try
        if not (
                max_temp_sum_pt < self.max_sum_processing_times or temp_squared_sum_pt < self.sum_squared_processing_times):
            self.sum_processing_times_per_machine[source_machine] -= sum_jobs_to_move
            self.sum_processing_times_per_machine[target_machine] += sum_jobs_to_move
            return False
        else:
            self.max_sum_processing_times = max_temp_sum_pt
            self.sum_squared_processing_times = temp_squared_sum_pt
            return True

    # @timer
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
        all_machines = list(range(self.number_of_machines))

        while all_machines:
            source_machine = all_machines.pop(0)
            available_jobs_to_move = self.curr_state[source_machine]

            self.max_search_space = len(available_jobs_to_move)

            if search_space < self.max_search_space:
                curr_search_space = search_space
            else:
                curr_search_space = self.max_search_space

            # creating list of indexes permutations of len(curr_search_space)
            keys_combinations = list(combinations(available_jobs_to_move.keys(), curr_search_space))

            # with the given parameters, check every iteration if we can transfer jobs
            while keys_combinations:

                jobs_to_move = []
                sum_jobs_to_move = 0
                key_comb = keys_combinations.pop()
                for key in key_comb:
                    value = available_jobs_to_move[key]
                    sum_jobs_to_move += value
                    jobs_to_move.append((key, value))

                # calculating indexes of available_machines , and then removes the index of source_machine from this list
                available_machines = list(range(self.number_of_machines))

                available_machines.pop(source_machine)
                """
                As long as available_machines is not empty, calculates possible target_machine,
                checks whether jobs transfer improves the state and if so - 
                updates curr_state and resets search_space to 2 (minimum)

                if transferring jobs doesn't improve state - try the next target_machine
                """
                while available_machines:

                    target_machine = available_machines[0]

                    # checking whether a transfer will improve objective/helper functions
                    if self._can_improve(source_machine, target_machine, sum_jobs_to_move):
                        # If so - Updates State
                        for key, job in jobs_to_move:
                            self.curr_state[target_machine][key] = job
                            del self.curr_state[source_machine][key]

                        # Update next step
                        all_machines = list(range(self.number_of_machines))
                        source_machine = all_machines.pop(0)

                        available_jobs_to_move = self.curr_state[source_machine]

                        self.max_search_space = len(available_jobs_to_move)

                        curr_search_space = 2

                        # creating list of indexes permutations of len(curr_search_space)
                        keys_combinations = list(combinations(available_jobs_to_move.keys(), curr_search_space))

                        is_changed = True

                        break
                    else:
                        available_machines.remove(target_machine)

        return is_changed

    def _can_improve(self, max_machine, destination_machine, sum_jobs_to_move):
        """
            deciding whether or not to transfer job according to objective and helper functions
        Args:
            max_machine: int, index of the machine with maximal jobs processing times
            sum_jobs_to_move: sum_jobs_to_move_from_target - sum_jobs_to_move_from_source
            i.e, the difference between the jobs from both machines - if transfer is need
            we'll add/subtract only this number and not re-calculate evreything
            destination_machine: index of machine to try to transfer jobs_to_move to

        Returns: bool, true if transferring the jobs improves objective *or* helper functions, false otherwise
        """

        # calculating objective function values given the jobs were transferred
        self.sum_processing_times_per_machine[max_machine] -= sum_jobs_to_move
        self.sum_processing_times_per_machine[destination_machine] += sum_jobs_to_move

        max_temp_sum_pt = np.max(self.sum_processing_times_per_machine)

        # calculating helper function values given the jobs were transferred
        temp_squared_sum_pt = np.sum(self.sum_processing_times_per_machine ** 2)

        # if objective/helper functions doesn't improve -  revert state to state before the transfer try
        if not (
                max_temp_sum_pt < self.max_sum_processing_times or temp_squared_sum_pt < self.sum_squared_processing_times):
            self.sum_processing_times_per_machine[max_machine] += sum_jobs_to_move
            self.sum_processing_times_per_machine[destination_machine] -= sum_jobs_to_move
            return False
        else:
            # update objective/helper functions values and return true
            self.max_sum_processing_times = max_temp_sum_pt
            self.sum_squared_processing_times = temp_squared_sum_pt
            return True
