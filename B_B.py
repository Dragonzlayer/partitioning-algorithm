from Timer import timer
from copy import deepcopy
from math import ceil
import numpy as np
from heapq import nsmallest
import sys
from random import randrange

from init_parameters import get_parameters

nodes = 1


# number_of_jobs = 10

class NoValidState(Exception):
    """Raised when the input value is too small"""
    pass


class B_B:
    def __init__(self, input_state):

        """
        Full Implementation of Branch & Bound Algorithm as defined

        Example Usage:
            local_searcher = LocalSearch(state)
            local_searcher.search()

        Results can be found at:
            local_searcher.curr_state
            local_searcher.sum_processing_times_per_machine
            local_searcher.sum_squared_processing_times

        Args:
            input_state: list of dictionaries of machine's indexes, the last dictionary (indexed '-1') holds all the
            input's jobs in the following format: (job_id : job_val)
        """

        self.number_of_machines = len(input_state) - 1  # the last dictionary only holds the jobs from the input
        self.basic_lower_bound = ceil(sum(input_state['-1'].values()) / self.number_of_machines)
        self.max_job = max(input_state['-1'].values())
        self.input_state = input_state
        self.leading_partition = {}
        self.current_obj_func_value = 0
        self.all_jobs_sum = sum(input_state['-1'].values())

    def _update_state(self, state, i):
        """
        The function updates the state by sending the according data to self.try_pop_and_move(state, source_machine=-1, destination_machine=i)
        function
        Args:
            state: copy of the current input_state (i.e - the machines with their according jobs,
                         and the remaining jobs after the partition from the previous level)
            i: son number i. of the node we're currently working on (each son represents a machine in which we'll move
               a current job to)

        Returns: None

        """
        self.try_pop_and_move(state, source_machine=-1, destination_machine=i)

    def _undo_update(self, state, i):
        self.try_pop_and_move(state, source_machine=i, destination_machine=-1)


    @timer
    def DFS_wrapper(self):

        """
        Performing prologue tasks for B&B:
            Calculating lower and upper bound for the root.
            Printing the initial bounds
            Calculating and printing the first partition

        Then send it to self.DFS(self.input_state, 1) method to perform the algorithm for the rest of the tree,
        with level = 1 (As we've done level = 0 ,i.e - the root)

        Returns: None

        """
        # lower bound for the root is a bit different - needs to include also max job
        # the first upper bound is the trivial one - sum of all jobs
        input_state_copy = deepcopy(self.input_state)
        # Using _bounds_calc - creates the first leading partition and assigns the first objective function value
        try:
            temp_lower, temp_upper, self.leading_partition = self._bounds_calc(input_state_copy)
        except NoValidState:
            print("Illegal input. No Valid State")
            raise NoValidState

        self.basic_lower_bound = max(temp_lower, self.max_job)

        upper = ceil(sum(self.input_state['-1'].values()))
        self.current_obj_func_value = upper
        print(f"initial bounds [Lower, Upper]: [{int(self.basic_lower_bound)}, {upper}]")

        # First layer is symmetric - run on first machine and cont from there
        self._update_state(self.input_state, 0)
        print("First partition:")
        for key, machine in self.leading_partition.items():
            if key != '-1':
                print("Machine", key, " jobs:", list(machine.values()))

        self.DFS(self.input_state, 1)

    def DFS(self, input_state, level):
        """
        Performing B&B algo using DFS order:
            it copies the current input_state and for each son of the current node (i.e - possible machine to move job to)
            moving the current job and update necessary values, if it cannot create a new partition
            (i.e - it will create an invalid partition)- raises an exception and will move on and calculate another possibilities.
            if at any point we find a better partition and bounds - we update the current leading partition and its values.
            if at any point lower bound == upper bound, we truncate this node and don't create its sub-tree.
             we continue calculating in DFS order.

        Args:
            input_state: list of dictionaries representing the current state, i.e - the machines with their according jobs,
                         and the remaining jobs after the partition from the previous level.
            level: the current tree level we're working on

        Returns: None

        """
        # update current level
        level += 1

        # if there are no jobs left - return and stop the program, as we've exhausted the possible partitions
        if not input_state['-1']:
            return

        # copy the input_state so we can calculate multiple possible partitions without changing the data
        state = deepcopy(input_state)

        for i in range(self.number_of_machines):
            # print(f"Checking out sibling {i=}")
            self._update_state(state, i)
            try:
                lower, upper, current_partition = self._bounds_calc(state)
            except NoValidState:
                # print("No valid State ")
                self._undo_update(state, i)
                continue

            # print(f"{level=}, {state=}")
            # print(f"{lower=}, {upper=}")
            global nodes
            nodes += 1
            # if nodes % 10000 == 0:
            # print(nodes)

            # truncate the node and don't create a subtree for it.
            # keep searching in his adjacent nodes (not in his subtree)
            if lower >= self.current_obj_func_value:
                # print(f"entered if clause with {i=}: {lower} >= {self.current_obj_func_value}")
                self._undo_update(state, i)
                continue

            # we've found a better solution - so:
            # Replace the leading_partition to this partition and change current_obj_func_value
            # to the upper bound of this partition
            elif upper < self.current_obj_func_value:

                # print(f"{upper} < {self.current_obj_func_value}. {self.leading_partition=}")
                self.leading_partition = current_partition
                self.current_obj_func_value = upper
                print("--------------------- new leading partition ---------------------")
                sum_each_machine = self.machines_sum(self.leading_partition)
                print(f"{sum_each_machine=}")
                x = [len(item) for item in self.leading_partition.values()]
                print("Number of jobs: ", x[:-1])
                for key, machine in self.leading_partition.items():
                    if key != '-1':
                        print("Machine", key, " jobs:", list(machine.values()))

                # print(f"New {self.leading_partition=}")

            # truncate the node and don't create a subtree for it.
            # keep searching in his adjacent nodes (not in his subtree)
            if lower == upper:
                self._undo_update(state, i)
                continue

            # print(f"{self.leading_partition=}")
            self.DFS(state, level)

            # when going all the way down in a sub-tree's node, search its adjacent nodes with the appropriate data
            self._undo_update(state, i)


    def _bounds_calc(self, input_state):
        """
        Calculating lower and upper bounds for the current input_state, as well as current partition using LPT algorithm
        * calculates special lower bound case when there are 3 machines, as seen in Lecture's slides

        Args:
            input_state: list of dictionaries representing the current state - i.e the machines with their according jobs,
                         and the remaining jobs after the partition from the previous level.

        Returns: lower and upper bounds, partition created by LPT algorithm
        """
        # calculates sum of jobs for each machine
        sums = self.machines_sum(input_state)

        # special lower bound case when there are 3 machines
        if self.number_of_machines == 3:
            x = input_state['-1'].values()
            if len(x) > 1:
                min_job1, min_job2 = nsmallest(2, x)
            else:
                min_job1 = 0
                min_job2 = 0
            Arbitrary_sum = sums[0]
            special_case_lower = min(Arbitrary_sum + min_job1 + min_job2, ceil((self.all_jobs_sum - Arbitrary_sum) / 2))
            lower = max(self.basic_lower_bound, max(sums), special_case_lower)
        else:
            # if there are not 3 machine - calculate 'regular' lower bound
            lower = max(self.basic_lower_bound, max(sums))

        # copy the input state, so we don't change the data
        temp_input_state = deepcopy(input_state)

        # calculating upper bound and partition using LPT
        try:
            upper, current_partition = self._LPT(temp_input_state)

        # if the state is too small - raise exception
        except NoValidState:
            raise NoValidState

        return lower, upper, current_partition

    def _LPT(self, LPT_state):
        """
        creating partition using LPT algorithm, and sets the upper bound to be the maximal value of all
        machines in this partition.

        Args:
            LPT_state: copy of the input_state

        Returns: upper bound and partition created by LPT algo
        """

        # balancing machines with odd number of jobs - s.t all machines now will have an even number of jobs
        for i in range(self.number_of_machines):
            if len(LPT_state[str(i)]) % 2 != 0:
                # checking if there are still jobs to assign, if not - the partition is illegal
                if not self.try_pop_and_move(LPT_state, source_machine=-1, destination_machine=i):
                    # the partition is illegal
                    raise NoValidState

        # Try to move and assign jobs - as long as there are jobs left
        while LPT_state['-1']:
            sum_each_machine = self.machines_sum(LPT_state)
            min_machine = np.argmin(sum_each_machine)

            self.try_pop_and_move(LPT_state, source_machine=-1, destination_machine=min_machine)
            self.try_pop_and_move(LPT_state, source_machine=-1, destination_machine=min_machine)

        # calculating upper bound by summing values of all machines, and setting the upper bound to be the maximal value
        sum_each_machine = self.machines_sum(LPT_state)
        upper = max(sum_each_machine)

        return upper, LPT_state

    def machines_sum(self, state):
        """
        summing the values of each machine in state

        Args:
            state: list of dictionaries representing the current state - i.e the machines with their according jobs

        Returns: list with sum of each machine, in the according position
        """
        sum_each_machine = [0] * self.number_of_machines
        for i in range(self.number_of_machines):
            sum_each_machine[i] = sum(state[str(i)].values())

        return sum_each_machine

    def try_pop_and_move(self, state, source_machine, destination_machine):
        """
        The function will try to move jobs from source machine to target machine,
        if the transfer was successful - will return True, and if not - returns False (this happens if there
        are no jobs left in source machine)

        Args:
            state: current input_state (i.e - the machines with their according jobs,
                   and the remaining jobs after the partition from the previous level)

            source_machine: The machine we'll try to move jobs from - this will be the machine indexed as '-1',
                            i.e - the machine that contains the remaining jobs

            destination_machine: The machine we'll try to move the jobs to

        Returns: boolean value True/False - if the transfer was successful or not
        """

        if state[str(source_machine)]:
            job_id, job_val = state[str(source_machine)].popitem()
            state[str(destination_machine)].update({job_id: job_val})
            return True
        return False


def second_smallest(numbers):
    return nsmallest(2, numbers)[-1]


def run_with_params(dict_param_num):
    """
    Start the run of the algorithm and print the data accordingly

    Args:
        dict_param_num: list of dictionaries representing the machines, and 1 dictionary that stores the jobs from input.

    Returns: None
    """

    # printing initial state, as given from the input
    print("*********************************************** New Run ***********************************************")
    print("Number of machines: ", len(dict_param_num)-1)
    print("Total number of jobs:", len(dict_param_num['-1']))
    B_B_RUN = B_B(dict_param_num)

    # Performing the first step of B&B algo, and then send it to another function to run the whole algorithm
    B_B_RUN.DFS_wrapper()

    # Printing final state after B&B has finishd its run
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Final State ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(f"In total rendered {nodes=}")
    sum_each_machine = B_B_RUN.machines_sum(B_B_RUN.leading_partition)
    print(f"{sum_each_machine=}")
    print("Number of jobs for each machine: ")
    x = [len(item) for item in B_B_RUN.leading_partition.values()]
    print(x[:-1])
    print(f"Objective function value: {B_B_RUN.current_obj_func_value}")


    for key, machine in B_B_RUN.leading_partition.items():
        if key != '-1':
            print("Machine", key, " jobs:", list(machine.values()))

def create_dict(case_id):
    """
    According each case, create list of dictionaries representing the machines, and dictionary
    that stores the remaining jobs

    Args:
        case_id: id of the current's input case we're performing the algorithm over.

    Returns: list of dictionaries s.t there are number_of_machines empty dictionaries
             and 1 dictionary indexed '-1' that stores the jobs from input.
    """

    jobs_dict = {}
    dict = {}
    if case_id == '3_14_60':
        number_of_machines = 3
        number_of_jobs = 14
        job_time = 60

        for i in range(number_of_machines):
            dict[str(i)] = {}

        for i in range(1, number_of_jobs + 1):
            jobs_dict[str(i)] = job_time

    if case_id == '3_16_60':
        number_of_machines = 3
        number_of_jobs = 16
        job_time = 60

        for i in range(number_of_machines):
            dict[str(i)] = {}

        for i in range(1, number_of_jobs + 1):
            jobs_dict[str(i)] = job_time

    if case_id == '3_20_60':
        number_of_machines = 3
        number_of_jobs = 20
        job_time = 60

        for i in range(number_of_machines):
            dict[str(i)] = {}

        for i in range(1, number_of_jobs + 1):
            jobs_dict[str(i)] = job_time

    if case_id == '123':
        number_of_jobs = 10

        number_of_machines = 2
        for i in range(number_of_machines):
            dict[str(i)] = {}

        for i in range(1, number_of_jobs + 1):
            jobs_dict[str(i)] = i

    # 20 jobs 4 machines
    if case_id == '4_20':
        number_of_machines = 4
        for i in range(number_of_machines):
            dict[str(i)] = {}

        for i in range(5):
            jobs_dict[str(i)] = 5
        for i in range(5, 15):
            jobs_dict[str(i)] = 6

        jobs_dict[str(15)] = 10

        jobs_dict[str(16)] = 11

        for i in range(17, 20):
            jobs_dict[str(i)] = 12

    # 60 jobs 16 machines
    if case_id == '16_60':
        number_of_machines = 16
        for i in range(number_of_machines):
            dict[str(i)] = {}

        for i in range(10):
            jobs_dict[str(i)] = 2

        for i in range(10, 40):
            jobs_dict[str(i)] = 3

        for i in range(40, 50):
            jobs_dict[str(i)] = 4

        for i in range(50, 60):
            jobs_dict[str(i)] = 6

    # 60 jobs 6 machines
    if case_id == '6_60':
        number_of_machines = 6
        for i in range(number_of_machines):
            dict[str(i)] = {}

        for i in range(10):
            jobs_dict[str(i)] = 2

        for i in range(10, 40):
            jobs_dict[str(i)] = 3

        for i in range(40, 50):
            jobs_dict[str(i)] = 4

        for i in range(50, 60):
            jobs_dict[str(i)] = 6

    # 12 jobs 4 machines
    if case_id == '4_12':
        number_of_machines = 4
        for i in range(number_of_machines):
            dict[str(i)] = {}

        for i, item in enumerate(
                [3, 29, 30, 31, 32, 33, 34, 35, 36, 37, 57, 58, 59, 60, 61, 72, 73, 74, 75, 98, 99, 100]):
            jobs_dict[str(i)] = item

    # 12 jobs 2 machines
    if case_id == '2_12':
        number_of_machines = 2
        for i in range(number_of_machines):
            dict[str(i)] = {}

        for i, item in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 33]):
            jobs_dict[str(i)] = item

    # 8 jobs 4 machines
    if case_id == '4_8':
        number_of_machines = 4
        for i in range(number_of_machines):
            dict[str(i)] = {}

        for i, item in enumerate([2, 2, 3, 3, 3, 3, 4, 4]):
            jobs_dict[str(i)] = item

    # 14 jobs 4 machines
    if case_id == '4_14':
        number_of_machines = 4
        for i in range(number_of_machines):
            dict[str(i)] = {}

        for i, item in enumerate([3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4]):
            jobs_dict[str(i)] = item

    # 100 jobs 5 machines
    if case_id == '5_100':
        number_of_machines = 5
        for i in range(number_of_machines):
            dict[str(i)] = {}

        for i in range(1, 101):
            jobs_dict[str(i)] = i

    # 3 Machines
    if case_id == '3_184':
        number_of_machines = 3
        for i in range(number_of_machines):
            dict[str(i)] = {}
        for i in range(10):
            jobs_dict[str(i)] = 2
        for i in range(10, 20):
            jobs_dict[str(i)] = 3
        for i in range(20, 30):
            jobs_dict[str(i)] = 4
        for i in range(30, 114):
            jobs_dict[str(i)] = 10
        for i in range(114, 154):
            jobs_dict[str(i)] = 21
        for i in range(154, 184):
            jobs_dict[str(i)] = 25

    # 5 Machines
    if case_id == '5_100':
        number_of_machines = 5
        for i in range(number_of_machines):
            dict[str(i)] = {}

        for i in range(1, 101):
            jobs_dict[str(i)] = i + 100

    # 4 Machines
    if case_id == '4_50':
        number_of_machines = 4
        for i in range(number_of_machines):
            dict[str(i)] = {}

        j = 11
        for i in range(50):
            jobs_dict[str(i)] = 2 * i + j

    # 2 Machines:
    #if case_id == '2_24':
    #    number_of_machines = 2
    #    for i in range(number_of_machines):
    #        dict[str(i)] = {}

    #    for i in range(5):
    #        jobs_dict[str(i)] = 10
    #    for i in range(5, 15):
    #        jobs_dict[str(i)] = 16
    #    for i in range(15, 19):
    #        jobs_dict[str(i)] = 20
    #    for i in range(19, 24):
    #        jobs_dict[str(i)] = 30

    # 4 machines, 18 jobs
    if case_id == '4_18_30':
        number_of_machines = 4
        for i in range(number_of_machines):
            dict[str(i)] = {}

        for i in range(18):
            jobs_dict[str(i)] = 30

    if case_id == '5_18_30':
        number_of_machines = 5
        for i in range(number_of_machines):
            dict[str(i)] = {}

        for i in range(18):
            jobs_dict[str(i)] = 30

    if case_id == '6_18_30':
        number_of_machines = 6
        for i in range(number_of_machines):
            dict[str(i)] = {}

        for i in range(18):
            jobs_dict[str(i)] = 30

    if case_id == '7_18_30':
        number_of_machines = 7
        for i in range(number_of_machines):
            dict[str(i)] = {}

        for i in range(18):
            jobs_dict[str(i)] = 30

    if case_id == '7_24_30':
        number_of_machines = 7
        for i in range(number_of_machines):
            dict[str(i)] = {}

        for i in range(24):
            jobs_dict[str(i)] = 30

    if case_id == '4_24_30':
        number_of_machines = 4
        for i in range(number_of_machines):
            dict[str(i)] = {}

        for i in range(24):
            jobs_dict[str(i)] = 30

    if case_id == '4_22_30':
        number_of_machines = 4
        for i in range(number_of_machines):
            dict[str(i)] = {}

        for i in range(22):
            jobs_dict[str(i)] = 30

    if case_id == '5_18_60':
        number_of_machines = 5
        for i in range(number_of_machines):
            dict[str(i)] = {}

        for i in range(18):
            jobs_dict[str(i)] = 60

    if case_id == '5_28_60':
        number_of_machines = 5
        for i in range(number_of_machines):
            dict[str(i)] = {}

        for i in range(28):
            jobs_dict[str(i)] = 60

    if case_id == '5_40_60':
        number_of_machines = 5
        for i in range(number_of_machines):
            dict[str(i)] = {}

        for i in range(40):
            jobs_dict[str(i)] = 60

    if case_id == '5_54_60':
        number_of_machines = 5
        for i in range(number_of_machines):
            dict[str(i)] = {}

        for i in range(54):
            jobs_dict[str(i)] = 60

    if case_id == '3_20_rand':
        number_of_machines = 3

        for i in range(number_of_machines):
            dict[str(i)] = {}

        for i in range(20):
            jobs_dict[str(i)] = randrange(10)

    if case_id == '4_20_rand':
        number_of_machines = 4

        for i in range(number_of_machines):
            dict[str(i)] = {}

        for i in range(20):
            jobs_dict[str(i)] = randrange(10)

    if case_id == '4_40_rand':
        number_of_machines = 4

        for i in range(number_of_machines):
            dict[str(i)] = {}

        for i in range(40):
            jobs_dict[str(i)] = randrange(10,20)

    if case_id == '5_10_rand':
        number_of_machines = 5

        for i in range(number_of_machines):
            dict[str(i)] = {}

        for i in range(10):
            jobs_dict[str(i)] = randrange(15, 60)

    if case_id == '3_18_rand':
        number_of_machines = 3

        for i in range(number_of_machines):
            dict[str(i)] = {}

        for i in range(18):
            jobs_dict[str(i)] = randrange(15, 25)

    if case_id == '5_30_rand':
        number_of_machines = 5

        for i in range(number_of_machines):
            dict[str(i)] = {}

        for i in range(30):
            jobs_dict[str(i)] = randrange(15, 25)

    if case_id == '5_30_rand_try':
        number_of_machines = 5

        for i in range(number_of_machines):
            dict[str(i)] = {}

        for i, k in enumerate([22, 15, 19, 20, 23, 16,18, 18, 22, 21, 15, 20 , 15, 18, 18, 21, 15, 22 ,21, 17, 18, 24, 16, 23 ,19, 19, 15, 18, 19, 18]):
            jobs_dict[str(i)] = k

# Input from Leah 31/05:

    if case_id == '3_12':
        number_of_machines = 3

        for i in range(number_of_machines):
            dict[str(i)] = {}

        for i,item in enumerate([17, 18, 19, 21, 22, 26, 30, 32, 34, 35, 36, 40]):
            jobs_dict[str(i)] = item

    if case_id == '4_22':
        number_of_machines = 4

        for i in range(number_of_machines):
            dict[str(i)] = {}

        for i in range(4):
            jobs_dict[str(i)] = 12

        for i in range(4,22):
            jobs_dict[str(i)] = 8

    if case_id == '2_24':
        number_of_machines = 2

        for i in range(number_of_machines):
            dict[str(i)] = {}

        for i in range(16):
            jobs_dict[str(i)] = i + 1

        for i in range(16, 24):
            jobs_dict[str(i)] = 17

    if case_id == '5_16':
        number_of_machines = 5

        for i in range(number_of_machines):
            dict[str(i)] = {}

        for i,item in enumerate([5, 5, 6, 6, 7, 7, 7, 7, 8, 8, 9, 9, 10, 10, 18, 18]):
            jobs_dict[str(i)] = item

    if case_id == '3_rand':
        number_of_machines = 3

        for i in range(number_of_machines):
            dict[str(i)] = {}

        for i in range(32):
            jobs_dict[str(i)] = randrange(15, 25)
    jobs_dict = {k: v for k, v in sorted(jobs_dict.items(), key=lambda key_value_tuple: key_value_tuple[1])}

    if case_id == '2_rand':
        number_of_machines = 2

        for i in range(number_of_machines):
            dict[str(i)] = {}

        for i in range(3000):
            jobs_dict[str(i)] = randrange(15, 25)
    jobs_dict = {k: v for k, v in sorted(jobs_dict.items(), key=lambda key_value_tuple: key_value_tuple[1])}

    if case_id == '5_rand':
        number_of_machines = 5

        for i in range(number_of_machines):
            dict[str(i)] = {}

        for i in range(14):
            jobs_dict[str(i)] = randrange(15, 25)
    jobs_dict = {k: v for k, v in sorted(jobs_dict.items(), key=lambda key_value_tuple: key_value_tuple[1])}

    dict['-1'] = jobs_dict

    return dict


if __name__ == '__main__':
    # sys.stdout = open(r'C:\Users\user1\PycharmProjects\partitioning-algorithm\BB_output\out1.txt', mode='a')

    for version_id in ["5_rand"]:
    #for version_id in ["3_14_60", "3_16_60", "3_20_60", "2_12", "4_8", "4_14",  "3_184", "5_100", "2_24", "5_18_30", "6_18_30", "7_18_30", "7_24_30", "4_24_30", "4_22_30", "5_18_60", "5_28_60", "5_40_60", "5_54_60"]:
        # "3_20_rand", "4_20_rand", "4_40_rand",
        dict1 = create_dict(version_id)
        run_with_params(dict1)
