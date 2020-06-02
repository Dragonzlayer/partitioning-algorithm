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
# TODO: fix upper bound calculation

class NoValidState(Exception):
    """Raised when the input value is too small"""
    pass


class B_B:
    def __init__(self, input_state):
        self.number_of_machines = len(input_state) - 1
        self.basic_lower_bound = ceil(sum(input_state['-1'].values()) / self.number_of_machines)
        self.max_job = max(input_state['-1'].values())
        self.input_state = input_state
        self.leading_partition = {}
        self.current_obj_func_value = 0
        self.all_jobs_sum = sum(input_state['-1'].values())

    def _update_state(self, state, i):
        self.try_pop_and_move(state, source_machine=-1, destination_machine=i)

    def _undo_update(self, state, i):
        self.try_pop_and_move(state, source_machine=i, destination_machine=-1)

    @timer
    def DFS_wrapper(self):
        # lower bound for the root is a bit different - needs to include also max job
        # the first upper bound is the trivial one - sum of all jobs
        input_state_copy = deepcopy(self.input_state)
        # Using _bounds_calc - creates the first leading partition and assigns the first objective function value
        try:
            temp_lower, temp_upper, self.leading_partition = self._bounds_calc(input_state_copy)
        except NoValidState:
            print("Illegal input. No Valid State")
            raise NoValidState
        # print("temp upper:", temp_upper, "temp lower:", temp_lower)

       # number_of_jobs = len(self.input_state['-1'].values())
       # lcm_val = np.lcm(2, self.number_of_machines)
       # num_jobs_for_min = number_of_jobs + lcm_val - number_of_jobs % lcm_val
       # new_lower = min(self.input_state['-1'].values()) * num_jobs_for_min / self.number_of_machines
        self.basic_lower_bound = max(temp_lower, self.max_job) #new_lower

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
        # print("Went down level")
        # print(state)
        level += 1
        if not input_state['-1']:
            return

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
                # TODO: assign the dictionaries properly
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
                # print(f"{lower=}=upper. {self.leading_partition=}")
                self._undo_update(state, i)
                continue

            # print(f"{self.leading_partition=}")
            self.DFS(state, level)

            self._undo_update(state, i)

    def _bounds_calc(self, input_state):
        # calculates sum of jobs for each machine
        sums = self.machines_sum(input_state)
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
            lower = max(self.basic_lower_bound, max(sums))

        temp_input_state = deepcopy(input_state)

        try:
            upper, current_partition = self._LPT(temp_input_state)
        except NoValidState:
            raise NoValidState
        return lower, upper, current_partition

    def _LPT(self, LPT_state):
        # balancing machines with odd number of jobs - s.t all machines now will have an even number of jobs
        for i in range(self.number_of_machines):
            if len(LPT_state[str(i)]) % 2 != 0:
                # checking if there are still jobs to assign, if not - the partition is illegal
                if not self.try_pop_and_move(LPT_state, source_machine=-1, destination_machine=i):
                    # the partition is illegal
                    raise NoValidState

        while LPT_state['-1']:
            sum_each_machine = self.machines_sum(LPT_state)
            min_machine = np.argmin(sum_each_machine)

            self.try_pop_and_move(LPT_state, source_machine=-1, destination_machine=min_machine)
            self.try_pop_and_move(LPT_state, source_machine=-1, destination_machine=min_machine)

        # print(f"{LPT_state=}")

        sum_each_machine = self.machines_sum(LPT_state)
        upper = max(sum_each_machine)
        return upper, LPT_state

    def machines_sum(self, state):
        sum_each_machine = [0] * self.number_of_machines
        for i in range(self.number_of_machines):
            sum_each_machine[i] = sum(state[str(i)].values())

        return sum_each_machine

    def try_pop_and_move(self, state, source_machine, destination_machine):

        if state[str(source_machine)]:
            job_id, job_val = state[str(source_machine)].popitem()
            state[str(destination_machine)].update({job_id: job_val})
            return True
        return False


def second_smallest(numbers):
    return nsmallest(2, numbers)[-1]


def run_with_params(dict_param_num):
    print("*********************************************** New Run ***********************************************")
    print("Number of machines: ", len(dict_param_num)-1)
    print("Total number of jobs:", len(dict_param_num['-1']))
    B_B_RUN = B_B(dict_param_num)

    B_B_RUN.DFS_wrapper()

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
    # for case_num in range(1, 2):
    # jobs_dict, number_of_machines = get_parameters(13)
    # print(f"{case_id}")
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

        for i in range(1200):
            jobs_dict[str(i)] = randrange(15, 25)
    jobs_dict = {k: v for k, v in sorted(jobs_dict.items(), key=lambda key_value_tuple: key_value_tuple[1])}

    dict['-1'] = jobs_dict

    return dict


if __name__ == '__main__':
    # sys.stdout = open(r'C:\Users\user1\PycharmProjects\partitioning-algorithm\BB_output\out1.txt', mode='a')

    for version_id in ["2_rand"]:
    #for version_id in ["3_14_60", "3_16_60", "3_20_60", "2_12", "4_8", "4_14",  "3_184", "5_100", "2_24", "5_18_30", "6_18_30", "7_18_30", "7_24_30", "4_24_30", "4_22_30", "5_18_60", "5_28_60", "5_40_60", "5_54_60"]:
        # "3_20_rand", "4_20_rand", "4_40_rand",
        dict1 = create_dict(version_id)
        run_with_params(dict1)
