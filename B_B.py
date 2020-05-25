from Timer import timer
from copy import deepcopy
from math import ceil
import numpy as np
from heapq import nsmallest
import sys

from init_parameters import get_parameters

nodes = 1
# number_of_jobs = 10
number_of_machines = 2
# TODO: fix upper bound calculation

class NoValidState(Exception):
   """Raised when the input value is too small"""
   pass

class B_B:
    def __init__(self, input_state):
        self.number_of_machines = len(input_state) - 1
        self.basic_lower_bound = ceil(sum(input_state['-1'].values()) / number_of_machines)
        self.max_job = max(input_state['-1'].values())
        self.input_state = input_state
        self.leading_partition = {}
        self.current_obj_func_value = 0
        self.all_jobs_sum = sum(input_state['-1'].values())

    def _update_state(self, state, i):
        self.try_pop_and_move(state, source_machine=-1, destination_machine=i)

    def _undo_update(self, state, i):
        self.try_pop_and_move(state, source_machine=i, destination_machine=-1 )

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
        #print("temp upper:", temp_upper, "temp lower:", temp_lower)

        number_of_jobs = len(self.input_state['-1'].values())
        lcm_val = np.lcm(2, self.number_of_machines)
        num_jobs_for_min = number_of_jobs + lcm_val - number_of_jobs % lcm_val
        new_lower = min(self.input_state['-1'].values()) * num_jobs_for_min/self.number_of_machines
        self.basic_lower_bound = max(temp_lower, self.max_job, new_lower)

        upper = ceil(sum(self.input_state['-1'].values()))
        self.current_obj_func_value = upper
        print("initial bounds: Lower = ", self.basic_lower_bound, "upper:", upper)

        # First layer is symmetric - run on first machine and cont from there
        self._update_state(self.input_state, 0)

        print("1st leading partition:", self.leading_partition)

        self.DFS(self.input_state, 1)

    def DFS(self, input_state, level):
        #print("Went down level")
        # print(state)
        level += 1
        if not input_state['-1']:
            return

        state = deepcopy(input_state)

        for i in range(self.number_of_machines):
            #print(f"Checking out sibling {i=}")
            self._update_state(state, i)
            try:
                lower, upper, current_partition = self._bounds_calc(state)
            except NoValidState:
                #print("No valid State ")
                self._undo_update(state, i)
                continue

            #print(f"{level=}, {state=}")
            #print(f"{lower=}, {upper=}")
            global nodes
            nodes += 1
            if nodes % 10000 == 0:
                 print(nodes)

            # truncate the node and don't create a subtree for it.
            # keep searching in his adjacent nodes (not in his subtree)
            if lower >= self.current_obj_func_value:
                #print(f"entered if clause with {i=}: {lower} >= {self.current_obj_func_value}")
                self._undo_update(state, i)
                continue

            # we've found a better solution - so:
            # Replace the leading_partition to this partition and change current_obj_func_value
            # to the upper bound of this partition
            elif upper < self.current_obj_func_value:
                # TODO: assign the dictionaries properly
                #print(f"{upper} < {self.current_obj_func_value}. {self.leading_partition=}")
                self.leading_partition = current_partition
                self.current_obj_func_value = upper
                print(f"New {self.leading_partition=}")


            # truncate the node and don't create a subtree for it.
            # keep searching in his adjacent nodes (not in his subtree)
            if lower == upper:
                #print(f"{lower=}=upper. {self.leading_partition=}")
                self._undo_update(state, i)
                continue

            # print(f"{self.leading_partition=}")
            self.DFS(state, level)

            self._undo_update(state, i)

    def _bounds_calc(self, input_state):
        # calculates sum of jobs for each machine
        sums = self.machines_sum(input_state)
        if self.number_of_machines == 3:
            x=input_state['-1'].values()
            if len(x)>1:
                min_job1, min_job2 = nsmallest(2, x)
            else:
                min_job1 = 0
                min_job2 = 0
            Arbitrary_sum = sums[0]
            special_case_lower = min(Arbitrary_sum+min_job1+min_job2, ceil((self.all_jobs_sum - Arbitrary_sum)/2))
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
        for i in range(number_of_machines):
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
        for i in range(number_of_machines):
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
    print("--------------- New Run --------------")

    B_B_RUN = B_B(dict_param_num)

    B_B_RUN.DFS_wrapper()
    print(f"In total rendered {nodes=}")
    sum_each_machine = B_B_RUN.machines_sum(B_B_RUN.leading_partition)
    print("Number of jobs for each machine: ")
    x = [len(item) for item in B_B_RUN.leading_partition.values()]
    print(x[:-1])
    print(f"{sum_each_machine=}, {B_B_RUN.current_obj_func_value=}, Final state: {B_B_RUN.leading_partition}")


def create_dict(case_id):
    #for case_num in range(1, 2):
    #jobs_dict, number_of_machines = get_parameters(13)

    jobs_dict = {}
    dict = {}
    if case_id == '3_14_60':
        number_of_machines = 3
        number_of_jobs = 14
        job_time = 60

        for i in range(number_of_machines):
            dict[str(i)] = {}

        for i in range(1,number_of_jobs+1):
            jobs_dict[str(i)] = job_time
        dict['-1'] = jobs_dict

    if case_id == '3_16_60':
        number_of_machines = 3
        number_of_jobs = 16
        job_time = 60

        for i in range(number_of_machines):
            dict[str(i)] = {}

        for i in range(1,number_of_jobs+1):
            jobs_dict[str(i)] = job_time
        dict['-1'] = jobs_dict

    if case_id == '3_20_60':
        number_of_machines = 3
        number_of_jobs = 20
        job_time = 60

        for i in range(number_of_machines):
            dict[str(i)] = {}

        for i in range(1,number_of_jobs+1):
            jobs_dict[str(i)] = job_time
        dict['-1'] = jobs_dict

    if case_id == '123':
        number_of_machines = 2
        number_of_jobs = 10

        for i in range(number_of_machines):
            dict[str(i)] = {}

        for i in range(1,number_of_jobs+1):
            jobs_dict[str(i)] = i
        dict['-1'] = jobs_dict

    return dict


if __name__ == '__main__':
    # sys.stdout = open(r'C:\Users\user1\PycharmProjects\partitioning-algorithm\BB_output\out1.txt', mode='a')

    for version_id in [ "123", "3_14_60", "3_16_60", "3_20_60"]:

        dict1 = create_dict(version_id)
        run_with_params(dict1)


