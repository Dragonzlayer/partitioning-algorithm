from Timer import timer
from copy import deepcopy
from math import ceil
import numpy as np
nodes = 1
number_of_jobs = 10000
number_of_machines =7
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
        lower = max(temp_lower, self.max_job)
        upper = ceil(sum(self.input_state['-1'].values()))
        self.current_obj_func_value = upper
        print("initial bounds: Lower = ", lower, "upper:", upper)

        # First layer is symmetric - run on first machine and cont from there
        self._update_state(self.input_state, 0)

        print("1st leading partition:", self.leading_partition)

        self.DFS(self.input_state, 1)

    def DFS(self, input_state, level):
        print("Went down level")
        # print(state)
        level += 1
        if not input_state['-1']:
            return

        state = deepcopy(input_state)

        for i in range(self.number_of_machines):
            print(f"Checking out sibling {i=}")
            self._update_state(state, i)
            try:
                lower, upper, current_partition = self._bounds_calc(state)
            except NoValidState:
                print("No valid State ")
                self._undo_update(state, i)
                continue

            print(f"{level=}, {state=}")
            print(f"{lower=}, {upper=}")
            global nodes
            nodes += 1

            # truncate the node and don't create a subtree for it.
            # keep searching in his adjacent nodes (not in his subtree)
            if lower >= self.current_obj_func_value:
                print(f"entered if clause with {i=}: {lower} >= {self.current_obj_func_value}")
                self._undo_update(state, i)
                continue

            # we've found a better solution - so:
            # Replace the leading_partition to this partition and change current_obj_func_value
            # to the upper bound of this partition
            elif upper < self.current_obj_func_value:
                # TODO: assign the dictionaries properly
                print(f"{upper} < {self.current_obj_func_value}. {self.leading_partition=}")
                self.leading_partition = current_partition
                self.current_obj_func_value = upper


            # truncate the node and don't create a subtree for it.
            # keep searching in his adjacent nodes (not in his subtree)
            if lower == upper:
                print(f"{lower=}=upper. {self.leading_partition=}")
                self._undo_update(state, i)
                continue

            # print(f"{self.leading_partition=}")
            self.DFS(state, level)

            self._undo_update(state, i)

    def _bounds_calc(self, input_state):
        # calculates sum of jobs for each machine
        sums = self.machines_sum(input_state)

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

        print(f"{LPT_state=}")

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



if __name__ == '__main__':

    dict = {}
    for i in range(number_of_machines):
        dict[str(i)] = {}
    jobs_dict = {}
    for i in range(1, number_of_jobs + 1):
        jobs_dict[str(i)] = i

    dict['-1'] = jobs_dict
    B_B_RUN = B_B(dict)

    B_B_RUN.DFS_wrapper()
    print(f"In total rendered {nodes=}")
    sum_each_machine = B_B_RUN.machines_sum(B_B_RUN.leading_partition)
    print(f"{sum_each_machine=}, {B_B_RUN.current_obj_func_value=}, Final state: {B_B_RUN.leading_partition}")
