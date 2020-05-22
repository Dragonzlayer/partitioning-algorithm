from Timer import timer
from copy import deepcopy
from math import ceil
import numpy as np

nodes = 1


class B_B:
    def __init__(self, input_state):
        self.number_of_machines = len(input_state) - 1
        self.basic_lower_bound = ceil(sum(input_state['-1'].values()) / number_of_machines)
        self.max_job = max(input_state['-1'].values())
        self.input_state = input_state

    def _update_state(self, state, i):
        job_id, job_value = state['-1'].popitem()

        state[str(i)].update({job_id: job_value})
        return state

    def _undo_update(self, state, i):
        job_id, job_value = state[str(i)].popitem()

        state['-1'].update({job_id: job_value})
        return state

    @timer
    def DFS_wrapper(self):
        # First layer is symmetric - run on first machine and cont from there
        state = self._update_state(self.input_state, 0)

        # lower bound for the root is a bit different - includes also max job
        temp_lower, upper = self._bounds_calc(self.input_state)
        lower = max(temp_lower, self.max_job)

        self.DFS(state, 1)

    def DFS(self, input_state, level):
        # print(state)
        level += 1
        if not input_state['-1']:
            return

        state = deepcopy(input_state)

        for i in range(self.number_of_machines):
            state = self._update_state(state, i)
            lower, upper = self._bounds_calc(input_state)

            print(f"{level=}, {state=}")
            print(f"{lower=}, {upper=}")
            global nodes
            nodes += 1
            self.DFS(state, level)

            state = self._undo_update(state, i)

    def _bounds_calc(self, input_state):
        # calculates sum of jobs for each machine
        sums = [0] * self.number_of_machines
        for i in range(number_of_machines):
            sums[i] = sum(input_state[str(i)].values())

        lower = max(self.basic_lower_bound, max(sums))

        temp_input_state = deepcopy(input_state)

        upper = self._LPT(temp_input_state)

        return lower, upper

    def _LPT(self, LPT_state):
        # balancing machines with odd number of jobs - s.t all machines now will have an even number of jobs
        for i in range(number_of_machines):
            if len(LPT_state[str(i)]) % 2 != 0:
                job_id, job_value = LPT_state['-1'].popitem()
                LPT_state[str(i)].update({job_id: job_value})

        sum_each_machine = [0] * self.number_of_machines
        for i in range(number_of_machines):
            sum_each_machine[i] = sum(LPT_state[str(i)].values())

        min_machine = np.argmin(sum_each_machine)

        if LPT_state['-1']:
            job_id_1, job_val_1 = LPT_state['-1'].popitem()
            LPT_state[str(i)].update({job_id_1: job_val_1})
        if LPT_state['-1']:
            job_id_2, job_val_2 = LPT_state['-1'].popitem()
            LPT_state[str(i)].update({job_id_2: job_val_2})

        print(f"{LPT_state=}")
        upper = max(sum_each_machine)
        return upper


if __name__ == '__main__':
    number_of_jobs = 4
    number_of_machines = 2
    dict = {}
    for i in range(number_of_machines):
        dict[str(i)] = {}
    jobs_dict = {}
    for i in range(1, number_of_jobs + 1):
        jobs_dict[str(i)] = i

    dict['-1'] = jobs_dict
    B_B_RUN = B_B(dict)

    B_B_RUN.DFS_wrapper()
    print(nodes)
