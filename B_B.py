from Timer import timer
from copy import deepcopy

nodes = 1


class B_B:
    def __init__(self, number_of_machines):
        self.number_of_machines = number_of_machines

    def _update_state(self, state, i):
        job_id, job_value = state['-1'].popitem()

        state[str(i)].update({job_id: job_value})
        return state

    def _undo_update(self, state, i):
        job_id, job_value = state[str(i)].popitem()

        state['-1'].update({job_id: job_value})
        return state

    @timer
    def DFS_wrapper(self, input_state):
        #First layer is symmetric - run on first machine and cont from there
        state = self._update_state(input_state, 0)
        self.DFS(state, 1)

    def DFS(self, input_state, level):
        # print(state)
        level += 1
        if not input_state['-1']:
            return

        state = deepcopy(input_state)

        for i in range(self.number_of_machines):
            state = self._update_state(state, i)

            print(f"{level=}, {state=}")
            global nodes
            nodes += 1
            self.DFS(state, level)

            state = self._undo_update(state, i)





if __name__ == '__main__':
    number_of_jobs =  3
    number_of_machines = 2
    dict = {}
    for i in range(number_of_machines):
        dict[str(i)] = {}
    jobs_dict = {}
    for i in range(1, number_of_jobs+1):
        jobs_dict[str(i)] = i

    dict['-1'] = jobs_dict
    B_B_RUN = B_B(number_of_machines)

    B_B_RUN.DFS_wrapper(dict)
    print(nodes)
