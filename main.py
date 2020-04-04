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
    
        select a job_to_move from jobs_to_move  # TODO select how? 
        
        available_machines = all_machines minus max_machine
        
        while available_machines not empty:
        
            select destination_machine from available_machines #TODO select how?
            
            temp_new_state = curr_state plus job_to_move was moved to destination_machine
            
            if obj_func(new_state) < obj_func(curr_state): # TODO implement more effienect look only at changes - look at delta)
            
                curr_state = temp_new_state
                max_machine = the machine with maximal processing time
                jobs_to_moves = all jobs from max_machine
                search_space = search_space.reset()
                break         
                
            else:
                available_machines = available_machines minus destination_machine
        
        jobs_to_move = jobs_to_move minus job_to_move       
        
    return curr_state
"""
import numpy as np
from copy import deepcopy
from init_parameters import get_parameters


class LocalSearch:
    def __init__(self, state):
        self.curr_state = state
        self.sum_pt = np.array([sum(np.array(element)) for element in self.curr_state])
        self.sum_squared_pt = np.array([sum(np.array(element) ** 2) for element in self.curr_state])
        self.max_sum_pt = max(self.sum_pt)
        self.max_sum_squared_pt = max(self.sum_squared_pt)
        self.temp_sum_pt = []
        self.temp_squared_sum_pt = []
        self.max_search_space = len(self.curr_state[0]) - 1

    def search(self):
        min_search_space = 2
        search_space = min_search_space

        # print(self.curr_state)
        while search_space < self.max_search_space:

            is_changed = self._transfer_jobs(search_space)
            if not is_changed:
                break
            else:
                search_space = search_space + 2
            print("State with search space {}: {}".format(search_space, self.curr_state))

    def _transfer_jobs(self, search_space):
        is_changed = False

        max_machine = np.argmax(self.sum_pt)  # TODO check if working
        # print("max machine = ", max_machine)

        available_jobs_to_move = list(self.curr_state[max_machine])
        # print("jobs to move = ", available_jobs_to_move)

        while available_jobs_to_move:

            self.max_search_space = len(available_jobs_to_move)

            if search_space < self.max_search_space:
                jobs_to_move = np.array([available_jobs_to_move.pop(0) for i in range(search_space)])  # TODO check for improvements - right now it pops the first/last element
            else:
                jobs_to_move = np.array([available_jobs_to_move.pop(0) for i in range(self.max_search_space)])
            # print(jobs_to_move)

            available_machines = [i for i in range(machines_num)]

            # print("available machines: ", available_machines)

            available_machines.pop(max_machine)

            # print("available machines after removal of max_machine: ", available_machines)

            while available_machines:

                destination_machine = available_machines[0]

                if self._can_improve(max_machine, destination_machine, jobs_to_move):

                    self.curr_state[destination_machine] = np.append(self.curr_state[destination_machine], jobs_to_move)
                    self.curr_state[max_machine] = np.delete(self.curr_state[max_machine], list(range(search_space)))

                    self.sum_pt = self.temp_sum_pt
                    self.max_sum_pt = max(self.sum_pt)

                    self.sum_squared_pt = self.temp_squared_sum_pt
                    self.max_sum_squared_pt = max(self.sum_squared_pt)

                    max_machine = np.argmax(self.sum_pt)

                    available_jobs_to_move = list(self.curr_state[max_machine])

                    search_space = 2

                    is_changed = True

                    break
                else:
                    available_machines.remove(destination_machine)

                # print("current state: ", curr_state)

        return is_changed

    def _can_improve(self, max_machine, destination_machine, job_to_move):
        self.temp_sum_pt = deepcopy(self.sum_pt)
        self.temp_sum_pt[max_machine] -= sum(job_to_move)
        self.temp_sum_pt[destination_machine] += sum(job_to_move)
        max_temp_sum_pt = max(self.temp_sum_pt)

        self.temp_squared_sum_pt = deepcopy(self.sum_squared_pt)
        self.temp_squared_sum_pt[max_machine] -= sum(job_to_move ** 2)
        self.temp_squared_sum_pt[destination_machine] += sum(job_to_move ** 2)
        max_temp_squared_sum_pt = max(self.temp_squared_sum_pt)
        # print("sum: ", max_temp_sum_pt, "Squared: ", max_temp_squared_sum_pt)
        # print("sum: ", self.temp_sum_pt, "Squared: ", self.temp_squared_sum_pt)
        return max_temp_sum_pt < self.max_sum_pt or max_temp_squared_sum_pt < self.max_sum_squared_pt


if __name__ == '__main__':
    process_time, machines_num = get_parameters()

    curr_state = [[] for machine in range(machines_num)]
    # print(machines)
    curr_state[0] = process_time
    # print(machines)

    local_searcher = LocalSearch(curr_state)
    local_searcher.search()

    print("\nFinal state:",local_searcher.curr_state)

    print(local_searcher.sum_pt, local_searcher.sum_squared_pt)
