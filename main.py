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
import copy
from init_parameters import get_parameters


def transfer_jobs(curr_state, search_space):
    is_changed = False
    sum_pt = [sum(element) for element in machines]
    sum_pt = np.array(sum_pt)
    max_sum_pt = max(sum_pt)
    max_machine = np.argmax(sum_pt)  # TODO check if working
    print("max machine = ", max_machine)

    jobs_to_move = list(curr_state[max_machine])
    print("jobs to move = ", jobs_to_move)

    while jobs_to_move:

        job_to_move = jobs_to_move.pop(0)  # TODO check for improvements - right now it pops the first/last element
        print(job_to_move)

        available_machines = [i for i in range(machines_num)]

        print("available machines: ", available_machines)

        available_machines.pop(max_machine)

        print("available machines after removal of max_machine: ", available_machines)

        while available_machines:

            destination_machine = available_machines[0]

            temp_sum_pt = copy.deepcopy(sum_pt)
            temp_sum_pt[max_machine] -= job_to_move
            temp_sum_pt[destination_machine] += job_to_move

            max_temp_sum_pt = max(temp_sum_pt)

            if max_temp_sum_pt < max_sum_pt:
                curr_state[destination_machine] = np.append(curr_state[destination_machine], job_to_move)
                curr_state[max_machine] = np.delete(curr_state[max_machine], 0)

                sum_pt = temp_sum_pt

                max_sum_pt = max(sum_pt)

                max_machine = np.argmax(sum_pt)

                jobs_to_move = list(curr_state[max_machine])

                search_space = 1

                is_changed = True

                break
            else:
                available_machines.remove(destination_machine)

            print("current state: ", curr_state)

    return is_changed


if __name__ == '__main__':
    process_time, machines_num = get_parameters()

    machines = [[] for machine in range(machines_num)]
    print(machines)
    machines[0] = process_time
    print(machines)

    min_search_space = 1
    search_space = min_search_space
    max_search_space = len(machines[0]) - 1

    curr_state = machines

    while search_space < max_search_space:

        is_changed = transfer_jobs(curr_state, search_space)
        if not is_changed:
            break
        else:
            search_space = search_space + 1
