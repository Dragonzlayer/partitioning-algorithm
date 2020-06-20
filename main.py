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
import sys

import numpy as np

from init_parameters import get_parameters
from ls import LocalSearch
from Genetic import genetic
DEBUG = False  # Switch to True for prints during the run
#from Timer import timer

RUN_LS = False
RUN_BB = False
RUN_GENETIC = True

def run_ls(state):
    # initialize LocalSearch class with initial state
    local_searcher = LocalSearch(state)
    print("Performing Local search...")
    # perform local search algorithm
    local_searcher.search()

    print("Final state:")
    for i, k in enumerate(local_searcher.curr_state):
        print("machine nr. ", i, ":", k)

    print("Number of jobs for each machine: ")
    print([len(item) for item in local_searcher.curr_state])
    print("Sum of process times for each machine: ", local_searcher.sum_processing_times_per_machine)
    print("Squared sum of process times for each machine: ", local_searcher.sum_squared_processing_times)

def run_genetic(jobs_process_time, number_of_machines):

    genetic_run = genetic(jobs_process_time, number_of_machines)
    genetic_run.action()


def main():
    # sys.stdout = open(r'C:\Users\user1\PycharmProjects\partitioning-algorithm\local-search_output\output.txt', mode='a')
    # sys.stdout = open(r'C:\Users\user1\PycharmProjects\partitioning-algorithm\local-search_output\output2.txt',
    #                  mode='a')
    print("--------------- New Run --------------")
    # receive input from user and draw process times
    # jobs_process_time, number_of_machines = get_parameters(CASE=1)

    jobs_process_time = []

    number_of_machines = 4

    # 14 jobs 4 machines

    jobs_process_time=[3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4]


    """
    # 20 jobs 4 machines
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
    """

    # initializing: putting all jobs in the first machine
    # initial_state = [{} for machine in range(number_of_machines)]

    # initializing step 2: performing version of LPT to distribute the jobs
    # while jobs_process_time:
    #    sum_processing_times_per_machine = np.array([sum(element.values()) for element in initial_state])
    #    min_machine = np.argmin(sum_processing_times_per_machine)
    #    job_id_1, job_val_1 = jobs_process_time.popitem()
    #    job_id_2, job_val_2 = jobs_process_time.popitem()
    #    initial_state[min_machine][job_id_1] = job_val_1
    #    initial_state[min_machine][job_id_2] = job_val_2

    # for i, k in enumerate(initial_state):
    #    print("machine nr. ", i, ":", k)

#    if RUN_LS:
#       run_ls(initial_state)

#    if RUN_BB:
#        pass # TODO


    if RUN_GENETIC:
        run_genetic(jobs_process_time, number_of_machines)


if __name__ == '__main__':
   main()
