from init_parameters import get_parameters

if __name__ == '__main__':
    process_time, machines_num = get_parameters()

"""
Local search:(greedy approach)
1: put all jobs to machine #1

max_machine = the machine with maximal processing time
jobs_to_moves = jobs from max_machine
while jobs_to_move not empty:
    select a job_to_move from jobs_to_moves (search space - k=1) 
    available_machines = all_machines minus machine with maximal processing time
    if available_machines not empty:
        select destination_machine from available_machines
        new_state = curr_state + job_to_move was moved to destination_machine
        if obj_func(new_state) < obj_func(curr_state): # TODO implement more effienect look only at changes - look at delta)
            curr_state = new_state
            
    
5. if obj func can be improved(ie remaining available jobs > 0):
        goto 2 with search space 1
6. goto 2 with search space k + 1
"""