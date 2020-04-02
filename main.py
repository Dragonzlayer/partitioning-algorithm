from init_parameters import get_parameters

if __name__ == '__main__':
    process_time, machines_num = get_parameters()


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