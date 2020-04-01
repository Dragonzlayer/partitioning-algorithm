"""
From user:
1:how many machines (more than two)
2:how many jobs (need to be divisible by two)
3: lower and upper bound of processing time

Then: draw processing time for each machine according to the lower and upper bounds given by the user and assign p.t for each job.
"""
import numpy as np


MIN_NUM_MACHINES = 2


def get_parameters():
    while True:
        try:
            machines_num = int(input("How many machines: "))  # TODO make sure valid (int, >= 2) input
            if machines_num >= MIN_NUM_MACHINES:
                break
        except ValueError:
            print("Please enter a valid number, integer >= 2.")
    jobs_num = int(input("How many jobs: "))  # TODO make sure valid - divisible by two
    lower_num = int(input("Lower bound: "))  # TODO make sure valid
    upper_num = int(input("Upper bound: "))  # TODO make sure valid
    process_time = np.random.randint(lower_num, upper_num, jobs_num)
    return process_time, machines_num

