"""
From user:
1:how many machines (more than two)
2:how many jobs (need to be divisible by two)
3: lower and upper bound of processing time

Then: draw processing time for each machine according to the lower and upper bounds given by the user and assign p.t for each job.
"""
import numpy as np
# np.random.seed(0)  # todo delete before submission


MIN_NUM_MACHINES = 2


def get_parameters():
    while True:
        try:
            number_of_machines = int(input("How many machines: "))  # TODO make sure valid (int, >= 2) input
            if number_of_machines >= MIN_NUM_MACHINES:
                break
            else:
                raise ValueError
        except ValueError:
            print("Please enter a valid number, integer >= 2.")

    while True:
        try:
            jobs_num = int(input("How many jobs: "))  # TODO make sure valid - divisible by two
            if jobs_num % 2 == 0 and jobs_num>0:
                break
            else:
                raise ValueError
        except ValueError:
            print("Please enter a positive number divisible by two")

    while True:
        try:
            lower_num = int(input("Lower bound: "))  # TODO make sure valid- non-negative number
            if lower_num >= 0:
                break
            else:
                raise ValueError
        except ValueError:
            print("Please enter a non-negative integer")

    while True:
        try:
            upper_num = int(input("Upper bound: "))  # TODO make sure valid
            if upper_num > lower_num:
                break
            else:
                raise ValueError
        except ValueError:
            print("Please print a value bigger than lower bound")

    jobs_process_time = np.random.randint(lower_num, upper_num, jobs_num)
    print("Drawing job times...")
    print("Job times: ", jobs_process_time)
    print("Number of machines: ", number_of_machines)
    return jobs_process_time, number_of_machines

