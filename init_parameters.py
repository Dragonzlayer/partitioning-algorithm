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
CASE = 10

def get_parameters():
    while True:
        try:
            number_of_machines = int(input("Number of machines: "))  # TODO make sure valid (int, >= 2) input
            if number_of_machines >= MIN_NUM_MACHINES:
                print(number_of_machines)
                break
            else:
                raise ValueError
        except ValueError:
            print("Please enter a valid number, integer >= 2.")

    while True:
        try:
            jobs_num = int(input("Number of jobs: "))  # TODO make sure valid - divisible by two
            if jobs_num % 2 == 0 and jobs_num>0:
                print(jobs_num)
                break
            else:
                raise ValueError
        except ValueError:
            print("Please enter a positive number divisible by two")

    while True:
        try:
            lower_num = int(input("Lower bound: "))  # TODO make sure valid- non-negative number
            if lower_num >= 0:
                print(lower_num)
                break
            else:
                raise ValueError
        except ValueError:
            print("Please enter a non-negative integer")

    while True:
        try:
            upper_num = int(input("Upper bound: "))  # TODO make sure valid
            if upper_num > lower_num:
                print(upper_num)
                break
            else:
                raise ValueError
        except ValueError:
            print("Please print a value bigger than lower bound")

    jobs_process_time = np.random.randint(lower_num, upper_num, jobs_num)

    #jobs_dict[str(10)] = 10
    #jobs_dict[str(11)] = 11

    #for i in range(30,40):
     #   jobs_dict[str(i)] = 2

    #for i in range(40,50):
     #   jobs_dict[str(i)] = 6

    #for i in range(50, 60):
     #   jobs_dict[str(i)] = 4


    jobs_dict = {}
    if CASE == 1:
        for i in range(1, 101):
            jobs_dict[str(i)] = i

    if CASE == 2:
        for i in range(10):
            jobs_dict[str(i)] = 6

        jobs_dict[str(10)] = 10

        jobs_dict[str(11)] = 11

        for i in range(12, 17):
            jobs_dict[str(i)] = 5

        for i in range(17, 20):
            jobs_dict[str(i)] = 12

    if CASE == 3:
        for i in range(30):
            jobs_dict[str(i)] = 3

        for i in range(30, 40):
            jobs_dict[str(i)] = 2

        for i in range(40, 50):
            jobs_dict[str(i)] = 6

        for i in range(50, 60):
            jobs_dict[str(i)] = 4

    if CASE == 4:
        for i in range(30):
            jobs_dict[str(i)] = 3

        for i in range(30, 40):
            jobs_dict[str(i)] = 2

        for i in range(40, 50):
            jobs_dict[str(i)] = 4

        for i in range(50, 60):
            jobs_dict[str(i)] = 6

    if CASE == 5:
        for i, item in enumerate([57, 58, 59, 60, 31, 32, 33, 34, 35, 36, 37, 29, 30, 61, 72, 73, 74, 75, 3, 98, 99, 100]):
            jobs_dict[str(i)] = item

    if CASE == 6:
        for i, item in enumerate([33, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]):
            jobs_dict[str(i)] = item

    if CASE == 7:
        for i, item in enumerate([3, 3, 3, 3, 2, 2, 4, 4]):
            jobs_dict[str(i)] = item

    if CASE == 8:
        for i, item in enumerate([3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4]):
            jobs_dict[str(i)] = item

    if CASE == 9:
        for i in range(1, 101):
            jobs_dict[str(i)] = i

    # Random input
    if CASE == 10:
        for i in range(jobs_num):
            jobs_dict[str(i)] = jobs_process_time[i]


    print("Drawing job times...")
    print("Job times: ", jobs_dict)
    print("average job time per machine:", sum(jobs_dict.values())/number_of_machines)
    # print("Number of machines: ", number_of_machines)

    return jobs_dict, number_of_machines

