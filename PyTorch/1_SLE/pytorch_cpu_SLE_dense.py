'''
Created On 04-11-2024
    By Nithin
'''

import torch
import time
import json

def prepare(size):
    return torch.randn(size, size), torch.randn(size, 1)

def compute(A,B):
    return torch.linalg.solve(A, B)

def main():

    # 0. Initialize Variables
    number_of_sizes = 50
    repeat = 10
    sizes = []
    for i in range(number_of_sizes):
        sizes.append(2**(i))
    # Initialise cut off size
    cutoff_index = 14  # < 10 sec - 14

    cutoff_size = sizes[cutoff_index]
    
    # Initialize preparation_times
    preparation_times = []
    # Initialize computation_times
    computation_times = []


    # 1. Initialise Random Dense matrices
    for size in sizes:
        if size > cutoff_size:
            preparation_times.append(0)
            computation_times.append(0)
        elif size < 16:
            preparation_times.append(0)
            computation_times.append(0)
        else:
            time1 = time.time()
            for i in range(repeat-1):
                A, B = prepare(size)
                A, B = 0, 0
            A, B = prepare(size)
            time2 = time.time()
            preparation_times.append((time2-time1)/repeat)
            time1 = time.time()
            for i in range(repeat):
                C = compute(A,B)
                C = 0
            time2 = time.time()
            computation_times.append((time2-time1)/repeat)
    
    # 2 .Print results
    print(f"Matrix Cutoff size: {cutoff_size} x {cutoff_size}, \n"
          f"Matrix Cutoff index: {cutoff_index}, \n"
          f"Time taken to prepare: {preparation_times[cutoff_index]:.6f}, \n"
          f"Time taken to compute: {computation_times[cutoff_index]:.6f}, \n"
          f"Total time: {(preparation_times[cutoff_index]+ computation_times[cutoff_index]):.6f} seconds.")
    
    # 3. Save results
    results = {
        'sizes': sizes,
        'cutoff_index': cutoff_index,
        'cutoff_sizes': cutoff_size,
        'preparation': preparation_times,
        'computation': computation_times
    }
    
    with open('pytorch_cpu_SLE_dense.json', 'w') as f:
        json.dump(results, f)
        
if __name__ == '__main__':
    main()