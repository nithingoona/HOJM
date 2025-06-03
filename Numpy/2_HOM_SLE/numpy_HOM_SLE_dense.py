'''
Created On 04-11-2024
    By Nithin
'''

import numpy as np
import time
import json

def prepare(size):
    return np.random.randn(size, size), np.random.randn(size, 1)

def compute_while_training(A,B):
    I = np.eye(A.shape[0])
    A0 = A + 2*I
    D_flat = B*0
    for j in range(len(B)):
        A0[j,j] = 0
        D_flat[j] = A[j,j]

    Alpha = A0/D_flat
    Beta = B/D_flat
    Beta0 = Beta*1

    num_iter = int(A.shape[0]*np.log(A.shape[0]))
    C = B*0
    for i in range(num_iter):
        Beta = - np.matmul(Alpha-I,Beta)
        Alpha = - np.matmul(Alpha,Alpha)
        C = -np.matmul(Alpha,C) + Beta

    return C, Alpha, Beta0

def compute_after_training(Alpha,Beta0):
    C = Beta0*0
    num_iter = int(Alpha.shape[0]*np.log(Alpha.shape[0]))
    for i in range(num_iter+1):
        C = -np.matmul(Alpha,C) + Beta0
    return C

def main():

    # 0. Initialize Variables
    number_of_sizes = 50
    repeat = 10
    sizes = []
    for i in range(number_of_sizes):
        sizes.append(2**(i))
    # Initialise cut off size
    cutoff_index = 9  # < 10 sec - 9

    cutoff_size = sizes[cutoff_index]
    
    # Initialize preparation_times
    preparation_times = []
    # Initialize computation_times
    computation_times_while_training = []
    computation_times_after_training = []

    # 1. Initialise Random Dense matrices
    for size in sizes:
        if size > cutoff_size:
            preparation_times.append(0)
            computation_times_while_training.append(0)
            computation_times_after_training.append(0)
        elif size < 16:
            preparation_times.append(0)
            computation_times_while_training.append(0)
            computation_times_after_training.append(0)
        else:
            time1 = time.time()
            for i in range(repeat-1):
                A, B = prepare(size)
                A, B = 0, 0
            A, B = prepare(size)
            time2 = time.time()
            preparation_times.append((time2-time1)/repeat)
            time1 = time.time()
            for i in range(repeat-1):
                C, Alpha, Beta0 = compute_while_training(A,B)
                C, Alpha, Beta0 = 0, 0, 0
            C, Alpha, Beta0 = compute_while_training(A,B)
            time2 = time.time()
            computation_times_while_training.append((time2-time1)/repeat)
            time1 = time.time()
            for i in range(repeat):
                C = compute_after_training(Alpha, Beta0)
                C = 0
            time2 = time.time()
            computation_times_after_training.append((time2-time1)/repeat)

    # 2 .Print results
    print(f"Matrix Cutoff size: {cutoff_size} x {cutoff_size}, \n"
          f"Matrix Cutoff index: {cutoff_index}, \n"
          f"Time taken to prepare: {preparation_times[cutoff_index]:.6f}, \n"
          f"Time taken to compute while training: {computation_times_while_training[cutoff_index]:.6f}, \n"
          f"Total time while training: {(preparation_times[cutoff_index]+ computation_times_while_training[cutoff_index]):.6f} seconds. \n"
          f"Time taken to compute after training: {computation_times_after_training[cutoff_index]:.6f}, \n"
          f"Total time after training: {(preparation_times[cutoff_index]+ computation_times_after_training[cutoff_index]):.6f} seconds.")
    
    # 3. Save results
    results = {
        'sizes': sizes,
        'cutoff_index': cutoff_index,
        'cutoff_sizes': cutoff_size,
        'preparation': preparation_times,
        'computation_while': computation_times_while_training,
        'computation_after': computation_times_after_training
    }
    
    with open('numpy_HOM_SLE_dense.json', 'w') as f:
        json.dump(results, f)
        
if __name__ == '__main__':
    main()