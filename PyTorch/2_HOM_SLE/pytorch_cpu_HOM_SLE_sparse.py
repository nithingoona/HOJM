'''
Created On 04-11-2024
    By Nithin
    At Adbutvaahak
'''

import torch
import time
import json
import numpy as np

def prepare(size):
    number_of_nodes_in_1_dimension = int(np.sqrt(size))
    total_number_of_nodes_in_2_dimensions = number_of_nodes_in_1_dimension**2
    
    # Create indices and values for sparse matrices
    indices_list = []
    values_list = []
    
    # Construct indices and values (similar pattern as before)
    # Bottom and top boundaries
    for i in range(number_of_nodes_in_1_dimension):
        for j in [0, number_of_nodes_in_1_dimension-1]:
            node = i + j*number_of_nodes_in_1_dimension
            indices_list.append([node, node])
            values_list.append(1.0)
            
    # Left and right boundaries
    for j in range(number_of_nodes_in_1_dimension):
        for i in [0, number_of_nodes_in_1_dimension-1]:
            node = i + j*number_of_nodes_in_1_dimension
            indices_list.append([node, node])
            values_list.append(1.0)
            
    # Interior region
    for i in range(1, number_of_nodes_in_1_dimension-1):
        for j in range(1, number_of_nodes_in_1_dimension-1):
            node = i + j*number_of_nodes_in_1_dimension
            # Center node
            indices_list.append([node, node])
            values_list.append(1.0)
            # Adjacent nodes
            for offset, val in [(1, -0.25), (-1, -0.25), 
                            (number_of_nodes_in_1_dimension, -0.25),
                            (-number_of_nodes_in_1_dimension, -0.25)]:
                indices_list.append([node, node + offset])
                values_list.append(val)
                
    indices = torch.tensor(indices_list, dtype=torch.long).t()
    values = torch.tensor(values_list, dtype=torch.float)
    
    source_indices = torch.tensor([int(number_of_nodes_in_1_dimension / 2) + 
               number_of_nodes_in_1_dimension * int(number_of_nodes_in_1_dimension / 2),0], dtype=torch.long).t()
    source_values_list = [1]
    source_values = torch.tensor(source_values_list, dtype=torch.float)

    # Create sparse tensors
    A = torch.sparse_coo_tensor(indices, values, 
                            (total_number_of_nodes_in_2_dimensions, 
                                total_number_of_nodes_in_2_dimensions)).coalesce() 
    # B = torch.sparse_coo_tensor(source_indices, source_values, 
    #                         (total_number_of_nodes_in_2_dimensions, 
    #                             1)).coalesce() 
    B = torch.randn(total_number_of_nodes_in_2_dimensions, 1)
    A = A.to_dense()
    return A, B


def compute_while_training(A,B):
    I = torch.eye(A.shape[0])
    A0 = A + 2*I
    D_flat = A0.diagonal()
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            A0[i,j] = A0[i,j]/D_flat[i]
    Alpha = A0*1
    Beta = B*0
    for i in range(A.shape[0]):
        Beta[i] = B[i]/D_flat[i]
    Beta0 = Beta*1

    num_iter = int(A.shape[0]*np.log(A.shape[0]))
    C = B*0
    for i in range(num_iter):
        Beta = - torch.matmul(Alpha-I,Beta)
        Alpha = - torch.matmul(Alpha,Alpha)
        C = - torch.matmul(Alpha,C) + Beta

    return C, Alpha, Beta0

def compute_after_training(Alpha,Beta0):
    C = Beta0*0
    num_iter = int(Alpha.shape[0]*np.log(Alpha.shape[0]))
    for i in range(num_iter+1):
        C = -torch.matmul(Alpha,C) + Beta0
    return C

def main():

    # 0. Initialize Variables
    number_of_sizes = 50
    repeat = 10
    sizes = []
    for i in range(number_of_sizes):
        sizes.append(2**(i))
    # Initialise cut off size
    cutoff_index = 10  # < 10 sec - 10

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
    
    with open('pytorch_cpu_HOM_SLE_sparse.json', 'w') as f:
        json.dump(results, f)
        
if __name__ == '__main__':
    main()