'''
Created On 04-11-2024
    By Nithin
'''

import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import numpy as np
import time
import json

def prepare(size):
    number_of_nodes_in_1_dimension = int(np.sqrt(size))
    total_number_of_nodes_in_2_dimensions = number_of_nodes_in_1_dimension**2

    # Create indices and values for sparse matrices
    indices_list = []
    values_list = []

    # Construct indices and values (same pattern as before)
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

    # Create indices and values for source matrix
    row_indices = [int(number_of_nodes_in_1_dimension / 2) + 
        number_of_nodes_in_1_dimension * int(number_of_nodes_in_1_dimension / 2)]
    col_indices = [0]
    source_values_list = [1]

    # Create the sparse matrix
    A = sparse.csr_matrix((values_list, np.array(indices_list).T), shape=(total_number_of_nodes_in_2_dimensions, total_number_of_nodes_in_2_dimensions))
    B = sparse.csr_matrix((source_values_list, (row_indices, col_indices)), shape=(total_number_of_nodes_in_2_dimensions, 1))
    
    return A, B

def compute(A,B):
    return spsolve(A, B)

def main():

    # 0. Initialize Variables
    number_of_sizes = 50
    repeat = 10
    sizes = []
    for i in range(number_of_sizes):
        sizes.append(2**(i))
    # Initialise cut off size
    cutoff_index = 18  # < 10 sec - 18

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
    
    with open('scipy_SLE_sparse.json', 'w') as f:
        json.dump(results, f)
        
if __name__ == '__main__':
    main()