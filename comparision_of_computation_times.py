"""
Comprehensive Performance Comparison of Higher Order Jacobi Method (HOJM)

This module benchmarks the Higher Order Jacobi Method against traditional linear algebra
libraries for both linear system solving and matrix inversion operations. The benchmarks
test performance across different matrix sizes, precision levels, and hardware configurations.

The test matrices are generated using a 2D finite difference discretization of Poisson's
equation, creating sparse, diagonally dominant systems that are well-suited for iterative
methods like HOJM.

Author: Nithin Kumar Goona, Ph.D.
Affiliation: The University of Texas at El Paso (UTEP)
Created on: July 29, 2025

Description:
    Benchmarks HOJM performance against:
    - NumPy dense linear algebra
    - SciPy sparse linear algebra  
    - PyTorch CPU and GPU implementations
    
    Tests both:
    - Linear system solving (Ax = b)
    - Matrix inversion (A^-1)
    
    Generates timing data and saves results for visualization.

License:
    MIT

Contact:
    nkgoona@utep.edu, nithin.goona@gmail.com
"""

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse import linalg as spla
import torch
import time
import matplotlib.pyplot as plt
import pickle
import os
import sys

def fill_A(A, nx, ny):
    """
    Fill matrix A with 2D finite difference discretization of Poisson's equation.
    
    Creates a sparse, diagonally dominant matrix representing the discrete Laplacian
    operator on a 2D rectangular grid with Dirichlet boundary conditions. The resulting
    matrix has the structure:
    - Interior points: 5-point stencil with diagonal value 4 and off-diagonal values -1
    - Boundary points: Identity (Dirichlet boundary conditions)
    
    Args:
        A: Matrix to fill (can be NumPy array, SciPy sparse matrix, or PyTorch tensor)
        nx (int): Number of grid points in x-direction
        ny (int): Number of grid points in y-direction
        
    Returns:
        A: The filled matrix with finite difference stencil
        
    Note:
        Total matrix size is (nx*ny) x (nx*ny). The grid is ordered row-wise,
        so point (i,j) corresponds to index i + j*nx.
    """
    # Fill interior points with 5-point finite difference stencil
    # This creates a diagonally dominant system suitable for iterative methods
    for i in range(1, nx-1):  # Interior x-points
        for j in range(1, ny-1):  # Interior y-points
            nn = i + j*nx  # Convert 2D index to 1D
            A[nn, nn] = 4.0        # Center point coefficient
            A[nn, nn - 1] = -1.0   # Left neighbor
            A[nn, nn + 1] = -1.0   # Right neighbor  
            A[nn, nn - nx] = -1.0  # Bottom neighbor
            A[nn, nn + nx] = -1.0  # Top neighbor

    # Apply Dirichlet boundary conditions (u = 0 on boundary)
    # Left and right boundaries
    for j in range(ny):
        A[j * nx, j * nx] = 1                           # Left boundary
        A[(nx - 1) + j * nx, (nx - 1) + j * nx] = 1    # Right boundary
        
    # Bottom and top boundaries  
    for i in range(nx):
        A[i, i] = 1                                     # Bottom boundary
        A[i + (ny - 1) * nx, i + (ny - 1) * nx] = 1   # Top boundary

    return A

def hojm_train(A, b, num_iter):
    """
    Train the Higher Order Jacobi Method by generating coefficient matrices.
    
    This function implements the training phase of HOJM, which pre-computes
    higher-order coefficient matrices that can be reused for multiple solves
    with the same system matrix A but different right-hand sides b.
    
    The method decomposes A = D + A0 where D is diagonal and A0 is off-diagonal,
    then iteratively computes higher-order coefficients using the recurrence:
    - α₀ = -A₀/D
    - β₀ = b/D  
    - αₖ₊₁ = αₖ²
    - βₖ₊₁ = (αₖ + I)βₖ
    
    Args:
        A (torch.Tensor): System matrix (must be diagonally dominant)
        b (torch.Tensor): Right-hand side vector
        num_iter (int): Number of iterations (higher-order terms to compute)
        
    Returns:
        tuple: (Alphas, x) where:
            - Alphas (list): List of coefficient matrices [α₀, α₁, ..., αₙ]
            - x (torch.Tensor): Solution vector obtained during training
            
    Note:
        The system matrix A must be diagonally dominant for convergence.
        More iterations generally improve accuracy but increase computational cost.
    """
    # Create identity matrix matching A's properties
    I = torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
    
    # Extract diagonal elements and reshape for broadcasting
    D_flat = torch.diag(A).unsqueeze(1)  # Diagonal of A as column vector
    
    # Create off-diagonal part by zeroing the diagonal
    A0 = A - torch.diag_embed(torch.diag(A))  # A₀ = A - D
    
    # Initialize first-order coefficients
    Alpha = -A0/D_flat  # α₀ = -A₀/D (Jacobi iteration matrix)
    Beta = b/D_flat     # β₀ = b/D (scaled RHS)

    # Initialize solution vector and coefficient storage
    x = b.clone()       # Starting guess for solution
    Alphas = [Alpha]    # Store all coefficient matrices
    
    # Generate higher-order coefficients iteratively
    for _ in range(num_iter):
        # Update scaled RHS: βₖ₊₁ = (αₖ + I)βₖ
        Beta = torch.matmul(Alpha + I, Beta)
        
        # Update coefficient matrix: αₖ₊₁ = αₖ²
        Alpha = torch.matmul(Alpha, Alpha)
        Alphas.append(Alpha)
        
        # Update solution estimate: xₖ₊₁ = αₖ₊₁xₖ + βₖ₊₁
        x = torch.matmul(Alpha, x) + Beta

    return Alphas, x

def hojm_solve(Alphas, A, b):
    """
    Solve linear system using pre-computed HOJM coefficients.
    
    This function implements the inference phase of HOJM, using pre-computed
    coefficient matrices to solve Ax = b efficiently. This is particularly
    advantageous when solving multiple systems with the same matrix A but
    different right-hand sides b.
    
    The solution is computed using the higher-order coefficients:
    x = Σᵢ αᵢ₊₁ * (∏ⱼ₌₀ⁱ (αⱼ + I)) * (b/D)
    
    Args:
        Alphas (list): Pre-computed coefficient matrices from hojm_train()
        A (torch.Tensor): Original system matrix (used for diagonal extraction)
        b (torch.Tensor): Right-hand side vector
        
    Returns:
        torch.Tensor: Solution vector x such that Ax ≈ b
        
    Note:
        This function assumes the coefficients were computed for the same
        matrix A. The accuracy depends on the number of iterations used
        during training.
    """
    # Create identity matrix matching A's properties
    I = torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
    
    # Extract diagonal and prepare scaled RHS
    D_flat = A.diagonal()
    Beta = b / D_flat.unsqueeze(1)  # β₀ = b/D

    # Initialize solution vector
    x = b.clone()
    
    # Apply higher-order corrections iteratively
    for i in range(len(Alphas)-1):
        # Update scaled RHS: βᵢ₊₁ = (αᵢ + I)βᵢ
        Beta = torch.matmul(Alphas[i] + I, Beta)
        
        # Update solution: x = αᵢ₊₁x + βᵢ₊₁
        x = torch.matmul(Alphas[i+1], x) + Beta

    return x

def find_inverse_with_coeffecients(Alphas, A):
    """
    Compute matrix inverse using pre-computed HOJM coefficients.
    
    This function computes the inverse of matrix A using the higher-order
    coefficient matrices generated by hojm_train(). The inverse is computed
    using the formula:
    
    A⁻¹ = D⁻¹ * ∏ᵢ₌₀ⁿ (αᵢ + I)
    
    where D is the diagonal of A and αᵢ are the coefficient matrices.
    
    Args:
        Alphas (list): Pre-computed coefficient matrices from hojm_train()
        A (torch.Tensor): Original system matrix (used for diagonal extraction)
        
    Returns:
        torch.Tensor: Approximate inverse matrix A⁻¹
        
    Note:
        The accuracy of the inverse depends on the number of iterations used
        during training. More iterations generally yield better approximations
        but at higher computational cost.
    """
    # Create identity matrix matching A's properties
    I = torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
    
    # Initialize with first coefficient plus identity
    inv_A = Alphas[0] + I  # (α₀ + I)
    
    # Extract diagonal for final scaling
    D_flat = torch.diag(A).unsqueeze(1)  # Diagonal of A as column vector

    # Multiply by remaining coefficient matrices
    for i in range(1, len(Alphas)):
        # inv_A = inv_A * (αᵢ + I)
        inv_A = torch.matmul(inv_A, Alphas[i] + I)

    # Apply diagonal scaling: A⁻¹ = D⁻¹ * ∏(αᵢ + I)
    # Use transpose operations for efficient broadcasting
    inv_A = (inv_A.t() / D_flat).t()

    return inv_A

def main():
    """
    Main benchmarking function that compares HOJM against traditional linear algebra methods.
    
    This function performs comprehensive performance benchmarks by:
    1. Testing linear system solving (Ax = b) across multiple methods and matrix sizes
    2. Testing matrix inversion (A^-1) across multiple methods and matrix sizes
    3. Measuring execution times until a time limit is reached
    4. Saving results for later visualization
    
    The benchmarks test matrices of increasing size (powers of 2) until the time limit
    is exceeded, providing performance scaling data across different problem sizes.
    
    Tested Methods:
    - NumPy dense linear algebra
    - SciPy sparse linear algebra
    - PyTorch CPU linear algebra
    - PyTorch GPU linear algebra  
    - HOJM CPU (training + inference phases)
    - HOJM GPU (training + inference phases)
    """
    
    # ============================================================================
    # 0. BENCHMARK CONFIGURATION AND SETUP
    # ============================================================================
    
    # Benchmark parameters
    precision = "32"        # Floating point precision ("32" or "64")
    time_limit = 1          # Maximum time per method (seconds)
    order = 4               # Starting matrix size order (2^order x 2^order grid)
    
    # Data storage for all timing results
    times = {}
    
    # Setup output directories and files
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)
    
    # Build filename with precision identifier
    filename = f"solve_times_precision_{precision}.pkl"
    save_path = os.path.join(save_dir, filename)
    
    # Redirect all print output to log file for later analysis
    log_path = os.path.join(save_dir, f"log_{precision}.txt")
    sys.stdout = open(log_path, 'w')

    # ============================================================================
    # 1. LINEAR SYSTEM SOLVING BENCHMARKS (Ax = b)
    # ============================================================================
    
    print("Starting linear system solving benchmarks...")
    solve_times = {}  # Storage for all solve timing results

    # ============================================================================
    # 1.1 NumPy Dense Linear Algebra Benchmark
    # ============================================================================
    
    print("Benchmarking NumPy dense linalg.solve()...")
    numpy_times = []        # Timing results for each matrix size
    smps = []              # Solution values at midpoint for verification
    within_limit = True    # Continue until time limit exceeded
    
    it = 0  # Iteration counter for matrix size scaling
    while within_limit:
        try:
            # Generate test problem of increasing size
            ny = 32                    # Fixed y-dimension
            current_order = order + it # Current size order
            nx = 2**current_order     # x-dimension grows as power of 2
            nt = nx * ny              # Total matrix size
            
            # Set floating point precision
            if precision == "32":
                dtype = np.float32
            elif precision == "64":
                dtype = np.float64
            else:
                raise ValueError("Select 32 or 64 bit precision")
            
            # Create test matrices
            A = np.zeros((nt, nt), dtype=dtype)  # System matrix
            b = np.zeros((nt, 1), dtype=dtype)   # Right-hand side
            
            # Fill with finite difference discretization
            A = fill_A(A, nx, ny)
            
            # Set point source at center of domain
            b[int(nt/2), 0] = 1
            
            # Time the solve operation
            t1 = time.time()
            x = np.linalg.solve(A, b)
            t2 = time.time()
            
            # Store solution at midpoint for verification
            smps.append(x[int(nx/2) + nx*int(ny/2)])
            
            # Record timing and check if time limit exceeded
            numpy_times.append(t2 - t1)
            if t2 - t1 > time_limit:
                within_limit = False
                
            it += 1
            
        except Exception as e:
            print(f"NumPy exception at order={current_order}: {e}")
            within_limit = False
    
    # Log results
    print("NumPy dense solve results:")
    print(f"Solution at mid points: {smps}")
    print(f"Execution times: {numpy_times}")
    print("")
    solve_times['numpy'] = numpy_times


    # ============================================================================
    # 1.2 SciPy Sparse Linear Algebra Benchmark
    # ============================================================================
    
    print("Benchmarking SciPy sparse linalg.spsolve()...")
    scipy_times = []        # Timing results for each matrix size
    smps = []              # Solution values at midpoint for verification
    within_limit = True    # Continue until time limit exceeded

    it = 0  # Iteration counter for matrix size scaling
    while within_limit:
        try:
            # Generate test problem of increasing size
            ny = 32                    # Fixed y-dimension
            current_order = order + it # Current size order
            nx = 2**current_order     # x-dimension grows as power of 2
            nt = nx * ny              # Total matrix size

            # Set floating point precision
            if precision == "32":
                dtype = np.float32
            elif precision == "64":
                dtype = np.float64
            else:
                raise ValueError("Select 32 or 64 bit precision")

            # Create sparse test matrices (LIL format for efficient construction)
            A = lil_matrix((nt, nt), dtype=dtype)  # Sparse system matrix
            b = np.zeros((nt, 1), dtype=dtype)     # Dense right-hand side

            # Fill with finite difference discretization
            A = fill_A(A, nx, ny)
            
            # Set point source at center of domain
            b[int(nt / 2), 0] = 1

            # Time the sparse solve operation (convert to CSR for efficiency)
            t1 = time.time()
            x = spla.spsolve(A.tocsr(), b)  # CSR format optimal for solving
            t2 = time.time()

            # Store solution at midpoint for verification
            smps.append(x[int(nx/2) + nx*int(ny/2)])

            # Record timing and check if time limit exceeded
            scipy_times.append(t2 - t1)
            if t2 - t1 > time_limit:
                within_limit = False

            it += 1

        except Exception as e:
            print(f"SciPy exception at order={current_order}: {e}")
            within_limit = False

    # Log results
    print("SciPy sparse solve results:")
    print(f"Solution at mid points: {smps}")
    print(f"Execution times: {scipy_times}")
    print("")
    solve_times['scipy'] = scipy_times


    # ============================================================================
    # 1.3 PyTorch CPU Linear Algebra Benchmark
    # ============================================================================
    
    print("Benchmarking PyTorch CPU linalg.solve()...")
    torch_linalg_cpu_times = []  # Timing results for each matrix size
    smps = []                   # Solution values at midpoint for verification
    within_limit = True         # Continue until time limit exceeded

    it = 0  # Iteration counter for matrix size scaling
    while within_limit:
        try:
            # Generate test problem of increasing size
            ny = 32                    # Fixed y-dimension
            current_order = order + it # Current size order
            nx = 2**current_order     # x-dimension grows as power of 2
            nt = nx * ny              # Total matrix size

            # Set floating point precision
            if precision == "32":
                dtype = torch.float32
            elif precision == "64":
                dtype = torch.float64
            else:
                raise ValueError("Select 32 or 64 bit precision")

            # Create PyTorch tensors on CPU
            A = torch.zeros(nt, nt, dtype=dtype, device='cpu')  # System matrix
            b = torch.zeros(nt, 1, dtype=dtype, device='cpu')   # Right-hand side

            # Fill with finite difference discretization
            A = fill_A(A, nx, ny)
            
            # Set point source at center of domain
            b[int(nt / 2), 0] = 1

            # Time the PyTorch CPU solve operation
            t1 = time.time()
            x = torch.linalg.solve(A, b)
            t2 = time.time()

            # Store solution at midpoint for verification
            smps.append(x[int(nx/2) + nx*int(ny/2)])

            # Record timing and check if time limit exceeded
            torch_linalg_cpu_times.append(t2 - t1)
            if t2 - t1 > time_limit:
                within_limit = False

            it += 1

        except Exception as e:
            print(f"PyTorch CPU exception at order={current_order}: {e}")
            within_limit = False

    # Log results
    print("PyTorch CPU solve results:")
    print(f"Solution at mid points: {smps}")
    print(f"Execution times: {torch_linalg_cpu_times}")
    print("")
    solve_times['torch_linalg_cpu'] = torch_linalg_cpu_times


    # ============================================================================
    # 1.4 PyTorch GPU Linear Algebra Benchmark
    # ============================================================================
    
    print("Benchmarking PyTorch GPU linalg.solve()...")
    torch_linalg_gpu_times = []  # Timing results for each matrix size
    smps = []                   # Solution values at midpoint for verification
    within_limit = True         # Continue until time limit exceeded

    it = 0  # Iteration counter for matrix size scaling
    while within_limit:
        try:
            # Generate test problem of increasing size
            ny = 32                    # Fixed y-dimension
            current_order = order + it # Current size order
            nx = 2**current_order     # x-dimension grows as power of 2
            nt = nx * ny              # Total matrix size

            # Set floating point precision
            if precision == "32":
                dtype = torch.float32
            elif precision == "64":
                dtype = torch.float64
            else:
                raise ValueError("Select 32 or 64 bit precision")

            # Create PyTorch tensors on GPU (CUDA)
            A = torch.zeros(nt, nt, dtype=dtype, device='cuda')  # System matrix
            b = torch.zeros(nt, 1, dtype=dtype, device='cuda')   # Right-hand side

            # Fill with finite difference discretization
            A = fill_A(A, nx, ny)
            
            # Set point source at center of domain
            b[int(nt / 2), 0] = 1

            # Time the PyTorch GPU solve operation
            t1 = time.time()
            x = torch.linalg.solve(A, b)
            t2 = time.time()

            # Store solution at midpoint for verification
            smps.append(x[int(nx/2) + nx*int(ny/2)])

            # Record timing and check if time limit exceeded
            torch_linalg_gpu_times.append(t2 - t1)
            if t2 - t1 > time_limit:
                within_limit = False

            it += 1

        except Exception as e:
            print(f"PyTorch GPU exception at order={current_order}: {e}")
            within_limit = False

    # Log results
    print("PyTorch GPU solve results:")
    print(f"Solution at mid points: {smps}")
    print(f"Execution times: {torch_linalg_gpu_times}")
    print("")
    solve_times['torch_linalg_gpu'] = torch_linalg_gpu_times


    # ============================================================================
    # 1.5 Higher Order Jacobi Method CPU Benchmark
    # ============================================================================
    
    print("Benchmarking HOJM CPU (training + inference phases)...")
    torch_hojm_cpu_while_times = []  # Training phase timing results
    torch_hojm_cpu_after_times = []  # Inference phase timing results
    smps = []                       # Solution values at midpoint for verification
    within_limit = True             # Continue until time limit exceeded

    it = 0  # Iteration counter for matrix size scaling
    while within_limit:
        try:
            # Generate test problem of increasing size
            ny = 32                    # Fixed y-dimension
            current_order = order + it # Current size order
            nx = 2**current_order     # x-dimension grows as power of 2
            nt = nx * ny              # Total matrix size

            # Set floating point precision
            if precision == "32":
                dtype = torch.float32
            elif precision == "64":
                dtype = torch.float64
            else:
                raise ValueError("Select 32 or 64 bit precision")

            # Create PyTorch tensors on CPU
            A = torch.zeros(nt, nt, dtype=dtype, device='cpu')  # System matrix
            b = torch.zeros(nt, 1, dtype=dtype, device='cpu')   # Right-hand side

            # Fill with finite difference discretization
            A = fill_A(A, nx, ny)
            
            # Set point source at center of domain
            b[int(nt / 2), 0] = 1

            # Time the HOJM training phase (coefficient generation + solve)
            t1 = time.time()
            model, x = hojm_train(A, b, 20)  # 20 iterations for higher-order terms
            t2 = time.time()

            # Store solution at midpoint for verification
            smps.append(x[int(nx/2) + nx*int(ny/2)])

            # Record training time
            while_time = t2 - t1
            torch_hojm_cpu_while_times.append(while_time)

            # Time the HOJM inference phase (solve with pre-computed coefficients)
            t1 = time.time()
            _ = hojm_solve(model, A, b)
            t2 = time.time()

            # Record inference time
            torch_hojm_cpu_after_times.append(t2 - t1)

            # Check if training time exceeded limit (training is the bottleneck)
            if while_time > time_limit:
                within_limit = False

            it += 1

        except Exception as e:
            print(f"HOJM CPU exception at order={current_order}: {e}")
            within_limit = False

    # Log results for both phases
    print("HOJM CPU training phase results:")
    print(f"Solution at mid points: {smps}")
    print(f"Training times: {torch_hojm_cpu_while_times}")
    print("")
    solve_times['torch_hojm_cpu_while'] = torch_hojm_cpu_while_times

    print("HOJM CPU inference phase results:")
    print(f"Inference times: {torch_hojm_cpu_after_times}")
    print("")
    solve_times['torch_hojm_cpu_after'] = torch_hojm_cpu_after_times


    # ============================================================================
    # 1.6 Higher Order Jacobi Method GPU Benchmark
    # ============================================================================
    
    print("Benchmarking HOJM GPU (training + inference phases)...")
    torch_hojm_gpu_while_times = []  # Training phase timing results
    torch_hojm_gpu_after_times = []  # Inference phase timing results
    smps = []                       # Solution values at midpoint for verification
    within_limit = True             # Continue until time limit exceeded

    it = 0  # Iteration counter for matrix size scaling
    while within_limit:
        try:
            # Generate test problem of increasing size
            ny = 32                    # Fixed y-dimension
            current_order = order + it # Current size order
            nx = 2**current_order     # x-dimension grows as power of 2
            nt = nx * ny              # Total matrix size

            # Set floating point precision
            if precision == "32":
                dtype = torch.float32
            elif precision == "64":
                dtype = torch.float64
            else:
                raise ValueError("Select 32 or 64 bit precision")

            # Create PyTorch tensors on GPU (CUDA)
            A = torch.zeros(nt, nt, dtype=dtype, device='cuda')  # System matrix
            b = torch.zeros(nt, 1, dtype=dtype, device='cuda')   # Right-hand side

            # Fill with finite difference discretization
            A = fill_A(A, nx, ny)
            
            # Set point source at center of domain
            b[int(nt / 2), 0] = 1

            # Time the HOJM training phase (coefficient generation + solve)
            t1 = time.time()
            model, x = hojm_train(A, b, 20)  # 20 iterations for higher-order terms
            t2 = time.time()

            # Store solution at midpoint for verification
            smps.append(x[int(nx/2) + nx*int(ny/2)])

            # Record training time
            while_time = t2 - t1
            torch_hojm_gpu_while_times.append(while_time)

            # Time the HOJM inference phase (solve with pre-computed coefficients)
            t1 = time.time()
            _ = hojm_solve(model, A, b)
            t2 = time.time()

            # Record inference time
            torch_hojm_gpu_after_times.append(t2 - t1)

            # Check if training time exceeded limit (training is the bottleneck)
            if while_time > time_limit:
                within_limit = False

            it += 1

        except Exception as e:
            print(f"HOJM GPU exception at order={current_order}: {e}")
            within_limit = False

    # Log results for both phases
    print("HOJM GPU training phase results:")
    print(f"Solution at mid points: {smps}")
    print(f"Training times: {torch_hojm_gpu_while_times}")
    print("")
    solve_times['torch_hojm_gpu_while'] = torch_hojm_gpu_while_times

    print("HOJM GPU inference phase results:")
    print(f"Inference times: {torch_hojm_gpu_after_times}")
    print("")
    solve_times['torch_hojm_gpu_after'] = torch_hojm_gpu_after_times

    # Store all solve timing results
    times["solve"] = solve_times



    # ============================================================================
    # 2. MATRIX INVERSION BENCHMARKS (A^-1)
    # ============================================================================
    
    print("Starting matrix inversion benchmarks...")
    inverse_times = {}  # Storage for all inversion timing results

    # ============================================================================
    # 2.1 NumPy Dense Matrix Inversion Benchmark
    # ============================================================================
    
    print("Benchmarking NumPy dense linalg.inv()...")
    numpy_times = []        # Timing results for each matrix size
    within_limit = True     # Continue until time limit exceeded

    it = 0  # Iteration counter for matrix size scaling
    while within_limit:
        try:
            # Generate test problem of increasing size
            ny = 32                    # Fixed y-dimension
            current_order = order + it # Current size order
            nx = 2**current_order     # x-dimension grows as power of 2
            nt = nx * ny              # Total matrix size

            # Set floating point precision
            if precision == "32":
                dtype = np.float32
            elif precision == "64":
                dtype = np.float64
            else:
                raise ValueError("Select 32 or 64 bit precision")
            
            # Create test matrix
            A = np.zeros((nt, nt), dtype=dtype)  # System matrix
            A = fill_A(A, nx, ny)

            # Time the matrix inversion operation
            t1 = time.time()
            _ = np.linalg.inv(A)
            t2 = time.time()

            # Record timing and check if time limit exceeded
            numpy_times.append(t2 - t1)
            if t2 - t1 > time_limit:
                within_limit = False

            it += 1

        except Exception as e:
            print(f"NumPy inversion exception at order={current_order}: {e}")
            within_limit = False

    # Log results
    print("NumPy dense inversion results:")
    print(f"Execution times: {numpy_times}")
    print("")
    inverse_times['numpy'] = numpy_times


    # ============================================================================
    # 2.2 SciPy Sparse Matrix Inversion Benchmark
    # ============================================================================
    
    print("Benchmarking SciPy sparse linalg.inv()...")
    scipy_times = []        # Timing results for each matrix size
    within_limit = True     # Continue until time limit exceeded

    it = 0  # Iteration counter for matrix size scaling
    while within_limit:
        try:
            # Generate test problem of increasing size
            ny = 32                    # Fixed y-dimension
            current_order = order + it # Current size order
            nx = 2**current_order     # x-dimension grows as power of 2
            nt = nx * ny              # Total matrix size

            # Set floating point precision
            if precision == "32":
                dtype = np.float32
            elif precision == "64":
                dtype = np.float64
            else:
                raise ValueError("Select 32 or 64 bit precision")

            # Create sparse test matrix (LIL format for efficient construction)
            A = lil_matrix((nt, nt), dtype=dtype)  # Sparse system matrix
            A = fill_A(A, nx, ny)

            # Time the sparse matrix inversion operation
            # Convert to CSC format for efficient inversion, then back to dense
            t1 = time.time()
            _ = spla.inv(A.tocsc()).toarray()  # CSC format optimal for inversion
            t2 = time.time()

            # Record timing and check if time limit exceeded
            scipy_times.append(t2 - t1)
            if t2 - t1 > time_limit:
                within_limit = False

            it += 1

        except Exception as e:
            print(f"SciPy inversion exception at order={current_order}: {e}")
            within_limit = False

    # Log results
    print("SciPy sparse inversion results:")
    print(f"Execution times: {scipy_times}")
    print("")
    inverse_times['scipy'] = scipy_times


    # ============================================================================
    # 2.3 PyTorch CPU Matrix Inversion Benchmark
    # ============================================================================
    
    print("Benchmarking PyTorch CPU linalg.inv()...")
    torch_linalg_cpu_times = []  # Timing results for each matrix size
    within_limit = True          # Continue until time limit exceeded

    it = 0  # Iteration counter for matrix size scaling
    while within_limit:
        try:
            # Generate test problem of increasing size
            ny = 32                    # Fixed y-dimension
            current_order = order + it # Current size order
            nx = 2**current_order     # x-dimension grows as power of 2
            nt = nx * ny              # Total matrix size

            # Set floating point precision
            if precision == "32":
                dtype = torch.float32
            elif precision == "64":
                dtype = torch.float64
            else:
                raise ValueError("Select 32 or 64 bit precision")

            # Create PyTorch tensor on CPU
            A = torch.zeros(nt, nt, dtype=dtype, device='cpu')  # System matrix
            A = fill_A(A, nx, ny)

            # Time the PyTorch CPU matrix inversion operation
            t1 = time.time()
            _ = torch.linalg.inv(A)
            t2 = time.time()

            # Record timing and check if time limit exceeded
            torch_linalg_cpu_times.append(t2 - t1)
            if t2 - t1 > time_limit:
                within_limit = False

            it += 1

        except Exception as e:
            print(f"PyTorch CPU inversion exception at order={current_order}: {e}")
            within_limit = False

    # Log results
    print("PyTorch CPU inversion results:")
    print(f"Execution times: {torch_linalg_cpu_times}")
    print("")
    inverse_times['torch_linalg_cpu'] = torch_linalg_cpu_times


    # ============================================================================
    # 2.4 PyTorch GPU Matrix Inversion Benchmark
    # ============================================================================
    
    print("Benchmarking PyTorch GPU linalg.inv()...")
    torch_linalg_gpu_times = []  # Timing results for each matrix size
    within_limit = True          # Continue until time limit exceeded

    it = 0  # Iteration counter for matrix size scaling
    while within_limit:
        try:
            # Generate test problem of increasing size
            ny = 32                    # Fixed y-dimension
            current_order = order + it # Current size order
            nx = 2**current_order     # x-dimension grows as power of 2
            nt = nx * ny              # Total matrix size

            # Set floating point precision
            if precision == "32":
                dtype = torch.float32
            elif precision == "64":
                dtype = torch.float64
            else:
                raise ValueError("Select 32 or 64 bit precision")

            # Create PyTorch tensor on GPU (CUDA)
            A = torch.zeros(nt, nt, dtype=dtype, device='cuda')  # System matrix
            A = fill_A(A, nx, ny)

            # Time the PyTorch GPU matrix inversion operation
            t1 = time.time()
            _ = torch.linalg.inv(A)
            t2 = time.time()

            # Record timing and check if time limit exceeded
            torch_linalg_gpu_times.append(t2 - t1)
            if t2 - t1 > time_limit:
                within_limit = False

            it += 1

        except Exception as e:
            print(f"PyTorch GPU inversion exception at order={current_order}: {e}")
            within_limit = False

    # Log results
    print("PyTorch GPU inversion results:")
    print(f"Execution times: {torch_linalg_gpu_times}")
    print("")
    inverse_times['torch_linalg_gpu'] = torch_linalg_gpu_times


    # ============================================================================
    # 2.5 Higher Order Jacobi Method CPU Matrix Inversion Benchmark
    # ============================================================================
    
    print("Benchmarking HOJM CPU matrix inversion...")
    torch_hojm_cpu_times = []  # Timing results for each matrix size
    within_limit = True        # Continue until time limit exceeded

    it = 0  # Iteration counter for matrix size scaling
    while within_limit:
        try:
            # Generate test problem of increasing size
            ny = 32                    # Fixed y-dimension
            current_order = order + it # Current size order
            nx = 2**current_order     # x-dimension grows as power of 2
            nt = nx * ny              # Total matrix size

            # Set floating point precision
            if precision == "32":
                dtype = torch.float32
            elif precision == "64":
                dtype = torch.float64
            else:
                raise ValueError("Select 32 or 64 bit precision")

            # Create PyTorch tensors on CPU
            A = torch.zeros(nt, nt, dtype=dtype, device='cpu')  # System matrix
            b = torch.zeros(nt, 1, dtype=dtype, device='cpu')   # Dummy RHS for training

            # Fill with finite difference discretization
            A = fill_A(A, nx, ny)
            
            # Set dummy point source (needed for HOJM training)
            b[int(nt / 2), 0] = 1

            # Time the complete HOJM inversion process
            # This includes both training and inversion computation
            t1 = time.time()
            model, _ = hojm_train(A, b, 20)  # Train with 20 iterations
            _ = find_inverse_with_coeffecients(model, A)  # Compute inverse
            t2 = time.time()

            # Record timing and check if time limit exceeded
            torch_hojm_cpu_times.append(t2 - t1)
            if t2 - t1 > time_limit:
                within_limit = False

            it += 1

        except Exception as e:
            print(f"HOJM CPU inversion exception at order={current_order}: {e}")
            within_limit = False

    # Log results
    print("HOJM CPU inversion results:")
    print(f"Execution times: {torch_hojm_cpu_times}")
    print("")
    inverse_times['torch_hojm_cpu'] = torch_hojm_cpu_times


    # ============================================================================
    # 2.6 Higher Order Jacobi Method GPU Matrix Inversion Benchmark
    # ============================================================================
    
    print("Benchmarking HOJM GPU matrix inversion...")
    torch_hojm_gpu_times = []  # Timing results for each matrix size
    within_limit = True        # Continue until time limit exceeded

    it = 0  # Iteration counter for matrix size scaling
    while within_limit:
        try:
            # Generate test problem of increasing size
            ny = 32                    # Fixed y-dimension
            current_order = order + it # Current size order
            nx = 2**current_order     # x-dimension grows as power of 2
            nt = nx * ny              # Total matrix size

            # Set floating point precision
            if precision == "32":
                dtype = torch.float32
            elif precision == "64":
                dtype = torch.float64
            else:
                raise ValueError("Select 32 or 64 bit precision")

            # Create PyTorch tensors on GPU (CUDA)
            A = torch.zeros(nt, nt, dtype=dtype, device='cuda')  # System matrix
            b = torch.zeros(nt, 1, dtype=dtype, device='cuda')   # Dummy RHS for training

            # Fill with finite difference discretization
            A = fill_A(A, nx, ny)
            
            # Set dummy point source (needed for HOJM training)
            b[int(nt / 2), 0] = 1

            # Time the complete HOJM inversion process on GPU
            # This includes both training and inversion computation
            t1 = time.time()
            model, _ = hojm_train(A, b, 20)  # Train with 20 iterations
            _ = find_inverse_with_coeffecients(model, A)  # Compute inverse
            t2 = time.time()

            # Record timing and check if time limit exceeded
            torch_hojm_gpu_times.append(t2 - t1)
            if t2 - t1 > time_limit:
                within_limit = False

            it += 1

        except Exception as e:
            print(f"HOJM GPU inversion exception at order={current_order}: {e}")
            within_limit = False

    # Log results
    print("HOJM GPU inversion results:")
    print(f"Execution times: {torch_hojm_gpu_times}")
    print("")
    inverse_times['torch_hojm_gpu'] = torch_hojm_gpu_times

    # Store all inversion timing results
    times["inverse"] = inverse_times


    # ============================================================================
    # 3. SAVE RESULTS AND CLEANUP
    # ============================================================================
    
    # Store all inversion timing results
    times["inverse"] = inverse_times

    # Save the complete timing results to pickle file for later analysis
    print("Saving benchmark results...")
    with open(save_path, "wb") as f:
        pickle.dump(times, f)

    print(f"Benchmark completed successfully!")
    print(f"Timing results saved to: {save_path}")
    print(f"Log file saved to: {log_path}")
    print("")
    print("Summary of benchmarks performed:")
    print("- Linear system solving (Ax = b):")
    for method in times["solve"].keys():
        print(f"  * {method}: {len(times['solve'][method])} matrix sizes tested")
    print("- Matrix inversion (A^-1):")
    for method in times["inverse"].keys():
        print(f"  * {method}: {len(times['inverse'][method])} matrix sizes tested")
    print("")
    print("Use plot_results.py to visualize the performance comparison.")

if __name__ == '__main__':
    main()