"""
Simple Implementation and Demonstration of Higher Order Jacobi Method (HOJM)

This module provides a clean, educational implementation of the Higher Order Jacobi Method
for solving linear systems and computing matrix inverses. It demonstrates the core algorithm
with clear examples showing both convergent and divergent cases.

The Higher Order Jacobi Method extends the classical Jacobi iteration by pre-computing
higher-order coefficient matrices that can be reused for multiple solves. This approach
is particularly beneficial when solving multiple systems with the same coefficient matrix
but different right-hand sides.

Key Features:
- Pure PyTorch implementation for GPU acceleration
- Educational examples with detailed comparisons
- Demonstration of convergence requirements (diagonal dominance)
- Both direct solving and matrix inversion capabilities

Mathematical Foundation:
The method decomposes A = D + A₀ where D is diagonal and A₀ is off-diagonal, then
iteratively computes higher-order coefficients using:
- α₀ = -A₀/D (initial Jacobi iteration matrix)
- αₖ₊₁ = αₖ² (higher-order terms)
- Solution: x = Σᵢ αᵢ₊₁ * (∏ⱼ₌₀ⁱ (αⱼ + I)) * (b/D)

Author: Nithin Kumar Goona, Ph.D.
Affiliation: The University of Texas at El Paso (UTEP)
Created on: July 29, 2025

Description:
    Educational implementation of HOJM featuring:
    - Core algorithm functions with clear mathematical formulation
    - Demonstration examples comparing HOJM vs PyTorch linalg
    - Convergent case: diagonally dominant tridiagonal matrix
    - Divergent case: non-diagonally dominant matrix
    - Both linear system solving and matrix inversion examples
    
    This serves as both a reference implementation and educational tool
    for understanding the Higher Order Jacobi Method.

License:
    MIT

Contact:
    nkgoona@utep.edu, nithin.goona@gmail.com
"""

import torch

def generate_coeffecients(A, b, num_iter, solve_while_generating=False):
    """
    Generate higher-order coefficient matrices for the HOJM algorithm.
    
    This is the core function that implements the Higher Order Jacobi Method by
    pre-computing coefficient matrices that encode the iterative solution process.
    These coefficients can be reused for multiple solves with the same matrix A.
    
    Mathematical Algorithm:
    1. Decompose A = D + A₀ where D is diagonal, A₀ is off-diagonal
    2. Initialize: α₀ = -A₀/D, β₀ = b/D
    3. Iterate: αₖ₊₁ = αₖ², βₖ₊₁ = (αₖ + I)βₖ
    4. Store all αₖ for later use
    
    Args:
        A (torch.Tensor): System matrix (must be diagonally dominant for convergence)
        b (torch.Tensor): Right-hand side vector
        num_iter (int): Number of higher-order terms to compute
        solve_while_generating (bool): If True, also compute solution during generation
        
    Returns:
        tuple: (Alphas, solution) where:
            - Alphas (list): Higher-order coefficient matrices [α₀, α₁, ..., αₙ]
            - solution (torch.Tensor): Solution vector if solve_while_generating=True,
                                     otherwise None
                                     
    Note:
        Convergence requires diagonal dominance: |A[i,i]| > Σⱼ≠ᵢ|A[i,j]| for all i.
        More iterations generally improve accuracy but increase computational cost.
    """
    # Create identity matrix matching A's properties (device, dtype)
    I = torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
    
    # Extract diagonal elements and reshape for broadcasting
    D_flat = torch.diag(A).unsqueeze(1)  # Diagonal as column vector
    
    # Create off-diagonal part by zeroing the diagonal
    A0 = A - torch.diag_embed(torch.diag(A))  # A₀ = A - D
    
    # Initialize first-order coefficients
    Alpha = -A0/D_flat  # α₀ = -A₀/D (Jacobi iteration matrix)
    Beta = b/D_flat     # β₀ = b/D (scaled right-hand side)

    # Initialize solution vector if requested
    if solve_while_generating:
        x = b.clone()   # Starting guess
        
    # Store coefficient matrices
    Alphas = [Alpha]    # List to store all αₖ
    
    # Generate higher-order coefficients iteratively
    for _ in range(num_iter):
        # Update scaled RHS: βₖ₊₁ = (αₖ + I)βₖ
        Beta = torch.matmul(Alpha + I, Beta)
        
        # Update coefficient matrix: αₖ₊₁ = αₖ²
        Alpha = torch.matmul(Alpha, Alpha)
        Alphas.append(Alpha)
        
        # Update solution estimate if requested: xₖ₊₁ = αₖ₊₁xₖ + βₖ₊₁
        if solve_while_generating:
            x = torch.matmul(Alpha, x) + Beta

    # Return coefficients and solution (if computed)
    if solve_while_generating:
        return Alphas, x
    else:
        return Alphas, None

def solve_with_coeffecients(Alphas, A, b):
    """
    Solve linear system using pre-computed HOJM coefficients.
    
    This function implements the inference phase of HOJM, using pre-computed
    coefficient matrices to solve Ax = b efficiently. This is particularly
    advantageous when solving multiple systems with the same matrix A but
    different right-hand sides b, as the expensive coefficient generation
    is done only once.
    
    Mathematical Algorithm:
    1. Initialize: β₀ = b/D, x₀ = b
    2. For each coefficient αᵢ:
       - Update RHS: βᵢ₊₁ = (αᵢ + I)βᵢ
       - Update solution: x = αᵢ₊₁x + βᵢ₊₁
    
    Args:
        Alphas (list): Pre-computed coefficient matrices from generate_coeffecients()
        A (torch.Tensor): Original system matrix (used for diagonal extraction)
        b (torch.Tensor): Right-hand side vector
        
    Returns:
        torch.Tensor: Solution vector x such that Ax ≈ b
        
    Note:
        This function assumes the coefficients were computed for the same
        matrix A. The accuracy depends on the number of iterations used
        during coefficient generation.
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
    coefficient matrices generated by generate_coeffecients(). The inverse
    is computed using the mathematical relationship:
    
    A⁻¹ = D⁻¹ * ∏ᵢ₌₀ⁿ (αᵢ + I)
    
    where D is the diagonal of A and αᵢ are the coefficient matrices.
    
    Mathematical Algorithm:
    1. Initialize: inv_A = (α₀ + I)
    2. For each remaining coefficient: inv_A = inv_A * (αᵢ + I)
    3. Apply diagonal scaling: A⁻¹ = D⁻¹ * inv_A
    
    Args:
        Alphas (list): Pre-computed coefficient matrices from generate_coeffecients()
        A (torch.Tensor): Original system matrix (used for diagonal extraction)
        
    Returns:
        torch.Tensor: Approximate inverse matrix A⁻¹
        
    Note:
        The accuracy of the inverse depends on the number of iterations used
        during coefficient generation. More iterations generally yield better
        approximations but at higher computational cost. The matrix A must
        be diagonally dominant for convergence.
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
    Demonstration of Higher Order Jacobi Method with educational examples.
    
    This function provides comprehensive examples showing both convergent and
    divergent cases of HOJM, comparing results against PyTorch's built-in
    linear algebra functions. It demonstrates:
    
    1. Convergent case: Diagonally dominant tridiagonal matrix
    2. Divergent case: Non-diagonally dominant matrix
    
    For each case, it compares:
    - Direct linear system solving (Ax = b)
    - Matrix inversion (A^-1)
    - Solution via matrix inversion (A^-1 * b)
    
    This serves as both a validation of the implementation and an educational
    tool for understanding convergence requirements.
    """
    
    # ============================================================================
    # SETUP AND CONFIGURATION
    # ============================================================================
    
    # Set computation device (CPU for this educational example)
    device = torch.device('cpu')
    
    # Configure PyTorch printing for better readability
    torch.set_printoptions(sci_mode=False, precision=6)
    
    print("Higher Order Jacobi Method - Educational Demonstration")
    print("=" * 60)

    # ============================================================================
    # 1. CONVERGENT CASE: DIAGONALLY DOMINANT MATRIX
    # ============================================================================
    
    print("\n1. CONVERGENT CASE: Diagonally Dominant Matrix")
    print("-" * 50)
    
    # 1.0 Define test matrices
    print("\n1.0 Matrix Definition:")
    
    # Option 1: Dense diagonally dominant matrix (commented out)
    # Diagonal dominance ensures convergence: |A[i,i]| > Σⱼ≠ᵢ|A[i,j]| for all i
    # A = torch.tensor([[2.0, -0.1, 0.3, 0.07], 
    #                 [0.1, 2.0, -0.2, -0.5],
    #                 [0.3, 0.2, 1.0, 0.1],
    #                 [-0.06, 0.3, -0.4, 1.0]], device=device)
    # b = torch.tensor([[9.0], [0.0], [-1.0], [2.0]], device=device)

    # Option 2: Sparse tridiagonal matrix (used in this example)
    # This represents a 1D finite difference discretization with boundary conditions
    # Diagonal elements = 2, off-diagonal elements = -1, boundary rows = identity
    A = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0],      # Boundary condition: u₀ = b₀
                    [-1, 2, -1, 0, 0, 0, 0, 0],       # Interior: -u₍ᵢ₋₁₎ + 2uᵢ - u₍ᵢ₊₁₎ = bᵢ
                    [0, -1, 2, -1, 0, 0, 0, 0],       # Interior point
                    [0, 0, -1, 2, -1, 0, 0, 0],       # Interior point
                    [0, 0, 0, -1, 2, -1, 0, 0],       # Interior point
                    [0, 0, 0, 0, -1, 2, -1, 0],       # Interior point
                    [0, 0, 0, 0, 0, -1, 2, -1],       # Interior point
                    [0, 0, 0, 0, 0, 0, 0, 1]], dtype=torch.float32, device=device)  # Boundary condition: u₇ = b₇
    
    # Right-hand side vector
    b = torch.tensor([[9.0], [0.0], [-1.0], [2.0], [9.0], [0.0], [-1.0], [2.0]], dtype=torch.float32, device=device)
    
    print(f"Matrix A shape: {A.shape}")
    print(f"Matrix A (tridiagonal with boundary conditions):")
    print(A)
    print(f"\nRight-hand side b:")
    print(b.flatten())


    # ============================================================================
    # 1.1 Linear System Solving Comparison (Ax = b)
    # ============================================================================
    
    print("\n1.1 Linear System Solving Comparison:")
    print("Comparing PyTorch linalg.solve() vs HOJM direct solving...")
    
    # Solve using PyTorch's built-in linear algebra (reference solution)
    linalg_solution = torch.linalg.solve(A, b)
    print("PyTorch linalg.solve() result:")
    print(linalg_solution.flatten())
    
    # Solve using HOJM with 6 iterations (should converge for diagonally dominant matrix)
    higher_order_coeffecients, higher_order_solution = generate_coeffecients(A, b, 6, solve_while_generating=True)
    print("\nHOJM solution (6 iterations):")
    print(higher_order_solution.flatten())
    
    # Compute and display solution error
    solution_error = torch.norm(linalg_solution - higher_order_solution).item()
    print(f"\nSolution error (L2 norm): {solution_error:.2e}")

    # ============================================================================
    # 1.2 Matrix Inversion Comparison (A^-1)
    # ============================================================================
    
    print("\n1.2 Matrix Inversion Comparison:")
    print("Comparing PyTorch linalg.inv() vs HOJM matrix inversion...")
    
    # Compute inverse using PyTorch's built-in linear algebra (reference)
    linalg_inverse = torch.linalg.inv(A)
    print("PyTorch linalg.inv() result:")
    print(linalg_inverse)
    
    # Compute inverse using HOJM with pre-computed coefficients
    higher_order_inverse = find_inverse_with_coeffecients(higher_order_coeffecients, A)
    print("\nHOJM inverse result:")
    print(higher_order_inverse)
    
    # Compute and display inverse error
    inverse_error = torch.norm(linalg_inverse - higher_order_inverse).item()
    print(f"\nInverse error (Frobenius norm): {inverse_error:.2e}")

    # ============================================================================
    # 1.3 Solution via Matrix Inversion (A^-1 * b)
    # ============================================================================
    
    print("\n1.3 Solution via Matrix Inversion:")
    print("Comparing solutions obtained by multiplying inverse with RHS...")
    
    # Solution via PyTorch inverse
    linalg_inverse_solution = torch.matmul(linalg_inverse, b)
    print("PyTorch inverse solution (A^-1 * b):")
    print(linalg_inverse_solution.flatten())
    
    # Solution via HOJM inverse
    higher_order_inverse_solution = torch.matmul(higher_order_inverse, b)
    print("\nHOJM inverse solution (A^-1 * b):")
    print(higher_order_inverse_solution.flatten())
    
    # Compute and display error between inverse-based solutions
    inverse_solution_error = torch.norm(linalg_inverse_solution - higher_order_inverse_solution).item()
    print(f"\nInverse solution error (L2 norm): {inverse_solution_error:.2e}")
    
    # Verify that A * A^-1 ≈ I (identity matrix check)
    identity_check = torch.matmul(A, higher_order_inverse)
    identity_error = torch.norm(identity_check - torch.eye(A.shape[0], device=device)).item()
    print(f"Identity verification (||A * A^-1 - I||): {identity_error:.2e}")



    # ============================================================================
    # 2. DIVERGENT CASE: NON-DIAGONALLY DOMINANT MATRIX
    # ============================================================================
    
    print("\n\n2. DIVERGENT CASE: Non-Diagonally Dominant Matrix")
    print("-" * 55)
    
    # 2.0 Define test matrices
    print("\n2.0 Matrix Definition:")
    print("WARNING: This matrix violates diagonal dominance - HOJM will diverge!")
    
    # Non-diagonally dominant matrix - violates convergence condition
    # For convergence, we need |A[i,i]| > Σⱼ≠ᵢ|A[i,j]| for all i
    # This matrix fails this condition, demonstrating divergence
    A = torch.tensor([[1.1, 2.0, 1.0],    # |1.1| < |2.0| + |1.0| = 3.0 ❌
                    [2.0, 1.15, 2.0],     # |1.15| < |2.0| + |2.0| = 4.0 ❌
                    [1.0, 2.0, 1.05]], device=device)  # |1.05| < |1.0| + |2.0| = 3.0 ❌

    b = torch.tensor([[4.0], [6.0], [4.0]], device=device)
    
    print(f"Matrix A (non-diagonally dominant):")
    print(A)
    print(f"\nRight-hand side b:")
    print(b.flatten())
    
    # Check diagonal dominance condition
    print("\nDiagonal dominance check:")
    for i in range(A.shape[0]):
        diag_val = abs(A[i, i].item())
        off_diag_sum = sum(abs(A[i, j].item()) for j in range(A.shape[1]) if j != i)
        dominance = "✓" if diag_val > off_diag_sum else "❌"
        print(f"Row {i}: |{A[i,i].item():.2f}| {'>' if diag_val > off_diag_sum else '<'} {off_diag_sum:.2f} {dominance}")

    # ============================================================================
    # 2.1 Linear System Solving Comparison - Divergence Demonstration
    # ============================================================================
    
    print("\n2.1 Linear System Solving Comparison:")
    print("Demonstrating HOJM divergence for non-diagonally dominant matrix...")
    
    # Solve using PyTorch's built-in linear algebra (reference solution)
    linalg_solution = torch.linalg.solve(A, b)
    print("PyTorch linalg.solve() result (correct):")
    print(linalg_solution.flatten())
    
    # Attempt to solve using HOJM - this will diverge!
    higher_order_coeffecients, higher_order_solution = generate_coeffecients(A, b, 6, solve_while_generating=True)
    print("\nHOJM solution (6 iterations) - DIVERGED:")
    print(higher_order_solution.flatten())
    
    # Compute and display solution error - will be large due to divergence
    solution_error = torch.norm(linalg_solution - higher_order_solution).item()
    print(f"\nSolution error (L2 norm): {solution_error:.2e} - LARGE ERROR DUE TO DIVERGENCE!")

    # ============================================================================
    # 2.2 Matrix Inversion Comparison - Divergence Demonstration
    # ============================================================================
    
    print("\n2.2 Matrix Inversion Comparison:")
    print("Demonstrating HOJM inverse divergence...")
    
    # Compute inverse using PyTorch's built-in linear algebra (reference)
    linalg_inverse = torch.linalg.inv(A)
    print("PyTorch linalg.inv() result (correct):")
    print(linalg_inverse)
    
    # Attempt to compute inverse using HOJM - this will also diverge!
    higher_order_inverse = find_inverse_with_coeffecients(higher_order_coeffecients, A)
    print("\nHOJM inverse result - DIVERGED:")
    print(higher_order_inverse)
    
    # Compute and display inverse error - will be large due to divergence
    inverse_error = torch.norm(linalg_inverse - higher_order_inverse).item()
    print(f"\nInverse error (Frobenius norm): {inverse_error:.2e} - LARGE ERROR DUE TO DIVERGENCE!")

    # ============================================================================
    # 2.3 Solution via Matrix Inversion - Divergence Propagation
    # ============================================================================
    
    print("\n2.3 Solution via Matrix Inversion:")
    print("Showing how inversion errors propagate to solution errors...")
    
    # Solution via PyTorch inverse
    linalg_inverse_solution = torch.matmul(linalg_inverse, b)
    print("PyTorch inverse solution (A^-1 * b) - correct:")
    print(linalg_inverse_solution.flatten())
    
    # Solution via HOJM inverse - errors compound
    higher_order_inverse_solution = torch.matmul(higher_order_inverse, b)
    print("\nHOJM inverse solution (A^-1 * b) - DIVERGED:")
    print(higher_order_inverse_solution.flatten())
    
    # Compute and display error between inverse-based solutions
    inverse_solution_error = torch.norm(linalg_inverse_solution - higher_order_inverse_solution).item()
    print(f"\nInverse solution error (L2 norm): {inverse_solution_error:.2e} - LARGE ERROR!")
    
    # Identity check will also fail badly
    identity_check = torch.matmul(A, higher_order_inverse)
    identity_error = torch.norm(identity_check - torch.eye(A.shape[0], device=device)).item()
    print(f"Identity verification (||A * A^-1 - I||): {identity_error:.2e} - FAILED!")
    
    # ============================================================================
    # EDUCATIONAL SUMMARY
    # ============================================================================
    
    print("\n" + "=" * 60)
    print("EDUCATIONAL SUMMARY:")
    print("=" * 60)
    print("\n✓ CONVERGENT CASE (Section 1):")
    print("  - Diagonally dominant matrix: |A[i,i]| > Σⱼ≠ᵢ|A[i,j]|")
    print("  - HOJM converges to correct solution")
    print("  - Small errors indicate successful convergence")
    
    print("\n❌ DIVERGENT CASE (Section 2):")
    print("  - Non-diagonally dominant matrix violates convergence condition")
    print("  - HOJM diverges, producing incorrect results")
    print("  - Large errors indicate divergence failure")
    
    print("\nKEY TAKEAWAY:")
    print("Diagonal dominance is ESSENTIAL for HOJM convergence!")
    print("Always verify this condition before applying HOJM.")


if __name__ == '__main__':
    main()