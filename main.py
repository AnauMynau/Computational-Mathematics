import numpy as np


def matrix_inverse_iterative(A, tol=1e-6, max_iter=100, initial_guess=None):
    """
    Find the inverse of a matrix using iterative method.
    
    The iterative formula is:
    1. Compute error matrix: E = AB - I
    2. Refine approximation: B_new = B(I - E + E²)
    
    Parameters:
    -----------
    A : numpy.ndarray
        Square matrix to invert
    tol : float, optional
        Tolerance for convergence (default: 1e-6)
    max_iter : int, optional
        Maximum number of iterations (default: 100)
    initial_guess : numpy.ndarray, optional
        Initial guess B for the inverse. If None, uses B = A^T / (||A||_1 * ||A||_inf)
    
    Returns:
    --------
    B : numpy.ndarray
        Approximate inverse of A
    history : list
        List of iteration history (B_k values)
    errors : list
        List of error matrix norms ||E||_F at each iteration
    converged : bool
        Whether the method converged
    iterations : int
        Number of iterations performed
    """
    A = np.array(A, dtype=float)
    
    # Check if matrix is square
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square to compute its inverse.")
    
    n = A.shape[0]
    I = np.eye(n)
    
    # Initialize B_0 (initial guess for A⁻¹)
    if initial_guess is None:
        # Use scaled transpose as initial guess: B = A^T / (||A||_1 * ||A||_inf)
        norm_A_1 = np.linalg.norm(A, ord=1)
        norm_A_inf = np.linalg.norm(A, ord=np.inf)
        scale = norm_A_1 * norm_A_inf
        if scale < 1e-12:
            raise ValueError("Matrix appears to be singular or near-singular.")
        B = A.T / scale
    else:
        B = np.array(initial_guess, dtype=float)
        if B.shape != A.shape:
            raise ValueError("Initial guess must have the same shape as A.")
    
    history = [B.copy()]
    errors = []
    
    # Iterative refinement
    for k in range(max_iter):
        # Step 1: Compute error matrix E = AB - I
        AB = A @ B
        E = AB - I
        
        # Compute error norm for convergence check
        error_norm = np.linalg.norm(E, ord='fro')
        errors.append(error_norm)
        
        # Check convergence based on error matrix norm
        if error_norm < tol:
            return B, history, errors, True, k + 1
        
        # Step 2: Compute E²
        E_squared = E @ E
        
        # Step 3: Compute I - E + E²
        correction = I - E + E_squared
        
        # Step 4: Refine approximation: B_new = B(I - E + E²)
        B_new = B @ correction
        
        history.append(B_new.copy())
        B = B_new
    
    # Did not converge within max_iter
    return B, history, errors, False, max_iter


def verify_inverse(A, A_inv, tol=1e-5):
    """
    Verify that A_inv is approximately the inverse of A.
    
    Parameters:
    -----------
    A : numpy.ndarray
        Original matrix
    A_inv : numpy.ndarray
        Computed inverse
    tol : float, optional
        Tolerance for verification (default: 1e-5)
    
    Returns:
    --------
    is_valid : bool
        True if A_inv is a valid inverse within tolerance
    error_AA : float
        ||AA_inv - I||_F
    error_AinvA : float
        ||A_inv A - I||_F
    """
    I = np.eye(A.shape[0])
    AA_inv = A @ A_inv
    A_invA = A_inv @ A
    
    error_AA = np.linalg.norm(AA_inv - I, ord='fro')
    error_A_invA = np.linalg.norm(A_invA - I, ord='fro')
    
    is_valid = (error_AA < tol) and (error_A_invA < tol)
    
    return is_valid, error_AA, error_A_invA


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("Iterative Matrix Inversion Method")
    print("=" * 60)
    
    # Example 1: 2x2 matrix from the document (exact example)
    print("\nExample 1: 2x2 Matrix (from document)")
    print("-" * 60)
    A1 = np.array([
        [4, 2],
        [1, 3]
    ], dtype=float)
    
    # Initial guess from the document
    B0 = np.array([
        [0.75, -0.50],
        [0.25, 1.0]
    ], dtype=float)
    
    print("Matrix A:")
    print(A1)
    print("\nInitial guess B:")
    print(B0)
    
    A_inv1, hist1, err1, converged1, iters1 = matrix_inverse_iterative(A1, tol=1e-8, initial_guess=B0)
    
    print(f"\nConverged: {converged1}")
    print(f"Iterations: {iters1}")
    print(f"Final error ||E||_F: {err1[-1]:.2e}")
    
    print("\nComputed inverse:")
    print(A_inv1)
    
    # Show first iteration details (matching the document)
    print("\n" + "=" * 60)
    print("First Iteration Details:")
    print("=" * 60)
    I = np.eye(2)
    AB_first = A1 @ B0
    E_first = AB_first - I
    print("\nE = AB - I:")
    print(E_first)
    print(f"\n||E||_F = {np.linalg.norm(E_first, ord='fro'):.6f}")
    
    E_squared_first = E_first @ E_first
    print("\nE²:")
    print(E_squared_first)
    
    correction_first = I - E_first + E_squared_first
    print("\nI - E + E²:")
    print(correction_first)
    
    B1 = B0 @ correction_first
    print("\nB_new = B(I - E + E²):")
    print(B1)
    
    # Verify with numpy's built-in inverse
    A_inv_exact1 = np.linalg.inv(A1)
    print("\n" + "=" * 60)
    print("Comparison with Exact Inverse:")
    print("=" * 60)
    print("\nExact inverse (numpy.linalg.inv):")
    print(A_inv_exact1)
    
    print("\nDifference (computed - exact):")
    print(A_inv1 - A_inv_exact1)
    print(f"Max absolute difference: {np.max(np.abs(A_inv1 - A_inv_exact1)):.2e}")
    
    is_valid1, err_AA1, err_AinvA1 = verify_inverse(A1, A_inv1)
    print(f"\nVerification: ||AA^(-1) - I||_F = {err_AA1:.2e}")
    print(f"Verification: ||A^(-1)A - I||_F = {err_AinvA1:.2e}")
    print(f"Inverse is valid: {is_valid1}")
    
    # Example 2: 4x4 matrix (from assignment)
    print("\n\nExample 2: 4x4 Matrix (from assignment)")
    print("-" * 60)
    A2 = np.array([
        [ 1, -6,  1, -1],
        [-7,  1, -1,  2],
        [ 1,  1,  8, -1],
        [-2,  1, -1,  7]
    ], dtype=float)
    
    print("Matrix A:")
    print(A2)
    
    A_inv2, hist2, err2, converged2, iters2 = matrix_inverse_iterative(A2, tol=1e-8)
    
    print(f"\nConverged: {converged2}")
    print(f"Iterations: {iters2}")
    print(f"Final error ||E||_F: {err2[-1]:.2e}")
    
    print("\nComputed inverse:")
    print(A_inv2)
    
    # Verify with numpy's built-in inverse
    A_inv_exact2 = np.linalg.inv(A2)
    print("\nExact inverse (numpy.linalg.inv):")
    print(A_inv_exact2)
    
    print("\nDifference (computed - exact):")
    print(A_inv2 - A_inv_exact2)
    print(f"Max absolute difference: {np.max(np.abs(A_inv2 - A_inv_exact2)):.2e}")
    
    is_valid2, err_AA2, err_AinvA2 = verify_inverse(A2, A_inv2)
    print(f"\nVerification: ||AA^(-1) - I||_F = {err_AA2:.2e}")
    print(f"Verification: ||A^(-1)A - I||_F = {err_AinvA2:.2e}")
    print(f"Inverse is valid: {is_valid2}")
    
    # Show convergence history
    print("\n\nConvergence History (Example 2):")
    print("-" * 60)
    print(f"{'Iteration':<12} {'||E||_F':<20}")
    print("-" * 60)
    for i, err in enumerate(err2[:min(10, len(err2))]):
        print(f"{i+1:<12} {err:.6e}")
    if len(err2) > 10:
        print(f"... ({len(err2) - 10} more iterations)")
        print(f"{len(err2):<12} {err2[-1]:.6e}")
    
    # Show convergence history for Example 1
    print("\n\nConvergence History (Example 1):")
    print("-" * 60)
    print(f"{'Iteration':<12} {'||E||_F':<20}")
    print("-" * 60)
    for i, err in enumerate(err1[:min(10, len(err1))]):
        print(f"{i+1:<12} {err:.6e}")
    if len(err1) > 10:
        print(f"... ({len(err1) - 10} more iterations)")
        print(f"{len(err1):<12} {err1[-1]:.6e}")
    
    # Show convergence history for Example 1
    print("\n\nConvergence History (Example 1):")
    print("-" * 60)
    print(f"{'Iteration':<12} {'||E||_F':<20}")
    print("-" * 60)
    for i, err in enumerate(err1[:min(10, len(err1))]):
        print(f"{i+1:<12} {err:.6e}")
    if len(err1) > 10:
        print(f"... ({len(err1) - 10} more iterations)")
        print(f"{len(err1):<12} {err1[-1]:.6e}")
