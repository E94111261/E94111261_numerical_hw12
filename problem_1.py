
import numpy as np

def solve_poisson():
    """
    Solves Poisson's Equation
    d^2u/dx^2 + d^2u/dy^2 = xy, for 0 < x < pi, 0 < y < pi/2
    Boundary Conditions:
    u(0,y) = cos(y), u(pi,y) = -cos(y)
    u(x,0) = cos(x), u(x,pi/2) = 0
    Use h = k = 0.1*pi
    """
    print("=============== Problem 1: Poisson's Equation ===============\n")

    h = 0.1 * np.pi
    k = 0.1 * np.pi

    # Based on h and k, calculate n and m 
    n = 9  # (pi/h) - 1
    m = 4  # (pi/2)/k - 1

    # Grid points 
    x = np.linspace(0, np.pi, n + 2)
    y = np.linspace(0, np.pi / 2, m + 2)

    N = n * m  # Total number of unknowns
    A = np.zeros((N, N))
    F = np.zeros(N)

    alpha = h**2 / k**2

    # Define f(x,y) = xy 
    f = lambda xi, yj: xi * yj

    # Build the matrix A and vector F 
    for j in range(1, m + 1):
        for i in range(1, n + 1):
            p = i + n * (j - 1) - 1  # Matrix index (starts from 0)

            # Diagonal elements 
            A[p, p] = -2 * (1 + alpha)

            # Off-diagonal elements 
            if i > 1:
                A[p, p - 1] = 1
            if i < n:
                A[p, p + 1] = 1
            if j > 1:
                A[p, p - n] = alpha
            if j < m:
                A[p, p + n] = alpha

            # Build the F vector from f(xi,yj) 
            F[p] = h**2 * f(x[i], y[j])

            # Incorporate boundary conditions 
            if i == 1:
                F[p] -= np.cos(y[j])      # Boundary u(0,y)
            if i == n:
                F[p] -= -np.cos(y[j])     # Boundary u(pi,y)
            if j == 1:
                F[p] -= alpha * np.cos(x[i]) # Boundary u(x,0)
            if j == m:
                F[p] -= alpha * 0         # Boundary u(x,pi/2)

    # Solve the linear system AU = F 
    U_vec = np.linalg.solve(A, F)
    U = U_vec.reshape((m, n))

    print("Calculated values for u(x,y) at the interior grid points:")
    # Transpose and flip for intuitive (x,y) display
    print(np.flipud(U).T)
    print("\nNote: Matrix rows correspond to the x-direction, and columns correspond to the y-direction.")

if __name__ == '__main__':
    solve_poisson()
