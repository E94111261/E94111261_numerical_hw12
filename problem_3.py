
import numpy as np 

def solve_laplace_polar():
    """
    Solves Laplace's Equation in Polar Coordinates
    T_rr + (1/r)T_r + (1/r^2)T_thetatheta = 0
    for 1/2 <= r <= 1, 0 <= theta <= pi/3
    BC: T(r,0)=0, T(r,pi/3)=0, T(1/2,theta)=50, T(1,theta)=100
    """
    print("=============== Problem 3: Laplace's Equation (Polar) ===============\n")

    # Discretization parameters
    r_start, r_end = 0.5, 1.0
    theta_start, theta_end = 0, np.pi/3

    dr = 0.1
    dtheta = np.pi / 9

    r_coords = np.arange(r_start, r_end + 1e-9, dr)
    theta_coords = np.arange(theta_start, theta_end + 1e-9, dtheta)

    n_p3 = len(r_coords) - 2      # Number of interior r points
    m_p3 = len(theta_coords) - 2  # Number of interior theta points

    N_p3 = n_p3 * m_p3
    A_p3 = np.zeros((N_p3, N_p3))
    F_p3 = np.zeros(N_p3)

    alpha_p3 = (dr / dtheta)**2

    for j in range(1, m_p3 + 1):      # theta index
        for i in range(1, n_p3 + 1):  # r index
            p = i + n_p3 * (j - 1) - 1
            ri = r_coords[i]

            c_center = -2 * (alpha_p3 + ri**2)
            c_iminus1 = ri**2 - (dr / 2) * ri
            c_iplus1 = ri**2 + (dr / 2) * ri

            A_p3[p,p] = c_center

            # r-direction connections 
            if i > 1:
                A_p3[p, p-1] = c_iminus1
            else: # Boundary at r=1/2, T=50 
                F_p3[p] -= c_iminus1 * 50

            if i < n_p3:
                A_p3[p, p+1] = c_iplus1
            else: # Boundary at r=1, T=100 
                F_p3[p] -= c_iplus1 * 100

            # theta-direction connections 
            if j > 1:
                A_p3[p, p-n_p3] = alpha_p3
            else: # Boundary at theta=0, T=0 
                F_p3[p] -= alpha_p3 * 0

            if j < m_p3:
                A_p3[p, p+n_p3] = alpha_p3
            else: # Boundary at theta=pi/3, T=0 
                F_p3[p] -= alpha_p3 * 0

    T_vec = np.linalg.solve(A_p3, F_p3)
    T_matrix = T_vec.reshape((m_p3, n_p3))

    print("Calculated T(r, theta) at interior grid points:")
    print(T_matrix.T) 
    print("\nNote: Matrix rows correspond to the r-direction, columns to the theta-direction.")

if __name__ == '__main__':
    solve_laplace_polar()
