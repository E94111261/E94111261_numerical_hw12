
import numpy as np

def solve_wave():
    """
    Solves the Wave Equation
    p_tt = p_xx, for 0 <= x <= 1, t >= 0
    BC: p(0,t)=1, p(1,t)=2
    IC: p(x,0)=cos(2*pi*x), dp/dt(x,0)=2*pi*sin(2*pi*x)
    Params: dx = dt = 0.1
    """
    print("=============== Problem 4: Wave Equation ===============\n")

    dx = 0.1
    dt = 0.1

    x_coords = np.arange(0, 1.0 + dx, dx)
    n_p4 = len(x_coords) - 2 # Interior points

    lambda_sq = (dt / dx)**2

    # Initial condition p(x,0) 
    p_j_minus_1 = np.cos(2 * np.pi * x_coords) # p at t=0

    # Calculate p at t=dt (j=1) using the special starting formula 
    p_j = np.zeros_like(p_j_minus_1)

    g = lambda x: 2 * np.pi * np.sin(2 * np.pi * x)
    f = lambda x: np.cos(2 * np.pi * x)

    for i in range(1, n_p4 + 1):
        p_j[i] = ( (1 - lambda_sq) * f(x_coords[i]) + 
                   dt * g(x_coords[i]) + 
                   (lambda_sq / 2) * (f(x_coords[i-1]) + f(x_coords[i+1])) )

    p_j[0] = 1.0 
    p_j[-1] = 2.0 

    print(f"t = 0.0: {np.round(p_j_minus_1, 4)}")
    print(f"t = 0.1: {np.round(p_j, 4)}")

    # Iterate for subsequent time steps up to t=0.5
    for t_step in range(2, 6):
        t = t_step * dt
        p_j_plus_1 = np.zeros_like(p_j)

        p_j_plus_1[0] = 1.0
        p_j_plus_1[-1] = 2.0

        # Interior points
        for i in range(1, n_p4 + 1):
            p_j_plus_1[i] = ( lambda_sq * p_j[i-1] + 
                              2 * (1 - lambda_sq) * p_j[i] +
                              lambda_sq * p_j[i+1] - p_j_minus_1[i] )

        # Update time steps
        p_j_minus_1 = p_j.copy()
        p_j = p_j_plus_1.copy()

        print(f"t = {t:.1f}: {np.round(p_j, 4)}")

if __name__ == '__main__':
    solve_wave()
