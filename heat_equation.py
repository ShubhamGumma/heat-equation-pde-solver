import numpy as np
import matplotlib.pyplot as plt
from pde import PDE, CartesianGrid, ScalarField

def solve_heat_equation():
    """
    Solve the 1D heat equation: du/dt = d²u/dx²
    with a Gaussian initial condition on [0, 2π].
    CFL-style time step selection is applied to improve stability.
    """

    # Grid: use more points for a smoother, more physical result
    n_points = 128
    grid = CartesianGrid([[0, 2*np.pi]], n_points)
    x = grid.axes_coords[0]

    # Initial condition: Gaussian pulse centered at π
    initial_data = np.exp(-((x - np.pi)**2) / 0.1)
    state = ScalarField(grid, initial_data)

    # PDE definition: heat equation u_t = Δu
    eq = PDE({"u": "laplace(u)"})

    # Space step and CFL-style dt selection
    dx = float(x[1] - x[0])
    # Rule of thumb from feedback: choose dt < (dx)^2
    # Add a safety factor c = 0.5 and cap at 0.01 as suggested
    dt_cfl = 0.5 * dx * dx
    dt = min(0.01, dt_cfl)

    # Longer time range so the solution visibly diffuses
    t_range = 1.0

    print("=== Heat Equation (1D) ===")
    print(f"Grid points: {n_points}")
    print(f"dx = {dx:.6f}")
    print(f"Chosen dt = {dt:.6f} (min(0.01, 0.5*dx^2) with CFL-style safety)")
    print(f"t_range = {t_range}")

    # Solve
    result = eq.solve(state, t_range=t_range, dt=dt)

    # Indices of diagnostic locations: near x=2 and at x=π
    idx_near_2 = int(np.argmin(np.abs(x - 2.0)))
    idx_pi = int(np.argmin(np.abs(x - np.pi)))

    # Diagnostics: values should not increase spuriously at cold points
    u0_near_2 = float(initial_data[idx_near_2])
    uT_near_2 = float(result.data[idx_near_2])
    u0_pi = float(initial_data[idx_pi])
    uT_pi = float(result.data[idx_pi])

    print("\n=== Diagnostics ===")
    print(f"x near 2.0 -> initial: {u0_near_2:.6f}, final: {uT_near_2:.6f}")
    print(f"x = π      -> initial: {u0_pi:.6f}, final: {uT_pi:.6f}")

    # Simple physicality checks for diffusion
    max0, maxT = float(initial_data.max()), float(result.data.max())
    min0, minT = float(initial_data.min()), float(result.data.min())
    print(f"max(initial) = {max0:.6f}, max(final) = {maxT:.6f}")
    print(f"min(initial) = {min0:.6f}, min(final) = {minT:.6f}")
    if maxT <= max0 + 1e-8:
        print("peak decreased or stayed the same, consistent with diffusion.")
    else:
        print("peak increased, consider refining dt or grid further.")

    # Optional Laplacian check at x ≈ π for the initial profile
    laplace_at_pi = (initial_data[idx_pi - 1] - 2 * initial_data[idx_pi] + initial_data[idx_pi + 1]) / (dx ** 2)
    print(f"Laplacian at x ≈ π (t=0): {laplace_at_pi:.6f}")

    # Plot initial and final profiles
    plt.figure(figsize=(12, 4))
    # Initial
    plt.subplot(1, 2, 1)
    plt.plot(x, initial_data, linewidth=2)
    plt.title("Initial Condition")
    plt.xlabel("x")
    plt.ylabel("u(x, 0)")
    plt.grid(True)

    # Final
    plt.subplot(1, 2, 2)
    plt.plot(x, result.data, linewidth=2)
    plt.title(f"Final State (t={t_range})")
    plt.xlabel("x")
    plt.ylabel(f"u(x, {t_range})")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print("Heat equation example completed!")

    return result

if __name__ == "__main__":
    print("Running Heat Equation Example...")
    solve_heat_equation()
    print("Heat equation example completed!")
