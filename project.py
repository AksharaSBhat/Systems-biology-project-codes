import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

delta_x, delta_y, delta_z = 10, 1, 0.1
alpha_x, alpha_y, alpha_z = 0.4, 0.3, 0.2
beta_x, beta_y, beta_z = 1, 1, 1  
gamma_x, gamma_y, gamma_z = 1, 1, 1
I_values = np.linspace(-2, 12, 100)


x = np.linspace(-10, 20, 400)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

def tilde_f(x_p): return 1 / (1 + np.exp(-x_p))
def tilde_g(x_p): return 1 / (1 + np.exp(-x_p))
def f(x):
    return (0.05 * x**3 - 0.10 * x**2 + 0.1 * x + 0.25)/10
def g(x):
    return (0.05 * x**3 - 0.40 * x**2 + 0.1 * x + 0.25)/10
def h(x):
    return (0.05 * x**3 - 0.50 * x**2 + 0.1 * x + 0.25)/10
def tilde_h(x_p, y_p): return 1 / (1 + np.exp(-(x_p + y_p)))

def system(t, vars, I, delta_y):
    x_r, x_p, y_r, y_p, z_r, z_p = vars
    dx_r = -delta_x * x_r + gamma_x * tilde_f(x_p) + I
    dx_p = -alpha_x * x_p + beta_x * x_r
    dy_r = -delta_y * y_r + gamma_y * tilde_g(x_p)
    dy_p = -alpha_y * y_p + beta_y * y_r
    dz_r = -delta_z * z_r + gamma_z * tilde_h(x_p, y_p)
    dz_p = -alpha_z * z_p + beta_z * z_r
    return [dx_r, dx_p, dy_r, dy_p, dz_r, dz_p]

delta_y_values = [3, 2.1651, 1]  # Varying delta_y
for delta_y in delta_y_values:
    z_p_values = []
    for I in I_values:
        sol = solve_ivp(system, [0, 100], [0, 0, 0, 0, 0, 0], args=(I, delta_y), t_eval=[100])
        z_p = sol.y[-1, -1]  # Steady-state z_p
        z_p_values.append(z_p)

axs[0].plot(x, f(x))
axs[0].set_xlabel("I")
axs[0].set_ylabel("z^P")
axs[0].grid(True)
axs[0].yaxis.set_ticks([])

axs[1].plot(x, g(x))
axs[1].set_xlabel("I")
axs[1].set_ylabel("z^P")
axs[1].grid(True)
axs[1].yaxis.set_ticks([])

axs[2].plot(x, h(x))
axs[2].set_xlabel("I")
axs[2].set_ylabel("z^P")
axs[2].grid(True)
axs[2].yaxis.set_ticks([])

plt.tight_layout()
plt.show()

