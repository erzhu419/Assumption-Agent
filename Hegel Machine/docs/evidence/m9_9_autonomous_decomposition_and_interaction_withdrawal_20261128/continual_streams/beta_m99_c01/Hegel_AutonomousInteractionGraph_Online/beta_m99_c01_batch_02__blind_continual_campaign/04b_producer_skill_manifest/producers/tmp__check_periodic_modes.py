import importlib.util
import numpy as np

spec = importlib.util.spec_from_file_location('downstream_solver', '/holdout_task/downstream_solver.py')
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)
nx, ny = 7, 10
lx, ly = 2.4, 4.1
hx, hy = lx / (nx-1), ly / (ny-1)
kx, ky, reaction = 0.8, 1.3, 0.27
shifts = [0.12, 0.51, 1.7]
i, j = np.indices((nx, ny))
phase = 2*np.pi*(i/nx + 2*j/ny)
wave = np.exp(1j*phase)
modes = np.stack([wave.real, wave.imag])
sensors = np.array([[0, 0], [nx-1, 0], [0, ny-1], [nx-1, ny-1], [2, 3], [5, 8]])

def symbol(theta, kappa, velocity, h):
    return 2*kappa*(1-np.cos(theta))/h**2 + abs(velocity)*(1-np.exp(-1j*np.sign(velocity)*theta))/h

worst_response = worst_terminal = 0.0
for vx, vy in ((0.61, -0.83), (-0.61, 0.83)):
    study = dict(nx=nx, ny=ny, Lx=lx, Ly=ly, kappa_x=np.full(nx, kx), kappa_y=np.full(ny, ky), vel_x=np.full(nx, vx), vel_y=np.full(ny, vy), V_reaction_2d=np.full((nx,ny), reaction), source_modes=modes, mu_shifts=shifts, sensor_ij=sensors)
    response, terminal = solver._forward(study)
    eigenvalue = symbol(2*np.pi/nx, kx, vx, hx) + symbol(4*np.pi/ny, ky, vy, hy) + reaction
    expected = [wave / (eigenvalue + shift) for shift in shifts]
    readings = np.concatenate([u[tuple(sensors.T)] for u in expected])
    worst_response = max(worst_response, np.max(np.abs(response - np.column_stack([readings.real, readings.imag]))))
    worst_terminal = max(worst_terminal, np.max(np.abs(terminal - np.stack([expected[-1].real, expected[-1].imag]))))
assert max(worst_response, worst_terminal) < 1e-12
print(f'Analytic periodic mode: worst corner/interior response error={worst_response:.3e}, worst terminal mode error={worst_terminal:.3e}')
