Delayed polynomial identification

Current acquired evidence supports meter[t] = A*u[t] + B*u[t-2] + C*u[t-2]^2 and target[t] = D + E*u[t] + F*meter[t]^2 + G*meter[t-1], with zero prehistory. Eliminating the meter yields the 11-term basis in delayed_polynomial.py. Coefficients are fitted from observations, never assumed transferable. A separately acquired 24-row trajectory tests generalization without the meter.
Applicability: finite memory of three controls and this polynomial structure. Refit coefficients for every task. Different delays, longer memory, stochastic readings, or different nonlinearities require revising the basis; an exact fit on one trajectory alone does not establish applicability.
