Delayed polynomial identification

The acquired mechanism is consistent with meter[t] = A*u[t] + B*u[t-d] + C*u[t-d]^2 and target[t] = D + E*u[t] + F*meter[t]^2 + G*meter[t-1], with zero prehistory. Eliminating the instrument yields the 11-term basis in delayed_polynomial.py.

Across tasks, delays and coefficients have changed. Task 1 used d=2; tasks 2 and 3 used d=1. On task 3, refitting to 96 instrumented observations identified d=1 and matched both instrument and response to numerical precision. The unchanged fitted control-only model also matched a separately acquired 16-row trajectory, including a nonzero reset, repeated controls, extreme switching, and relaxation. No auxiliary channel is needed for this deterministic mechanism.

Reuse the basis, identify the delay from current observations, and refit coefficients rather than copying previous coefficients. Memory extends through lag d+1. Changed nonlinearities or stochastic instrument readings can invalidate instrument elimination and require a revised model.
