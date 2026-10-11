Delayed polynomial identification

Acquired evidence supports meter[t] = A*u[t] + B*u[t-d] + C*u[t-d]^2 and target[t] = D + E*u[t] + F*meter[t]^2 + G*meter[t-1], with zero prehistory. Eliminating the meter yields the 11-term basis in delayed_polynomial.py.

The previous task used d=2; the current task requires d=1. The previous delay failed on current observations, whereas the revised delay matched the instrument and target to numerical precision. A separately acquired 24-row control-only trajectory, including repeated extremes, switching, and reset transients, also matched without refitting. No auxiliary channel is needed for prediction.

Reuse the basis with an evidence-supported delay and refit coefficients on each task. Applicable memory extends through lag d+1. Changed delays or nonlinearities require revising the model; stochastic readings may make eliminating the instrument inappropriate.
