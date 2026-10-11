# Finite-memory identification

Task 000: instrument readings were exactly quadratic in the preceding control and linear in the current control. Response was an intercept plus a current-control term, squared current instrument reading, and previous instrument reading. Consequently the response is a sparse polynomial of u[t], u[t-1], and u[t-2]. Missing pre-trajectory controls are zero.

Instrumented fitting used 96 rows; an independent 48-row uninstrumented trajectory agreed within 6e-16, including a nonzero initial control and boundary transitions. The equivalent 11-column channel-free basis was full rank and independently validated. The final model refits that basis on all acquired rows and needs no meter.

Reuse finite_memory.py for identified finite-memory polynomial mechanisms. Refit coefficients on each task. Select memory depth and monomials from new evidence; this task does not establish those choices for other mechanisms. Persistent state, nonpolynomial responses, or different reset rules require another model.
