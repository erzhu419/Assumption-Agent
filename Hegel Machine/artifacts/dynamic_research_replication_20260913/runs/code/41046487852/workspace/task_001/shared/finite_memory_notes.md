# Finite-memory identification

Task 000: instrument readings were exactly quadratic in the preceding control and linear in the current control. Response was an intercept plus a current-control term, squared current instrument reading, and previous instrument reading. Consequently the response is a sparse polynomial of u[t], u[t-1], and u[t-2]. Missing pre-trajectory controls are zero.

Instrumented fitting used 96 rows; an independent 48-row uninstrumented trajectory agreed within 6e-16, including a nonzero initial control and boundary transitions. The equivalent 11-column channel-free basis was full rank and independently validated. The final model refits that basis on all acquired rows and needs no meter.

Reuse finite_memory.py for identified finite-memory polynomial mechanisms. Refit coefficients on each task. Select memory depth and monomials from new evidence; this task does not establish those choices for other mechanisms. Persistent state, nonpolynomial responses, or different reset rules require another model.

Task 001: The same 11-term, two-lag basis was refitted on current observations. Fitting the first 64 rows gave full rank and a maximum error of 1.22e-15 on the remaining 32 rows. Refitting all 96 rows gave MSE 6.17e-31. This supports reusing the channel-free structure for this task; coefficients were estimated anew.
