# Selective revision: new behavior with an old-capability regression

Luna learned both measured repair paths in8calls/450058tokens (37.89%cached input),
using960observations/1728measurement/47298compute. The final source confirms all
four practice cases. On the12new calibration cases it preserves the instrument
in4 and reopens it after refutation in8; both methods pass12/12, with computation
269964→187656 and unchanged observations/measurement.

Full comparison: old44/44, new36/44, zero false confirmations. All8failures are
earlier stationary instrument/joint changes (episodes24–31). The new source
requires a failed parent confirmation before rebuilding an instrument, so an
initially refuted instrument only triggers repeated measurement. Original32
regression accuracy falls32/32→24/32; overall improvement is not established.

`summary.json` retains the comparison and actual selective-repair traces. Next:
add an actual initial-refutation failure to practice, retain the original four
cases, and let the model repair its method; use additional new calibration cases
after that development step. No learned source was hand-edited.
