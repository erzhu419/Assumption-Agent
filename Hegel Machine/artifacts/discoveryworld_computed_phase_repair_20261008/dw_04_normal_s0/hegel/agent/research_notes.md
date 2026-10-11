## Observations
- Reference crystals: gaps/frequencies are crystal 1 = 27.47 nm / 3880.56 Hz (records 120, 107) and crystal 2 = 36.56 nm / 2239.92 Hz (122, 110). Target gaps: crystal 3 = 47.52 nm (30), crystal 4 = 58.2 nm (35).
- Gap-based candidates were unsuccessful: reactor 3 remained uncalibrated at 1290, 1296, 1297, 1300, 1310, 1311, and 1723 Hz (451, 428, 429, 449, 448, 470, 477); empty tuning/insertion at 1290 also failed (464, 466). Reactor 4 remained uncalibrated at 870, 884, 886, and 916 Hz (441, 433, 370-380, 400); empty setting/insertion at 886 failed (370-380). These exclude those specific tests, not every possible condition. ACTIVATE is unsupported on the uncalibrated reactor (295).
- Density readings: crystals 1-4 = 39.36, 22.27, 52.62, 65.02 g/cm^3 (96, 98, 28, 33). A two-point linear fit to reference crystals gives f=96d+102, fitting the two reference values and predicting 5153.52 Hz for crystal 3 and 6343.92 Hz for crystal 4. This is an extrapolation, not a uniquely established law.
- Physical confirmation for crystal 3: record 487 shows crystal 3 inside reactor 3 at 5153.52 Hz and reactor publicly activated; record 488 remains activated at 5154 Hz. Child test_crystal3_density5154 is supported on evidence 487.
- Reference reactor 1 changed activated to uncalibrated and back to activated at 3880.56, 3980, and 3880.56 Hz (422-424), validating that public state readout there is responsive.

## Working explanation and alternatives
Density-linear extrapolation is a promising candidate: it successfully predicted crystal 3's activation setting. Crystal 4's corresponding prediction is 6343.92 Hz, rounded to 6344 Hz. Similarity of this success does not prove the law or guarantee transfer.

## Proposed test (not yet observed)
Test reactor 4 at 6344 Hz with crystal 4 inside, then interpret only actual readback, public containment, and reactor state. The query first exits reactor 3 dialog, teleports to reactor 4 and opens its live menu, then applies +5474 Hz from the known 870-Hz baseline. A successful activated state would support the candidate; an uncalibrated outcome would refute this specific setting, leaving other models/conditions unresolved.