"""5 task slices, 50 problems each = 250 total, each with numeric/string gold.

Slices designed so gemini-3-flash baseline accuracy is in 30-70% range
(not floor, not ceiling) — leaving headroom for cards to demonstrably help.

Schema per problem:
  {pid, slice, prompt, gold, tol (numeric)}
gold can be float (numeric, with tol), int (exact), or str (substring match).
"""

# ============================================================
# Slice 1: BAYESIAN — base-rate, conditional, Monty-Hall family
# ============================================================
SLICE_BAYESIAN = [
    # Base-rate disease problems (variations)
    {"prompt": "1% of patients have disease X. Test sensitivity 95%, specificity 95%. P(disease|positive)? Decimal 3 places.", "gold": 0.161, "tol": 1.10},
    {"prompt": "0.5% prevalence. Test sensitivity 99%, specificity 98%. P(disease|positive)? Decimal 3 places.", "gold": 0.199, "tol": 1.15},
    {"prompt": "0.1% prevalence. Test 95% sensitive, 90% specific. P(disease|positive)? Decimal 4 places.", "gold": 0.0094, "tol": 1.20},
    {"prompt": "10% prevalence. Test 80% sens, 90% spec. P(disease|positive)? Decimal 3 places.", "gold": 0.471, "tol": 1.10},
    {"prompt": "0.01% prevalence. Test 99% sens, 99% spec. P(disease|positive)? Decimal 4 places.", "gold": 0.0098, "tol": 1.25},
    # Two-children family
    {"prompt": "A family has 2 children. At least one is a girl. P(both girls)? Decimal.", "gold": 0.333, "tol": 1.05},
    {"prompt": "A family has 2 children. The first born is a girl. P(both girls)? Decimal.", "gold": 0.5, "tol": 1.02},
    {"prompt": "A family has 3 children. At least one is a boy. P(all 3 boys)? Decimal 3 places.", "gold": 0.143, "tol": 1.10},
    {"prompt": "A family has 4 children. At least 2 are boys. P(all 4 boys)? Decimal 3 places.", "gold": 0.091, "tol": 1.10},
    {"prompt": "A family has 2 children, born on different days. At least one was born on Tuesday. P(both children boys | one boy born Tuesday)? Decimal 3 places.", "gold": 0.481, "tol": 1.12},
    # Monty Hall + variants
    {"prompt": "3 doors, 1 car. You pick door 1. Host opens door 3 (goat). Should you switch? P(car if switch)? Decimal 3 places.", "gold": 0.667, "tol": 1.05},
    {"prompt": "5 doors, 1 car. You pick door 1. Host opens 3 goat doors. P(car if switch to last)? Decimal 3 places.", "gold": 0.8, "tol": 1.05},
    {"prompt": "10 doors, 1 car. You pick door 1. Host opens 8 goat doors. P(car if switch)? Decimal 3 places.", "gold": 0.9, "tol": 1.03},
    {"prompt": "Monty Forgetful: 3 doors, host doesn't know where car is, randomly opens door 3 which happens to be a goat. P(car if you switch to door 2)? Decimal.", "gold": 0.5, "tol": 1.04},
    {"prompt": "3 doors, 1 car. You pick door 1. Before opening anything, host says 'door 2 has a goat'. P(car behind door 3)? Decimal.", "gold": 0.667, "tol": 1.05},
    # Conditional probability via enumeration
    {"prompt": "Roll 2 fair dice. Given sum=8, P(at least one is 5)? Decimal 3 places.", "gold": 0.4, "tol": 1.05},
    {"prompt": "Roll 2 fair dice. Given product=6, P(one die is 2)? Decimal 3 places.", "gold": 0.5, "tol": 1.05},
    {"prompt": "Roll 3 fair dice. Given sum=5, P(at least one is 1)? Decimal 3 places.", "gold": 1.0, "tol": 1.001},
    {"prompt": "Roll 2 fair dice. Given the larger is 5 (or both are 5), P(sum=8)? Decimal 3 places.", "gold": 0.222, "tol": 1.10},
    {"prompt": "Draw 2 cards from a standard deck without replacement. Given first is red, P(second is red)? Decimal 3 places.", "gold": 0.490, "tol": 1.05},
    # Three-cards / signal-detection
    {"prompt": "Three cards: card 1 has both sides red, card 2 has both sides blue, card 3 has one red side and one blue side. Pick a card at random, look at one side: it's red. P(other side is red)? Decimal 3 places.", "gold": 0.667, "tol": 1.05},
    {"prompt": "Two coins: A is fair, B has heads on both sides. You pick one and flip: heads. P(you picked B)? Decimal 3 places.", "gold": 0.667, "tol": 1.05},
    {"prompt": "Three boxes A,B,C. A has 1 white, 2 black; B has 2 white, 1 black; C has 3 white. Pick a box, draw a ball: white. P(box was C)? Decimal 3 places.", "gold": 0.5, "tol": 1.05},
    {"prompt": "1 in 10000 athletes uses a banned drug. The test has 0.5% false-positive rate, 0% false-negative. P(actual user | positive test)? Decimal 3 places.", "gold": 0.020, "tol": 1.20},
    {"prompt": "5% have condition. Test catches 98% of true positives, flags 8% of healthy. P(condition | positive)? Decimal 3 places.", "gold": 0.392, "tol": 1.10},
    # Order-of-events conditional
    {"prompt": "Box A: 4 red 6 blue. Box B: 7 red 3 blue. Pick box at random, draw 1 ball: red. P(box was B)? Decimal 3 places.", "gold": 0.636, "tol": 1.05},
    {"prompt": "P(rain)=0.3. P(forecast says rain | rain)=0.9. P(forecast says rain | no rain)=0.2. Forecast says rain. P(rain)? Decimal 3 places.", "gold": 0.659, "tol": 1.05},
    {"prompt": "P(spam)=0.4. P(word 'free' | spam)=0.6. P(word 'free' | not spam)=0.05. Email contains 'free'. P(spam)? Decimal 3 places.", "gold": 0.889, "tol": 1.05},
    {"prompt": "Two factories. F1 makes 60% of widgets, 2% defective. F2 makes 40%, 5% defective. A widget is defective. P(it came from F2)? Decimal 3 places.", "gold": 0.625, "tol": 1.05},
    {"prompt": "Coin is fair with P=0.5, biased coin with P(heads)=0.7. You pick one randomly. Flip 3 times: HHH. P(it was the biased coin)? Decimal 3 places.", "gold": 0.733, "tol": 1.05},
    # Independence vs dependence
    {"prompt": "Two events A,B are independent. P(A)=0.4, P(B)=0.3. P(A and B)? Decimal.", "gold": 0.12, "tol": 1.02},
    {"prompt": "P(A)=0.5, P(B)=0.5, P(A and B)=0.3. Are A and B independent? Answer 'yes' or 'no'.", "gold": "no"},
    {"prompt": "P(rain)=0.3, P(traffic)=0.4, P(rain and traffic)=0.18. P(traffic | rain)? Decimal 3 places.", "gold": 0.6, "tol": 1.05},
    {"prompt": "Roll 2 dice. Are events 'first die =3' and 'sum =7' independent? Answer yes/no.", "gold": "yes"},
    {"prompt": "Roll 2 dice. Are events 'first die >=4' and 'sum >=10' independent? Answer yes/no.", "gold": "no"},
    # Birthday / collision
    {"prompt": "23 people in a room. P(at least 2 share a birthday) assuming uniform 365-day year? Decimal 3 places.", "gold": 0.507, "tol": 1.05},
    {"prompt": "10 people. P(at least 2 share a birthday)? Decimal 3 places.", "gold": 0.117, "tol": 1.10},
    {"prompt": "5 people pick a number 1-100 independently uniformly. P(at least 2 pick same)? Decimal 3 places.", "gold": 0.097, "tol": 1.10},
    {"prompt": "100 people randomly distributed across 50 boxes. P(some box gets at least 3)? Approx decimal 2 places.", "gold": 0.95, "tol": 1.10},
    {"prompt": "Roll 6-sided die 6 times. P(see all 6 faces)? Decimal 3 places.", "gold": 0.0154, "tol": 1.10},
    # Subtle conditional
    {"prompt": "1000 people. 100 sick (test+ 99, test- 1). 900 healthy (test+ 90, test- 810). Of those test-, P(sick)? Decimal 4 places.", "gold": 0.0012, "tol": 1.20},
    {"prompt": "P(A)=0.2, P(B|A)=0.5, P(B|not A)=0.1. P(A|B)? Decimal 3 places.", "gold": 0.556, "tol": 1.05},
    {"prompt": "Family has 3 children. At least one is a girl born on a Sunday. P(at least 2 girls)? Decimal 3 places.", "gold": 0.514, "tol": 1.10},
    {"prompt": "Bag: 3 red, 2 blue. Draw 2 without replacement. P(both red)? Decimal 3 places.", "gold": 0.3, "tol": 1.05},
    {"prompt": "Bag: 4 white, 6 black. Draw 3 without replacement. P(at least one white)? Decimal 3 places.", "gold": 0.833, "tol": 1.05},
    # False positive rate puzzles
    {"prompt": "Sensitivity 100%, specificity 95%, prevalence 1%. P(disease | positive)? Decimal 3 places.", "gold": 0.168, "tol": 1.10},
    {"prompt": "Sensitivity 100%, specificity 99%, prevalence 0.1%. P(disease | positive)? Decimal 3 places.", "gold": 0.091, "tol": 1.15},
    {"prompt": "Sensitivity 90%, specificity 90%, prevalence 50%. P(disease | positive)? Decimal 2 places.", "gold": 0.9, "tol": 1.05},
    {"prompt": "If sensitivity = specificity = 99% and prevalence = 50%, P(disease | positive)? Decimal 2 places.", "gold": 0.99, "tol": 1.02},
    {"prompt": "Two independent positive tests of a rare disease (prev 0.1%, each 95% sens, 95% spec). P(disease | both positive)? Decimal 3 places.", "gold": 0.276, "tol": 1.20},
]
for i, p in enumerate(SLICE_BAYESIAN):
    p["pid"] = f"BA_{i:02d}"; p["slice"] = "bayesian"
assert len(SLICE_BAYESIAN) == 50, len(SLICE_BAYESIAN)


# ============================================================
# Slice 2: QUANTIFIER — universal / existential / necessary-vs-sufficient
# ============================================================
SLICE_QUANTIFIER = [
    {"prompt": "All cats are mammals. Some mammals are pets. Therefore all cats are pets. Valid or invalid?", "gold": "invalid"},
    {"prompt": "All birds fly. Penguins are birds. Therefore penguins fly. Valid or invalid?", "gold": "invalid"},
    {"prompt": "All A are B. No B are C. Therefore no A are C. Valid or invalid?", "gold": "valid"},
    {"prompt": "Some A are B. All B are C. Therefore some A are C. Valid or invalid?", "gold": "valid"},
    {"prompt": "If P then Q. Q is true. Therefore P is true. Valid or invalid?", "gold": "invalid"},
    {"prompt": "If P then Q. P is false. Therefore Q is false. Valid or invalid?", "gold": "invalid"},
    {"prompt": "If P then Q. Q is false. Therefore P is false. Valid or invalid?", "gold": "valid"},
    {"prompt": "Either P or Q. P is true. Therefore Q is false. Valid or invalid? (assume inclusive or)", "gold": "invalid"},
    {"prompt": "Either P or Q (exclusive). P is true. Therefore Q is false. Valid or invalid?", "gold": "valid"},
    {"prompt": "Some birds are not penguins. All penguins are birds. Conclude: some birds are penguins. Valid or invalid?", "gold": "invalid"},
    {"prompt": "All squares are rectangles. Some rectangles are not squares. Conclude: some shapes are not squares. Valid or invalid?", "gold": "invalid"},
    {"prompt": "If a number is divisible by 6 then it is even. 8 is even. Is 8 divisible by 6? Answer yes/no.", "gold": "no"},
    {"prompt": "All multiples of 4 are even. 12 is even. Is 12 a multiple of 4? Answer yes/no/cannot determine.", "gold": "yes"},
    {"prompt": "All multiples of 4 are even. 14 is even. Is 14 a multiple of 4? Answer yes/no.", "gold": "no"},
    {"prompt": "Every even integer >2 is the sum of two primes (assume Goldbach). 100 is even. Is 100 the sum of two primes? Answer yes/no.", "gold": "yes"},
    {"prompt": "No prime greater than 2 is even. 9 is not even. Is 9 prime? Answer yes/no.", "gold": "no"},
    {"prompt": "All A imply B. All B imply C. No C is D. Conclude: no A is D. Valid or invalid?", "gold": "valid"},
    {"prompt": "Some A are B. Some B are C. Conclude: some A are C. Valid or invalid?", "gold": "invalid"},
    {"prompt": "All A are B. Some C are A. Conclude: some C are B. Valid or invalid?", "gold": "valid"},
    {"prompt": "No A is B. All C are B. Conclude: no A is C. Valid or invalid?", "gold": "valid"},
    {"prompt": "All flowers are plants. No plant is an animal. Conclude: no flower is an animal. Valid or invalid?", "gold": "valid"},
    {"prompt": "Some doctors are wealthy. All teachers are educated. Conclude: some teachers are wealthy. Valid or invalid?", "gold": "invalid"},
    {"prompt": "If you study hard, you will pass. You did not pass. Did you study hard? yes/no/cannot determine.", "gold": "no"},
    {"prompt": "If you study hard, you will pass. You passed. Did you study hard? yes/no/cannot determine.", "gold": "cannot determine"},
    {"prompt": "Only seniors can vote. Bob can vote. Is Bob a senior? Answer yes/no.", "gold": "yes"},
    {"prompt": "All seniors can vote. Bob can vote. Is Bob a senior? Answer yes/no/cannot determine.", "gold": "cannot determine"},
    {"prompt": "Only those over 18 are allowed in. Alice is 16. Is Alice allowed? yes/no.", "gold": "no"},
    {"prompt": "It rains only if there are clouds. There are clouds. Will it rain? yes/no/cannot determine.", "gold": "cannot determine"},
    {"prompt": "Necessary condition for fire: oxygen. There is oxygen. Is there fire? yes/no/cannot determine.", "gold": "cannot determine"},
    {"prompt": "Sufficient condition for ignition: spark + fuel + oxygen. All three present. Will it ignite? yes/no.", "gold": "yes"},
    {"prompt": "All elephants in the room are pink. The room is empty. How many pink elephants are in the room?", "gold": 0},
    {"prompt": "Every unicorn in this field is purple. The field has no unicorns. Are there any non-purple unicorns in the field? yes/no.", "gold": "no"},
    {"prompt": "Every student passed at least one test. Are there students who failed every test? yes/no.", "gold": "no"},
    {"prompt": "Every student passed at least one test. Did every student pass every test? yes/no.", "gold": "no"},
    {"prompt": "All birds in the cage are red. The cage contains 0 birds. Is the statement 'all birds in the cage are red' true? yes/no.", "gold": "yes"},
    {"prompt": "Some students are athletes. All athletes are healthy. Conclude: some students are healthy. Valid or invalid?", "gold": "valid"},
    {"prompt": "All A imply B. B is necessary for C. Conclude: A is necessary for C. Valid or invalid?", "gold": "invalid"},
    {"prompt": "All A imply B. B is sufficient for C. Conclude: A is sufficient for C. Valid or invalid?", "gold": "valid"},
    {"prompt": "If x is even, then x^2 is even. 9^2 = 81. Is 9 even? yes/no.", "gold": "no"},
    {"prompt": "If x is even, then x^2 is even. x^2 = 36. Is x even? yes/no/cannot determine.", "gold": "cannot determine"},
    {"prompt": "x is divisible by 12 iff x is divisible by 3 and 4. x is divisible by 3 and 4. Is x divisible by 12? yes/no.", "gold": "yes"},
    {"prompt": "All natural numbers > 2 are composite or prime. 5 is > 2 and prime. Is 5 composite? yes/no.", "gold": "no"},
    {"prompt": "P implies Q. Q implies R. P. Conclude: R. Valid or invalid?", "gold": "valid"},
    {"prompt": "P implies Q. Q implies R. ~R. Conclude: ~P. Valid or invalid?", "gold": "valid"},
    {"prompt": "All birds have wings. All things with wings can fly. Are all birds able to fly? Answer yes/no based on premises only.", "gold": "yes"},
    {"prompt": "All eels are fish. No fish breathe air. Are eels air-breathers? yes/no.", "gold": "no"},
    {"prompt": "Some A are not B. All C are B. Conclude: some A are not C. Valid or invalid?", "gold": "valid"},
    {"prompt": "All squares have 4 equal sides. This shape has 4 equal sides. Is this shape a square? yes/no/cannot determine.", "gold": "cannot determine"},
    {"prompt": "All A are B; all B are C; all C are D. Conclude: all A are D. Valid or invalid?", "gold": "valid"},
    {"prompt": "If P xor Q, then R. P is true, Q is true. Is R true? yes/no.", "gold": "no"},
]
for i, p in enumerate(SLICE_QUANTIFIER):
    p["pid"] = f"QU_{i:02d}"; p["slice"] = "quantifier"
assert len(SLICE_QUANTIFIER) == 50, len(SLICE_QUANTIFIER)


# ============================================================
# Slice 3: MULTISTEP — multi-operation arithmetic with traps
# ============================================================
SLICE_MULTISTEP = [
    {"prompt": "$1000 invested at 5% annually compounded for 3 years, then 7% for 2 years. Final amount, rounded to integer dollars.", "gold": 1326, "tol": 1.01},
    {"prompt": "$2000 at 4% for 5 years compounded. Final amount integer.", "gold": 2433, "tol": 1.01},
    {"prompt": "Item costs $200. 30% off, then 20% off the discounted price. Final price.", "gold": 112, "tol": 1.02},
    {"prompt": "Item costs $200. 20% off, then 30% off the discounted price. Final price.", "gold": 112, "tol": 1.02},
    {"prompt": "Phone $1000. 15% discount, then 8% sales tax on the discounted price. Final price, integer.", "gold": 918, "tol": 1.02},
    {"prompt": "$5000 loan at 6% annually, paid back in 4 equal annual installments. Total interest paid (rounded to integer).", "gold": 644, "tol": 1.05},
    {"prompt": "$10000 loan at 5% annual, 5 equal annual installments. Total interest, integer.", "gold": 1310, "tol": 1.05},
    {"prompt": "Investment $5000 grows 10%/yr. After 5 years, withdraw $2000. Continues at 10%/yr for 3 more years. Final amount, integer.", "gold": 8051, "tol": 1.02},
    {"prompt": "Investment $1000 grows 8%/yr. After 4 years, deposit additional $500. Continues at 8%/yr for 3 more. Final amount, integer.", "gold": 2342, "tol": 1.02},
    {"prompt": "$10000 grows 6%/yr first 3 years, then loses 5%/yr next 2 years. Final amount, integer.", "gold": 10752, "tol": 1.02},
    {"prompt": "Factory: 3 lines. A: 100 units/h, 5% defect. B: 80/h, 3% defect. C: 120/h, 7% defect. After 8 hours total good units?", "gold": 2360, "tol": 1.02},
    {"prompt": "Factory: 2 lines. A: 200/h, 4% defect. B: 150/h, 6% defect. After 10h total good units?", "gold": 3330, "tol": 1.02},
    {"prompt": "School: 240 students. 60% boys. 40% of boys play football. 30% of girls play. Total football players.", "gold": 86, "tol": 1.05},
    {"prompt": "Office: 300 workers. 55% men. 70% of men commute by car. 40% of women commute by car. Total car commuters.", "gold": 169, "tol": 1.05},
    {"prompt": "Salary $80000. After-tax 25%. Save 30% of take-home. Annual savings, integer.", "gold": 18000, "tol": 1.02},
    {"prompt": "Salary $50000. After-tax 22%. Save 25% of take-home, then add a $1000 year-end bonus saving. Annual savings, integer.", "gold": 10750, "tol": 1.02},
    {"prompt": "Train A leaves at 9am at 60 km/h. Train B leaves at 10am at 90 km/h, same direction. When does B catch A? Format HH:MM.", "gold": "12:00"},
    {"prompt": "Train A leaves at 8am at 50 km/h east. Train B leaves at 9am at 75 km/h east, same direction same point. Catch time HH:MM.", "gold": "11:00"},
    {"prompt": "Two trains leave same station: A east at 60 km/h at 9am, B west at 80 km/h at 10am. When are they 220 km apart? HH:MM.", "gold": "11:00"},
    {"prompt": "Cylinder r=5 h=10. Sphere r=3. (Volume cylinder) - (volume sphere), use pi=3.14, rounded integer.", "gold": 672, "tol": 1.02},
    {"prompt": "Cone r=4 h=9. Sphere r=2. (Volume cone) - (volume sphere), pi=3.14, rounded integer.", "gold": 117, "tol": 1.02},
    {"prompt": "Rectangle: perimeter 40, area 96. Find length and width. Sum L+W.", "gold": 20, "tol": 1.001},
    {"prompt": "Rectangle: area 60, length is 4 more than width. What is the width?", "gold": 6, "tol": 1.01},
    {"prompt": "Mix 12 oz coffee that is 80% water with 4 oz pure water. New % water (integer).", "gold": 85, "tol": 1.02},
    {"prompt": "Mix 20 oz solution 30% salt with 10 oz solution 50% salt. New % salt, decimal 1 place.", "gold": 36.7, "tol": 1.02},
    {"prompt": "Mix 5 L of 10% acid with 15 L of 30% acid. New % acid (decimal 1 place).", "gold": 25.0, "tol": 1.02},
    {"prompt": "Box: 12 ft x 8 ft x 6 ft. Each small cube = 8 cu ft. How many small cubes fit? Integer.", "gold": 72, "tol": 1.001},
    {"prompt": "Wall: 20 ft x 8 ft. Each tile is 1 ft x 2 ft. How many tiles needed? Integer.", "gold": 80, "tol": 1.001},
    {"prompt": "Tank A holds 100L, drains at 5 L/min. Tank B 50L, fills at 3 L/min. After how many minutes is total water in both tanks equal? Decimal.", "gold": 6.25, "tol": 1.05},
    {"prompt": "Tank A 200L drains at 8 L/min. Tank B 0L fills at 4 L/min. When are both equal? Decimal minutes.", "gold": 16.67, "tol": 1.05},
    {"prompt": "5 workers do a job in 12 days. After 4 days, 2 more workers join. Total days from start? Decimal.", "gold": 9.71, "tol": 1.05},
    {"prompt": "8 workers do a job in 15 days. After 5 days, 4 leave. Total days from start? Decimal.", "gold": 25.0, "tol": 1.05},
    {"prompt": "Car uses 8 L/100km city, 6 L/100km highway. Trip: 50 km city + 200 km highway. Total fuel, L (integer).", "gold": 16, "tol": 1.05},
    {"prompt": "Car uses 10 L/100km city, 7 L/100km highway. Trip: 80 km city + 300 km highway. Total fuel, L (integer).", "gold": 29, "tol": 1.05},
    {"prompt": "Pipe A fills tank in 6h. Pipe B fills in 4h. Pipe C drains in 12h. All open. How many hours to fill? Decimal 2 places.", "gold": 3.0, "tol": 1.05},
    {"prompt": "Pipe A fills in 3h. Pipe B in 5h. Both open. How many hours to fill? Decimal 2 places.", "gold": 1.875, "tol": 1.05},
    {"prompt": "$1000 grows 8% in year 1, loses 6% in year 2, gains 5% in year 3. Final, integer.", "gold": 1067, "tol": 1.02},
    {"prompt": "Stock $50 → +20% → -10% → +15%. Final price, decimal.", "gold": 62.10, "tol": 1.02},
    {"prompt": "Athletes: 200 total. 60% run, 40% swim, 25% do both. How many do exactly one of the two?", "gold": 110, "tol": 1.05},
    {"prompt": "Class: 50 students. 30 like math, 25 like science, 12 like both. How many like neither?", "gold": 7, "tol": 1.05},
    {"prompt": "Trip cost $1500 split among 5 friends. One backs out, splitting cost among 4. New per-person cost - original? Integer dollars.", "gold": 75, "tol": 1.05},
    {"prompt": "30 people share a $600 prize equally. 10 backout. New per-person amount - original? Integer dollars.", "gold": 10, "tol": 1.05},
    {"prompt": "Population grows from 1000 to 1331 in 3 years (compound). Annual growth rate, integer percent.", "gold": 10, "tol": 1.05},
    {"prompt": "Population grows from 500 to 605 in 2 years (compound). Annual rate, integer percent.", "gold": 10, "tol": 1.05},
    {"prompt": "Bank: $500 at 4% annual compounded quarterly for 1 year. Final, decimal 2 places.", "gold": 520.30, "tol": 1.005},
    {"prompt": "Bank: $1000 at 6% annual compounded monthly for 1 year. Final, decimal 2 places.", "gold": 1061.68, "tol": 1.005},
    {"prompt": "Distance 240 km. Speed up by 20 km/h, save 1 hour. Original speed, integer km/h.", "gold": 60, "tol": 1.05},
    {"prompt": "Distance 360 km. Speed up by 15 km/h, save 1 hour. Original speed, integer.", "gold": 60, "tol": 1.05},
    {"prompt": "Profit margin: cost $80, sell $100. Margin %, integer.", "gold": 20, "tol": 1.05},
    {"prompt": "Markup: cost $80, sell $100. Markup %, integer.", "gold": 25, "tol": 1.05},
]
for i, p in enumerate(SLICE_MULTISTEP):
    p["pid"] = f"MS_{i:02d}"; p["slice"] = "multistep"
assert len(SLICE_MULTISTEP) == 50, len(SLICE_MULTISTEP)


# ============================================================
# Slice 4: CONSTRAINT — knights/knaves, scheduling, simultaneous constraints
# ============================================================
SLICE_CONSTRAINT = [
    {"prompt": "Knights tell truth, knaves lie. A says 'B is a knight'. B says 'A is a knave'. Identify A and B.", "gold": "A knave, B knave"},
    {"prompt": "A says 'we are both knaves'. What is A?", "gold": "knave"},
    {"prompt": "A says 'I am a knight'. What can we conclude?", "gold": "cannot determine"},
    {"prompt": "A says 'B and C are both knaves'. B says 'A is a knight'. C says nothing. If exactly one is a knight, who?", "gold": "B"},
    {"prompt": "A,B,C: each says 'the next is a knave'. (A about B, B about C, C about A). How many knights are there?", "gold": 0},
    {"prompt": "A says 'B lies'. B says 'C lies'. C says 'A and B both tell truth'. Liars are 1 or 3? Answer integer.", "gold": 1},
    {"prompt": "5 hats: 3 red, 2 blue. 3 prisoners in line. P1 (back) sees P2,P3. P2 sees P3. P3 sees nothing. P1 doesn't know. P2 doesn't know. What hat is P3 wearing?", "gold": "red"},
    {"prompt": "3 boxes, exactly 1 has gold. Box A: 'B has gold'. Box B: 'this box is empty'. Box C: 'B has gold'. Exactly one statement is true. Which box has gold?", "gold": "A"},
    {"prompt": "3 sisters: youngest is best painter. Middle is best singer. Oldest is best dancer. Beth older than the painter. Carla younger than the dancer. Beth not the singer. What is Beth best at?", "gold": "dancer"},
    {"prompt": "3 friends Alice, Bob, Carol sit in a row. Alice is not next to Bob. Carol is to the right of Alice. Order from left to right.", "gold": "A C B"},
    {"prompt": "4 people sit in a row: P, Q, R, S. P is not first. R is to the left of S. Q is in position 2. Order from position 1.", "gold": "R Q S P"},
    {"prompt": "Numbers 1-9 placed in a 3x3 grid. Sum of each row=15, each column=15, each diagonal=15. What number is in the center?", "gold": 5},
    {"prompt": "x in {1,...,20}: x^2 - 5x + 6 = 0. Find x.", "gold": 2},
    {"prompt": "x in {1,...,30}: x is prime AND x^2 + 1 is prime. Find smallest x > 2.", "gold": 4, "tol": 0.5},
    {"prompt": "Set: {1,2,3,4,5}. Pick 3 numbers summing to 9. How many such subsets?", "gold": 2},
    {"prompt": "Set: {1,2,3,4,5,6}. Pick 3 numbers summing to 10. Count subsets.", "gold": 3},
    {"prompt": "5 people seated at a round table. Alice not next to Bob, Carol not next to Dave. Eve must be between two non-friends. Who can be next to Eve?", "gold": "any"},
    {"prompt": "8x8 chessboard with opposite corners removed. Can it be tiled with 1x2 dominoes? yes/no.", "gold": "no"},
    {"prompt": "5 boxes labeled A-E. Each has 1 prize hidden. Clues: prize in B is left of prize in D; prize in A is at one end; E is at the other end. Position 3 (middle)?", "gold": "C"},
    {"prompt": "A says: 'I am a knave or B is a knight.' What is A?", "gold": "knight"},
    {"prompt": "A says: 'B is a knight iff C is a knave'. B says 'A is a knight'. C says 'A is a knave'. Who is which? (Answer: A type, B type, C type)", "gold": "A knight, B knight, C knave"},
    {"prompt": "100 students in 4 clubs. 3 clubs have 30, 25, 20 members. 4th club has at least 10 members. What's the minimum total membership across all clubs (each student counted once per club)?", "gold": 100, "tol": 1.05},
    {"prompt": "Every student passed at least 1 test. No student passed all 4 tests. Min number of students?", "gold": 1},
    {"prompt": "10 socks in a drawer: 5 black, 5 white, all unpaired. Min you must draw to guarantee a matching pair?", "gold": 3},
    {"prompt": "Drawer: 5 red, 5 blue, 5 green socks. Min draw to guarantee 2 matching?", "gold": 4},
    {"prompt": "Drawer: 5 red, 5 blue, 5 green. Min to guarantee 2 of same color, none red?", "gold": 8},
    {"prompt": "5 lamps in a row. Each can be on or off. Lamp i is on iff lamp i-1 was off (lamp 1 always on). State of lamps 1-5?", "gold": "1 0 1 0 1"},
    {"prompt": "Among 5 students each with a unique color, A=red, B=blue, C wears not green, D doesn't wear yellow, E wears purple. The 5 colors are red,blue,green,yellow,purple. C wears? (single color)", "gold": "yellow"},
    {"prompt": "If A then B. If B then C. If C then D. ~D. Conclude about A.", "gold": "not A"},
    {"prompt": "12 coins, 1 is fake (lighter). Min weighings on a balance to find it?", "gold": 3},
    {"prompt": "9 coins, 1 is heavier. Min balance weighings?", "gold": 2},
    {"prompt": "Cross a river: man, wolf, goat, cabbage. Boat takes man + 1. Wolf+goat or goat+cabbage cannot be left alone. Min trips needed (one direction = 1 trip)?", "gold": 7},
    {"prompt": "3 missionaries 3 cannibals cross river. Boat 2 max. Cannibals can't outnumber missionaries on either bank. Min one-way trips?", "gold": 11},
    {"prompt": "Tower of Hanoi, 3 disks. Min moves?", "gold": 7},
    {"prompt": "Tower of Hanoi, 5 disks. Min moves?", "gold": 31},
    {"prompt": "Sudoku constraint: in a row, digits 1-9 appear exactly once. Given a row [_,2,_,4,_,6,_,8,_], the missing digits in order from left are: (5 numbers separated by spaces)", "gold": "1 3 5 7 9"},
    {"prompt": "Magic constant for a 4x4 magic square using 1-16: ?", "gold": 34},
    {"prompt": "Number of solutions to x+y+z=10, x,y,z non-negative integers?", "gold": 66},
    {"prompt": "Number of ways to seat 4 people at a round table (rotations equivalent)?", "gold": 6},
    {"prompt": "Truth table: A→B has how many TRUE rows out of 4?", "gold": 3},
    {"prompt": "5 prisoners with hats from {red,blue}. Each sees others, not own. Whisper allowed before hats placed. They must guess their own. Strategy guaranteeing at least 1 correct: max correct guaranteed?", "gold": 1},
    {"prompt": "Birthday cake to 7 people in 3 cuts. Each cut is straight. Max possible pieces?", "gold": 7},
    {"prompt": "12 balls, 1 is different weight (could be heavier or lighter). Min balance weighings to identify?", "gold": 3},
    {"prompt": "5 houses in a row, each different color. Yellow is left of green. Red is at position 3. Blue is not adjacent to red. White is at position 1. Position of blue?", "gold": 5},
    {"prompt": "A,B,C,D,E enter a race. A finishes before C, D before A, E before B, B before D. Order of finish (winner first)?", "gold": "E B D A C"},
    {"prompt": "Find x in {2,3,5,7,11,13}: x is prime AND x+2 is prime AND x+4 is prime. (Answer: smallest such x or 'none')", "gold": "3"},
    {"prompt": "5x5 grid: place 5 non-attacking rooks. How many ways?", "gold": 120},
    {"prompt": "8 queens on 8x8 board, none attack. Number of solutions?", "gold": 92},
    {"prompt": "How many ways to color a 2x2 grid with 3 colors so no two adjacent cells share a color?", "gold": 18},
    {"prompt": "Path on 3x3 grid from top-left to bottom-right, only right or down moves. Number of distinct paths?", "gold": 6},
]
for i, p in enumerate(SLICE_CONSTRAINT):
    p["pid"] = f"CS_{i:02d}"; p["slice"] = "constraint"
assert len(SLICE_CONSTRAINT) == 50, len(SLICE_CONSTRAINT)


# ============================================================
# Slice 5: COUNTERFACTUAL — hidden assumptions, narrative traps, lateral
# ============================================================
SLICE_COUNTERFACTUAL = [
    {"prompt": "A man builds a house with all four walls facing south. He sees a bear. What color is the bear?", "gold": "white"},
    {"prompt": "There are 30 cows in a field. 28 chickens. How many didn't?", "gold": 10},
    {"prompt": "A truck driver going down a one-way street the wrong way passes 10 police cars without being stopped. Why?", "gold": "walking"},
    {"prompt": "A doctor tells you, 'every Sunday I shave myself only.' Are you a man or a woman? (paradox)", "gold": "woman"},
    {"prompt": "Mary's father has 5 daughters: Nana, Nene, Nini, Nono, and ___?", "gold": "Mary"},
    {"prompt": "I went to the store to buy 6 apples and 4 bananas. On the way back I dropped some. Now I have 4 apples and 3 bananas. How many did I drop?", "gold": 3},
    {"prompt": "If you have only one match in a freezing cabin with a candle, an oil lamp, and a wood stove, which do you light first?", "gold": "match"},
    {"prompt": "What 5-letter word becomes shorter when you add 2 letters to it?", "gold": "short"},
    {"prompt": "A man pushes his car to a hotel and tells the owner he is bankrupt. Why? (1 word)", "gold": "Monopoly"},
    {"prompt": "Forward I am heavy, backward I am not. What am I?", "gold": "ton"},
    {"prompt": "Two boxers in a championship fight. Round 5, neither has thrown a punch. Why? (1 sentence)", "gold": "women"},
    {"prompt": "A man went outside in the rain with no umbrella and no hat, but not a hair on his head got wet. How?", "gold": "bald"},
    {"prompt": "If you see a stationary clock at 3 o'clock, what hand might NOT be there? (1 word)", "gold": "second"},
    {"prompt": "A rope ladder hangs from a ship. Each rung is 30 cm apart. There are 10 rungs and the lowest is just above sea level at low tide. Tide rises 1.5 m. How many rungs are now under water?", "gold": 0},
    {"prompt": "The day before yesterday I was 25, next year I'll be 28. Today is January 1. What day was my birthday?", "gold": "December 31"},
    {"prompt": "What gets wetter the more it dries?", "gold": "towel"},
    {"prompt": "I have keys but no doors, space but no rooms. What am I?", "gold": "keyboard"},
    {"prompt": "What goes up but never comes down?", "gold": "age"},
    {"prompt": "What 4-letter word can be written forward, backward, and upside down, and still be read from left to right?", "gold": "noon"},
    {"prompt": "What disappears when you say its name?", "gold": "silence"},
    {"prompt": "A boy and his father are in a car accident. The father dies. The boy is rushed to hospital. The surgeon says 'I cannot operate, this is my son.' How? (1 word)", "gold": "mother"},
    {"prompt": "There is a clear glass jar with a marble inside. The lid is sealed. How do you remove the marble without breaking jar or lid?", "gold": "no way"},
    {"prompt": "A horse is tied to a 5m rope but reaches food 10m away. How?", "gold": "rope not tied to anything"},
    {"prompt": "What runs but never walks, has a mouth but never talks?", "gold": "river"},
    {"prompt": "Two coins sum to $0.30. One is not a quarter. What are the coins?", "gold": "quarter and nickel"},
    {"prompt": "A barber shaves all men in town who do not shave themselves. Does the barber shave himself? Yes/no/paradox.", "gold": "paradox"},
    {"prompt": "Three men split a $30 hotel bill. Manager says room only $25, gives $5 back via bellhop. Bellhop pockets $2, gives each man $1. Each man paid $9, total $27, plus $2 = $29. Where's the $1? (1 word)", "gold": "wrong"},
    {"prompt": "A man lives on the 20th floor. He takes elevator down every day. Coming back, he takes elevator to 10th floor and walks up — except on rainy days, when he takes elevator all the way. Why?", "gold": "short"},
    {"prompt": "What word in English is always pronounced incorrectly?", "gold": "incorrectly"},
    {"prompt": "Take 1000. Add 40. Add 1000. Add 30. Add 1000. Add 20. Add 1000. Add 10. What's the total?", "gold": 4100},
    {"prompt": "If you're running in a marathon and you overtake the second-place runner, what place are you in?", "gold": "second"},
    {"prompt": "A doctor in a remote village has 10 patients with a rare disease. He has 10 doses of medicine. He gives 5 doses to one patient. How many can he treat now?", "gold": 6, "tol": 0.5},
    {"prompt": "A snail climbs 3m up a 10m wall during the day, slides 2m down at night. How many days to reach top?", "gold": 8},
    {"prompt": "Two trains 200 km apart, approach at 50 km/h each. A bird flies between them at 75 km/h until they meet. Total distance bird flies?", "gold": 150, "tol": 1.05},
    {"prompt": "A man is found hanging in a sealed room, no chair, just water on the floor. How? (1 word)", "gold": "ice"},
    {"prompt": "What can travel around the world while staying in a corner?", "gold": "stamp"},
    {"prompt": "If 1=3, 2=3, 3=5, 4=4, 5=4, then 6=?", "gold": 3},
    {"prompt": "A man weighs 200 lbs but jumps from a plane and survives without parachute. How?", "gold": "didn't die"},
    {"prompt": "If a hen and a half lays an egg and a half in a day and a half, how many eggs do 6 hens lay in 6 days?", "gold": 24},
    {"prompt": "What goes through a forest but never moves? (1 word)", "gold": "path"},
    {"prompt": "I am taken from a mine and shut up in a wooden case. What am I?", "gold": "pencil"},
    {"prompt": "What has a head, a tail, but no body?", "gold": "coin"},
    {"prompt": "If you drop a yellow hat in the Red Sea, what does it become?", "gold": "wet"},
    {"prompt": "A cowboy rides into town on Friday, stays 3 days, leaves on Friday. How? (1 word)", "gold": "horse"},
    {"prompt": "What can you catch but not throw?", "gold": "cold"},
    {"prompt": "I speak without a mouth and hear without ears. I have no body, but come alive with the wind. What am I?", "gold": "echo"},
    {"prompt": "What has many keys but cannot open a single lock?", "gold": "piano"},
    {"prompt": "A man dies of old age on his 25th birthday. How is this possible? (1 word)", "gold": "leap"},
    {"prompt": "How many months have 28 days?", "gold": 12},
    {"prompt": "If a plane crashes on the border of US and Canada, where do they bury the survivors?", "gold": "you don't bury survivors"},
]
for i, p in enumerate(SLICE_COUNTERFACTUAL):
    p["pid"] = f"CF_{i:02d}"; p["slice"] = "counterfactual"
assert len(SLICE_COUNTERFACTUAL) == 50, len(SLICE_COUNTERFACTUAL)


ALL_SLICES = {
    "bayesian": SLICE_BAYESIAN,
    "quantifier": SLICE_QUANTIFIER,
    "multistep": SLICE_MULTISTEP,
    "constraint": SLICE_CONSTRAINT,
    "counterfactual": SLICE_COUNTERFACTUAL,
}


def all_problems():
    out = []
    for slice_name, ps in ALL_SLICES.items():
        out.extend(ps)
    return out


if __name__ == "__main__":
    n_total = sum(len(v) for v in ALL_SLICES.values())
    print(f"Total problems: {n_total}")
    for k, v in ALL_SLICES.items():
        print(f"  {k}: {len(v)}")
