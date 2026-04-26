Below are candidate Exp 71 domains where the missing ingredient is plausibly *procedural methodology*, not “try harder.”

## Candidate domains

### 1. Hamming(7,4) syndrome decoding  
**Class:** Decode a received 7-bit Hamming codeword with at most one flipped bit; output the 4 data bits.  
**Why baseline likely fails:** LLMs may have seen “Hamming code” facts, but exact parity convention, syndrome orientation, and correction step are brittle. Many models confound bit positions and syndrome order.  
**Why generic warning won’t help:** The missing piece is an algorithm: compute three parity checks, interpret syndrome as an error position, flip, extract data bits.  
**Wisdom:** “For positions 1–7, parity bits are 1,2,4. Even checks: s1 over {1,3,5,7}, s2 over {2,3,6,7}, s4 over {4,5,6,7}. Syndrome value = s1 + 2s2 + 4s4. If nonzero, flip that position. Output positions 3,5,6,7.”  
**Worked example:** Received `0110111`. Checks: s1 over 1,3,5,7 = 0⊕1⊕1⊕1 = 1; s2 over 2,3,6,7 = 1⊕1⊕1⊕1 = 0; s4 over 4,5,6,7 = 0⊕1⊕1⊕1 = 1. Syndrome = 1+0+4=5, so flip bit 5: `0110011`. Data positions 3,5,6,7 = `1011`.

---

### 2. Sprague-Grundy / nimber evaluation for impartial games  
**Class:** Normal-play impartial games composed of several independent piles/subgames; determine FIRST/SECOND winner.  
**Why baseline likely fails:** LLMs often know Nim superficially but fail on non-Nim subtraction games, especially with large heap sizes requiring recurrence/periodicity.  
**Why generic warning won’t help:** Carefulness does not supply mex values, xor composition, or periodic reduction.  
**Wisdom:** “For each heap size n, compute Grundy value g(n)=mex({g(n−m): legal move m≤n}). A sum of independent games is losing iff xor of all g-values is 0. For subtraction set {1,3,4}, the sequence is periodic mod 7: n mod 7 → g: 0→0, 1→1, 2→0, 3→1, 4→2, 5→3, 6→2.”  
**Worked example:** Piles 10,24,28. Residues mod 7 are 3,3,0, so g-values are 1,1,0. XOR = 1⊕1⊕0 = 0, so SECOND wins.

---

### 3. Absorbing Markov chain “first-step equations”  
**Class:** Given a small stochastic process, compute probability of eventually hitting state A before B.  
**Why baseline likely fails:** Models often simulate intuitively or average paths incorrectly.  
**Why generic warning won’t help:** Need to set up linear equations for hitting probabilities.  
**Wisdom:** Define h(s)=Pr(hit target before failure from s). Boundary states have h=1 or 0. For transient states, h(s)=Σ P(s→t)h(t). Solve linear system.  
**Worked example:** A two-state loop where h(X)=0.5h(X)+0.5h(Y), h(Y)=0.25+0.75h(X), solve.

---

### 4. Inclusion-exclusion with forbidden overlaps  
**Class:** Count strings, arrangements, or selections avoiding multiple overlapping forbidden events.  
**Why baseline likely fails:** LLMs overcount/undercount overlaps.  
**Why generic warning won’t help:** Need systematic inclusion-exclusion or automaton DP.  
**Wisdom:** Define events, subtract singles, add pairwise intersections, etc.; for repeated substrings use state automata.

---

## Strongest candidates for Exp 71

I would start with **Hamming(7,4)** and **Sprague-Grundy subtraction games**. Both are established human-taught methods, objectively scorable, compactly teachable, and likely to show low baseline on fresh instances.

---

# Sample problems: Hamming(7,4)

Use convention: positions 1–7; parity bits 1,2,4; even parity; data bits are positions 3,5,6,7. One bit may be corrupted. Decode the 4 data bits.

1. Received: `0110111` → **1011**  
2. Received: `0000101` → **0101**  
3. Received: `0010111` → **1110**  
4. Received: `0001000` → **0000**  
5. Received: `0010101` → **1101**  
6. Received: `0001101` → **0111**  
7. Received: `1110000` → **1000**  
8. Received: `1101010` → **0010**

---

# Sample problems: Sprague-Grundy subtraction game

Game: There are three independent piles. On a turn, choose one pile and remove exactly 1, 3, or 4 counters. Normal play: player unable to move loses. Return FIRST if the first player has a winning strategy, else SECOND.

For this game, g(n) by n mod 7 is:  
0→0, 1→1, 2→0, 3→1, 4→2, 5→3, 6→2.

1. Piles: 17, 24, 35 → **SECOND**  
2. Piles: 19, 26, 28 → **SECOND**  
3. Piles: 20, 25, 31 → **FIRST**  
4. Piles: 18, 23, 30 → **FIRST**  
5. Piles: 15, 16, 21 → **FIRST**  
6. Piles: 27, 34, 41 → **FIRST**  
7. Piles: 10, 24, 28 → **SECOND**  
8. Piles: 12, 33, 47 → **FIRST**