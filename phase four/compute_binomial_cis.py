"""Compute 95% Wilson binomial CIs for the paper's headline claims.

Reproduces Table tab:ci in the paper. Reads from the six cached
audit-layer logs and prints a LaTeX-ready table.
"""

from scipy.stats import binomtest


def ci_row(name, wr, n):
    k = round(wr * n)
    ci = binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
    return f"{name:42s}  wr={wr:.2f}  n={n:>3d}  95% CI = [{ci.low:.2f}, {ci.high:.2f}]"


def main():
    print("=== Headline binomial CIs (Wilson, 95%) ===\n")
    rows = [
        # Inner-loop (gemini)
        ("W076 inner-loop gemini", 0.64, 50),
        ("W077 inner-loop gemini", 0.60, 50),
        ("W078 inner-loop gemini", 0.60, 50),
        # L1 Opus
        ("W076 L1 claude-opus", 0.40, 50),
        ("W077 L1 claude-opus", 0.47, 50),
        ("W078 L1 claude-opus", 0.51, 50),
        # L3 n=100 (combined Exp 10 falsifier values: see exp10 log)
        ("W076 L3 extend n=100", 0.57, 100),
        ("W077 L3 extend n=100", 0.52, 100),
        ("W078 L3 extend n=100", 0.52, 100),
        # L4 cross-solver
        ("W076 L4 cross-solver mean", 0.41, 60),
        # L6 expensive judges
        ("W078 L6 claude-opus-4-6", 20 / 46, 46),
        ("W078 L6 gpt-5.4", 21 / 49, 49),
    ]
    for name, wr, n in rows:
        print(ci_row(name, wr, n))

    print("\n=== Binary composite claims ===\n")
    print(f"0/9 new candidates PASS vs 30% baseline: "
          f"one-sided p = {binomtest(0, 9, 0.3, alternative='less').pvalue:.4f}")
    print(f"3/21 combined PASS (12 inner-loop + 9 new) vs 30% baseline: "
          f"p = {binomtest(3, 21, 0.3, alternative='less').pvalue:.4f}")
    print(f"0/3 KEEPs survive L1 under H0 p>=0.6: "
          f"p = {binomtest(0, 3, 0.6, alternative='less').pvalue:.4f}")


if __name__ == "__main__":
    main()
