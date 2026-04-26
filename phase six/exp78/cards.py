"""Exp 78 procedural wisdom cards.

Each card targets one specific bug pattern. Card includes:
  - name: short id
  - pattern: matches problem.bug_patterns tag (one card -> one pattern)
  - trigger_keywords: regex/substring patterns that, if present in the task
                      prompt, indicate the wisdom should fire
  - procedure: the stepwise procedure (this is what we test if matters)
  - worked_example: a brief worked example
"""

CARDS = [
    {
        "name": "mut_default",
        "pattern": "MUT_DEFAULT",
        "title": "Mutable default arguments are shared across calls",
        "trigger_keywords": ["default", "log=", "registry=", "counts=", "scores=",
                                "optional", "optional dict", "optional list"],
        "trigger_description": "The function has a parameter with a mutable default (list/dict/set), e.g. `def f(..., x=[])` or `def f(..., x={})`.",
        "failure_to_avoid": "Using a mutable literal as a default value. The default object is created once at function definition and shared across all calls without an explicit argument, leaking state.",
        "procedure": [
            "Default the parameter to None: `def f(..., x=None)`.",
            "Inside the function, write `if x is None: x = []` (or `{}`).",
            "Then proceed with x as a fresh local object.",
            "Return that fresh object so each call without explicit argument is independent.",
        ],
        "verification": "Call the function twice without the optional argument and confirm the two return values are not the same object and do not share state.",
        "worked_example": (
            "BAD:\n"
            "  def append_to(value, lst=[]):\n"
            "      lst.append(value)\n"
            "      return lst\n"
            "  # append_to(1) -> [1], append_to(2) -> [1,2]  ← shared!\n\n"
            "GOOD:\n"
            "  def append_to(value, lst=None):\n"
            "      if lst is None:\n"
            "          lst = []\n"
            "      lst.append(value)\n"
            "      return lst\n"
            "  # append_to(1) -> [1], append_to(2) -> [2]  ✓"
        ),
    },
    {
        "name": "slice_boundary",
        "pattern": "SLICE_BOUNDARY",
        "title": "Slice / range / index boundary cases (especially zero and length)",
        "trigger_keywords": ["last n", "first n", "every k", "kth", "slice",
                                "n elements", "indices"],
        "trigger_description": "The function does positional slicing or index arithmetic that depends on a length, count, or step. Edge cases at 0 and at the boundary need to be checked.",
        "failure_to_avoid": "Using `lst[-n:]` when n could be 0 (returns the entire list, not empty). Using `range(start, stop)` and forgetting that stop is exclusive. Off-by-one in stride/step arithmetic.",
        "procedure": [
            "Identify what the function should return when n=0 or the input is empty.",
            "If the spec says n=0 → empty result, do not use `lst[-n:]` (which is `lst[0:]` = whole list); use an explicit `if n == 0: return []` guard.",
            "For range/step expressions, manually trace one or two boundary inputs (n=0, n=1, n=len) before submitting.",
            "After writing the function, run through each spec example mentally — including the edge cases — and confirm the result.",
        ],
        "verification": "For every boundary case the spec mentions (n=0, empty input, single element, n=length), explicitly trace the function and check the output.",
        "worked_example": (
            "Task: return last n elements; n=0 should give [].\n\n"
            "BAD: `return lst[-n:]`  # n=0 -> lst[0:] -> entire list (wrong)\n\n"
            "GOOD:\n"
            "  if n == 0: return []\n"
            "  return lst[-n:] if n < len(lst) else lst[:]"
        ),
    },
    {
        "name": "float_eq",
        "pattern": "FLOAT_EQ",
        "title": "Floats are not equal to themselves under arithmetic",
        "trigger_keywords": ["close", "equal", "tolerance", "float", "compare",
                                "0.1", "0.2", "0.3", "decimal"],
        "trigger_description": "The function compares floats for equality, sums them and checks against a target, or otherwise relies on exact float arithmetic.",
        "failure_to_avoid": "Using `==` to compare floats. `0.1 + 0.2 != 0.3` in IEEE 754 (it equals 0.30000000000000004).",
        "procedure": [
            "Never use `a == b` or `a != b` for floats.",
            "Use `math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)` (Python ≥ 3.5).",
            "If you cannot import math, use `abs(a - b) < tol` with a small absolute tol.",
            "When summing many floats, recognize the cumulative error grows roughly as sqrt(n) * machine epsilon.",
        ],
        "verification": "Test with `0.1 + 0.2` vs `0.3`; with a sum of 10 copies of 0.1 vs `1.0`. Both should compare equal under your tolerance.",
        "worked_example": (
            "BAD: `return a == b`  # for a=0.1+0.2, b=0.3 returns False\n\n"
            "GOOD:\n"
            "  import math\n"
            "  return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)"
        ),
    },
    {
        "name": "late_bind",
        "pattern": "LATE_BIND",
        "title": "Closures capture variables by reference, not value",
        "trigger_keywords": ["list of functions", "lambda", "closure", "factory",
                                "make_", "create functions", "list comprehension"],
        "trigger_description": "The function builds a list (or dict) of functions inside a loop or comprehension, where each function references a loop variable.",
        "failure_to_avoid": "`fns = [lambda x: x + i for i in [10, 20, 30]]`. Every lambda captures the SAME `i`, which by the time you call any of them is the last value (30), so all three behave identically.",
        "procedure": [
            "If you build functions in a loop and each needs its own captured value, bind the value as a default argument: `lambda x, i=i: x + i`.",
            "Or use a factory function: `def make(i): return lambda x: x + i`.",
            "Either pattern forces the value to be captured at function-definition time, not at call time.",
        ],
        "verification": "Construct the list, then call funcs[0], funcs[1], funcs[-1] separately and verify each returns the expected value, not all the same.",
        "worked_example": (
            "BAD: `fns = [lambda x: x + i for i in [10, 20, 30]]`\n"
            "  fns[0](1) -> 31  (not 11!)\n\n"
            "GOOD: `fns = [lambda x, i=i: x + i for i in [10, 20, 30]]`\n"
            "  fns[0](1) -> 11  ✓"
        ),
    },
    {
        "name": "copy_semantics",
        "pattern": "COPY_SEMANTICS",
        "title": "Shallow copy shares nested mutable references",
        "trigger_keywords": ["copy", "independent copy", "do not mutate", "do not modify",
                                "without modifying", "fresh list", "fresh dict",
                                "merge", "deep"],
        "trigger_description": "The function returns a copy of a structure that may contain nested mutables (list of lists, dict of lists, etc.), and the caller expects the copy to be fully independent.",
        "failure_to_avoid": "Using `list(x)`, `x[:]`, or `dict(x)` for nested structures — these are shallow. Inner objects are shared. Modifying `copy[0][0]` modifies `x[0][0]`.",
        "procedure": [
            "If the structure has only one level (list of ints, dict of strings), shallow copy via `list(x)` / `dict(x)` / `x.copy()` is fine.",
            "If any value is itself mutable (list of lists, dict of lists), use `copy.deepcopy(x)`.",
            "When merging dicts where values may be lists, deep-copy the source's mutable values into the new dict, not just assign references.",
        ],
        "verification": "After the copy, mutate one nested element of the copy. Confirm the original is unchanged. If you skipped this verification, you have not tested copy semantics.",
        "worked_example": (
            "BAD:\n"
            "  def copy_grid(g):\n"
            "      return list(g)  # shallow! inner lists shared\n\n"
            "GOOD:\n"
            "  import copy\n"
            "  def copy_grid(g):\n"
            "      return copy.deepcopy(g)"
        ),
    },
    {
        "name": "empty_edge",
        "pattern": "EMPTY_EDGE",
        "title": "Empty input and single-element input are separate cases",
        "trigger_keywords": ["empty", "if empty", "or empty", "may be empty",
                                "could be empty", "0 or", "single element",
                                "may have", "list of"],
        "trigger_description": "The function takes a list/dict/iterable that the spec explicitly says may be empty, may have one element, or may have arbitrary length.",
        "failure_to_avoid": "Calling `max([])` (raises). Indexing `lst[0]` without checking. Dividing by len(lst) when len could be 0.",
        "procedure": [
            "Read the spec; identify every case mentioned (empty, single element, length-of-edge).",
            "At the top of the function, write explicit guards for each: `if not lst: return ...`.",
            "Trace the function on len-0 and len-1 inputs before submitting.",
        ],
        "verification": "For each empty/single/edge case the spec mentions, write a one-line dry trace of what the function should return.",
        "worked_example": (
            "BAD:\n"
            "  def safe_max(values):\n"
            "      return max(values)  # raises ValueError on []\n\n"
            "GOOD:\n"
            "  def safe_max(values):\n"
            "      if not values:\n"
            "          return None\n"
            "      return max(values)"
        ),
    },
    {
        "name": "iter_mutate",
        "pattern": "ITER_MUTATE",
        "title": "Iterating and mutating the same container produces undefined results",
        "trigger_keywords": ["modify", "in place", "do not mutate", "filter",
                                "remove", "delete", "without modifying"],
        "trigger_description": "The function either (a) is supposed to modify a container in place while iterating it, or (b) is supposed NOT to mutate the input but the natural implementation might.",
        "failure_to_avoid": "Iterating a list and removing elements from it during iteration (skips elements / raises). Returning a reference to the same list as input when the spec says return a new list.",
        "procedure": [
            "If you must NOT mutate the input, build the result in a NEW list/dict and return it. Never assign the input directly to the output variable.",
            "If you must mutate in place, iterate over a snapshot (`for x in list(original): ...`) and modify by index or with `.remove()`.",
            "After implementing, run a quick mental check: 'did I mutate the input when the spec said do not?'",
        ],
        "verification": "Pass a fresh list, capture it before calling, call, then assert the original list is unchanged after the call (if spec says no mutation).",
        "worked_example": (
            "BAD (returns input unchanged):\n"
            "  def remove_evens(lst):\n"
            "      return [x for x in lst if x % 2 == 1]  # OK actually\n"
            "  # but: lst.remove(x) inside a `for x in lst:` loop is BAD\n\n"
            "GOOD:\n"
            "  def remove_evens(lst):\n"
            "      return [x for x in lst if x % 2 == 1]  # input untouched"
        ),
    },
    {
        "name": "type_coerce",
        "pattern": "TYPE_COERCE",
        "title": "Mixed int/float/str inputs need explicit coercion or filtering",
        "trigger_keywords": ["str", "string", "int", "float", "convert",
                                "numeric", "may contain", "skip", "type"],
        "trigger_description": "The function takes a list/iterable whose elements may be a mix of types (e.g. ints, floats, strings) and the spec says to either coerce or skip non-numerics.",
        "failure_to_avoid": "Calling `sum(values)` when values may contain a non-numeric string raises TypeError. Calling `int(s)` raises on non-numeric s.",
        "procedure": [
            "Iterate elements one by one.",
            "For each element, attempt the conversion in a try/except (`try: float(x); except: continue`).",
            "Sum / aggregate only the successfully converted values.",
            "Confirm the function returns the correct value when ALL elements are non-numeric (typically 0).",
        ],
        "verification": "Test with a mix that includes valid numerics, numeric strings, and non-numeric strings. Test with an all-non-numeric list.",
        "worked_example": (
            "BAD: `return sum(values)`  # raises on str\n\n"
            "GOOD:\n"
            "  total = 0\n"
            "  for x in values:\n"
            "      try:\n"
            "          total += float(x)\n"
            "      except (ValueError, TypeError):\n"
            "          continue\n"
            "  return total"
        ),
    },
]


def render_card(card, include_procedure=True, include_example=True):
    """Render card for prompt injection."""
    lines = [
        f"## METHODOLOGICAL HINT: {card['name']}",
        f"Title: {card['title']}",
        f"Trigger: {card['trigger_description']}",
        f"Failure to avoid: {card['failure_to_avoid']}",
    ]
    if include_procedure:
        lines.append("Procedure:")
        for i, step in enumerate(card['procedure'], 1):
            lines.append(f"  {i}. {step}")
        lines.append(f"Verification: {card['verification']}")
    if include_example and card.get('worked_example'):
        lines.append(f"Worked example:\n{card['worked_example']}")
    return "\n".join(lines)


def render_card_lite(card):
    """Trigger + failure-label only, no procedure, no example."""
    return (
        f"## METHODOLOGICAL HINT: {card['name']}\n"
        f"Title: {card['title']}\n"
        f"Trigger: {card['trigger_description']}\n"
        f"Failure to avoid: {card['failure_to_avoid']}"
    )


GENERIC_WARNING = (
    "## METHODOLOGICAL HINT: be_careful\n"
    "Title: General coding caution\n"
    "Trigger: This problem may be tricky.\n"
    "Failure to avoid: Subtle bugs, edge cases, off-by-one errors."
)


if __name__ == "__main__":
    print(f"Total cards: {len(CARDS)}")
    for c in CARDS:
        print(f"  {c['name']:18s} → {c['pattern']}")
    print()
    print("=== Sample full render of mut_default card ===")
    print(render_card(CARDS[0]))
    print()
    print("=== Sample lite render ===")
    print(render_card_lite(CARDS[0]))
