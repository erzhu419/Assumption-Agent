"""Exp 78 problem set: Python coding tasks with specific bug patterns.

Each problem has:
  - pid: short id
  - prompt: the task description shown to the LLM
  - test_code: pytest-style test that returns 0 (pass) or 1+ (fail) when run
              against the LLM's solution
  - bug_patterns: list of bug-pattern tags this problem can trigger (used by
                  trigger detection — a wisdom card whose pattern matches one
                  of these tags fires for this problem)
  - solution_hint: a known-good solution (for our reference, not shown to LLM)

Bug patterns covered (8 total):
  MUT_DEFAULT  — mutable default argument
  SLICE_BOUNDARY — off-by-one in slicing/ranges
  FLOAT_EQ — float equality / precision
  LATE_BIND — late binding in closures
  COPY_SEMANTICS — shallow vs deep copy
  EMPTY_EDGE — empty iterable / single-element edge case
  ITER_MUTATE — modifying iterable during iteration
  TYPE_COERCE — implicit type coercion (str vs int, bytes vs str)
"""

PROBLEMS = [
    # ===== MUT_DEFAULT =====
    {
        "pid": "MD_01",
        "prompt": (
            "Write a function `add_to_log(message, log=None)` that appends the message "
            "to a log list and returns the log. If log is None, create a fresh empty list. "
            "Each call without a log argument must return a fresh list, NOT share state "
            "with previous calls.\n"
            "Output ONLY the function definition, no example calls, no print statements."
        ),
        "test_code": """
import importlib.util, sys
spec = importlib.util.spec_from_loader('soln', loader=None)
m = importlib.util.module_from_spec(spec)
exec(SOLUTION_CODE, m.__dict__)
add_to_log = m.add_to_log
# Test 1: explicit log
out1 = add_to_log("hello", log=[])
assert out1 == ["hello"], f"Expected ['hello'], got {out1}"
# Test 2: two separate calls without log must NOT share state
a = add_to_log("first")
b = add_to_log("second")
assert a == ["first"], f"First call got {a}, expected ['first']"
assert b == ["second"], f"Second call got {b}, expected ['second'] (NOT shared)"
print("PASS")
""",
        "bug_patterns": ["MUT_DEFAULT"],
    },
    {
        "pid": "MD_02",
        "prompt": (
            "Write a function `register(name, registry=None)` that adds the name to the registry "
            "dict (with value True) and returns it. If registry is None, create a fresh dict. "
            "Multiple calls without an explicit registry must produce independent dicts.\n"
            "Output ONLY the function definition."
        ),
        "test_code": """
import importlib.util, sys
spec = importlib.util.spec_from_loader('soln', loader=None)
m = importlib.util.module_from_spec(spec)
exec(SOLUTION_CODE, m.__dict__)
register = m.register
r1 = register("alice")
r2 = register("bob")
assert r1 == {"alice": True}, f"r1={r1}"
assert r2 == {"bob": True}, f"r2={r2} should not contain alice"
print("PASS")
""",
        "bug_patterns": ["MUT_DEFAULT"],
    },

    # ===== SLICE_BOUNDARY =====
    {
        "pid": "SB_01",
        "prompt": (
            "Write a function `last_n(lst, n)` that returns the last n elements of lst. "
            "Behavior:\n"
            "- if n == 0: return [] (empty list)\n"
            "- if n >= len(lst): return all of lst\n"
            "- else: return the last n elements.\n"
            "Output ONLY the function definition."
        ),
        "test_code": """
import importlib.util
spec = importlib.util.spec_from_loader('soln', loader=None)
m = importlib.util.module_from_spec(spec)
exec(SOLUTION_CODE, m.__dict__)
f = m.last_n
assert f([1,2,3], 0) == [], f"n=0 case: {f([1,2,3], 0)}"
assert f([1,2,3], 2) == [2,3]
assert f([1,2,3], 5) == [1,2,3]
assert f([], 3) == []
print("PASS")
""",
        "bug_patterns": ["SLICE_BOUNDARY", "EMPTY_EDGE"],
    },
    {
        "pid": "SB_02",
        "prompt": (
            "Write a function `every_kth(lst, k)` that returns elements at positions 0, k, 2k, ... "
            "(i.e. every k-th element including the first).\n"
            "Examples:\n"
            "  every_kth([1,2,3,4,5,6,7], 3) -> [1, 4, 7]\n"
            "  every_kth([1,2,3,4,5,6,7], 1) -> [1,2,3,4,5,6,7]\n"
            "  every_kth([], 3) -> []\n"
            "Edge: k=1 returns the entire list. k must be >= 1 (you may assume).\n"
            "Output ONLY the function definition."
        ),
        "test_code": """
import importlib.util
spec = importlib.util.spec_from_loader('soln', loader=None)
m = importlib.util.module_from_spec(spec)
exec(SOLUTION_CODE, m.__dict__)
f = m.every_kth
assert f([1,2,3,4,5,6,7], 3) == [1,4,7], f"k=3 case: {f([1,2,3,4,5,6,7], 3)}"
assert f([1,2,3,4,5,6,7], 1) == [1,2,3,4,5,6,7]
assert f([], 3) == []
assert f([1,2,3,4,5], 2) == [1,3,5]
print("PASS")
""",
        "bug_patterns": ["SLICE_BOUNDARY", "EMPTY_EDGE"],
    },

    # ===== FLOAT_EQ =====
    {
        "pid": "FE_01",
        "prompt": (
            "Write a function `is_close(a, b)` that returns True iff a and b are within "
            "1e-9 absolute or relative tolerance of each other. Both could be floats.\n"
            "It must return True for is_close(0.1 + 0.2, 0.3).\n"
            "Output ONLY the function definition."
        ),
        "test_code": """
import importlib.util
spec = importlib.util.spec_from_loader('soln', loader=None)
m = importlib.util.module_from_spec(spec)
exec(SOLUTION_CODE, m.__dict__)
f = m.is_close
assert f(0.1 + 0.2, 0.3) == True, "0.1+0.2 should be close to 0.3"
assert f(1.0, 1.0) == True
assert f(1.0, 1.1) == False
print("PASS")
""",
        "bug_patterns": ["FLOAT_EQ"],
    },
    {
        "pid": "FE_02",
        "prompt": (
            "Write a function `sum_to_target(values, target)` that returns True iff the sum of values "
            "equals target (within floating-point tolerance). Use 1e-9 tolerance.\n"
            "Output ONLY the function definition."
        ),
        "test_code": """
import importlib.util
spec = importlib.util.spec_from_loader('soln', loader=None)
m = importlib.util.module_from_spec(spec)
exec(SOLUTION_CODE, m.__dict__)
f = m.sum_to_target
assert f([0.1, 0.2], 0.3) == True
assert f([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 1.0) == True
assert f([1.0, 2.0], 4.0) == False
print("PASS")
""",
        "bug_patterns": ["FLOAT_EQ"],
    },

    # ===== LATE_BIND =====
    {
        "pid": "LB_01",
        "prompt": (
            "Write a function `make_adders(values)` that takes a list of numbers and returns "
            "a list of functions, where the i-th function adds values[i] to its argument.\n"
            "Example:\n"
            "  fns = make_adders([10, 20, 30])\n"
            "  fns[0](1) -> 11\n"
            "  fns[1](1) -> 21\n"
            "  fns[2](1) -> 31\n"
            "Output ONLY the function definition."
        ),
        "test_code": """
import importlib.util
spec = importlib.util.spec_from_loader('soln', loader=None)
m = importlib.util.module_from_spec(spec)
exec(SOLUTION_CODE, m.__dict__)
make = m.make_adders
fns = make([10, 20, 30])
assert fns[0](1) == 11, f"fns[0](1)={fns[0](1)}"
assert fns[1](1) == 21, f"fns[1](1)={fns[1](1)}"
assert fns[2](1) == 31, f"fns[2](1)={fns[2](1)}"
print("PASS")
""",
        "bug_patterns": ["LATE_BIND"],
    },

    # ===== COPY_SEMANTICS =====
    {
        "pid": "CS_01",
        "prompt": (
            "Write a function `safe_copy_grid(grid)` that takes a list of lists (2D grid) and returns "
            "a fully independent copy: modifying any cell of the copy must NOT affect the original.\n"
            "Output ONLY the function definition."
        ),
        "test_code": """
import importlib.util
spec = importlib.util.spec_from_loader('soln', loader=None)
m = importlib.util.module_from_spec(spec)
exec(SOLUTION_CODE, m.__dict__)
f = m.safe_copy_grid
g = [[1,2],[3,4]]
c = f(g)
c[0][0] = 99
assert g[0][0] == 1, f"original got modified, g={g}"
assert c[0][0] == 99
print("PASS")
""",
        "bug_patterns": ["COPY_SEMANTICS"],
    },
    {
        "pid": "CS_02",
        "prompt": (
            "Write a function `merge_configs(default, override)` that returns a dict with override "
            "applied on top of default — but neither default nor override should be mutated. "
            "default and override are dicts whose values may be lists.\n"
            "Output ONLY the function definition."
        ),
        "test_code": """
import importlib.util
spec = importlib.util.spec_from_loader('soln', loader=None)
m = importlib.util.module_from_spec(spec)
exec(SOLUTION_CODE, m.__dict__)
f = m.merge_configs
d = {"a": [1,2], "b": 1}
o = {"b": 2}
out = f(d, o)
assert out == {"a": [1,2], "b": 2}, f"out={out}"
out["a"].append(99)
assert d["a"] == [1,2], f"default mutated: d['a']={d['a']}"
print("PASS")
""",
        "bug_patterns": ["COPY_SEMANTICS"],
    },

    # ===== EMPTY_EDGE =====
    {
        "pid": "EE_01",
        "prompt": (
            "Write a function `safe_max(values)` that returns the maximum of values, or None if "
            "values is empty. Do not raise an exception on empty input.\n"
            "Output ONLY the function definition."
        ),
        "test_code": """
import importlib.util
spec = importlib.util.spec_from_loader('soln', loader=None)
m = importlib.util.module_from_spec(spec)
exec(SOLUTION_CODE, m.__dict__)
f = m.safe_max
assert f([3,1,2]) == 3
assert f([]) is None, f"empty case: {f([])}"
assert f([5]) == 5
print("PASS")
""",
        "bug_patterns": ["EMPTY_EDGE"],
    },

    # ===== ITER_MUTATE =====
    {
        "pid": "IM_01",
        "prompt": (
            "Write a function `remove_evens(lst)` that takes a list of integers and returns a NEW list "
            "containing only the odd values. Do not mutate the input. Be safe even if the function is "
            "called concurrently or repeatedly.\n"
            "Output ONLY the function definition."
        ),
        "test_code": """
import importlib.util
spec = importlib.util.spec_from_loader('soln', loader=None)
m = importlib.util.module_from_spec(spec)
exec(SOLUTION_CODE, m.__dict__)
f = m.remove_evens
inp = [1,2,3,4,5,6]
out = f(inp)
assert out == [1,3,5], f"out={out}"
assert inp == [1,2,3,4,5,6], f"input mutated: {inp}"
print("PASS")
""",
        "bug_patterns": ["ITER_MUTATE", "COPY_SEMANTICS"],
    },
    {
        "pid": "IM_02",
        "prompt": (
            "Write a function `dedup_inplace(lst)` that removes duplicates from `lst` IN PLACE while "
            "preserving the first occurrence of each value, and returns lst.\n"
            "Example: dedup_inplace([1,2,2,3,1,4]) -> [1,2,3,4] (and lst itself is now [1,2,3,4]).\n"
            "Output ONLY the function definition."
        ),
        "test_code": """
import importlib.util
spec = importlib.util.spec_from_loader('soln', loader=None)
m = importlib.util.module_from_spec(spec)
exec(SOLUTION_CODE, m.__dict__)
f = m.dedup_inplace
inp = [1,2,2,3,1,4]
out = f(inp)
assert out == [1,2,3,4], f"out={out}"
assert inp == [1,2,3,4], f"input not modified: {inp}"
print("PASS")
""",
        "bug_patterns": ["ITER_MUTATE"],
    },

    # ===== TYPE_COERCE =====
    {
        "pid": "TC_01",
        "prompt": (
            "Write a function `numeric_sum(values)` that returns the numeric sum of values. "
            "values may contain a mix of int, float, and string-encoded numbers (e.g. '3', '4.5'). "
            "Strings that are not numeric should be skipped (not raise).\n"
            "Output ONLY the function definition."
        ),
        "test_code": """
import importlib.util
spec = importlib.util.spec_from_loader('soln', loader=None)
m = importlib.util.module_from_spec(spec)
exec(SOLUTION_CODE, m.__dict__)
f = m.numeric_sum
out = f([1, 2.5, "3", "abc", "4.5"])
assert abs(out - 11.0) < 1e-9, f"out={out}, expected 11.0"
assert f([]) == 0 or f([]) == 0.0
print("PASS")
""",
        "bug_patterns": ["TYPE_COERCE", "EMPTY_EDGE"],
    },

    # ===== Multi-pattern integration tasks =====
    {
        "pid": "MIX_01",
        "prompt": (
            "Write a function `tally(items, counts=None)` that takes a list of strings `items` and an "
            "optional `counts` dict mapping item -> int. Increment each item's count by 1 in counts and "
            "return counts. If counts is None, start from a fresh empty dict; calls without counts must "
            "produce independent results.\n"
            "Output ONLY the function definition."
        ),
        "test_code": """
import importlib.util
spec = importlib.util.spec_from_loader('soln', loader=None)
m = importlib.util.module_from_spec(spec)
exec(SOLUTION_CODE, m.__dict__)
f = m.tally
a = f(["x","y","x"])
b = f(["z"])
assert a == {"x":2, "y":1}, f"a={a}"
assert b == {"z":1}, f"b={b} (must not contain x or y)"
print("PASS")
""",
        "bug_patterns": ["MUT_DEFAULT"],
    },
    {
        "pid": "MIX_02",
        "prompt": (
            "Write a function `pairwise_close(lst, tol=1e-9)` that returns True iff every consecutive "
            "pair in lst is within tol. Returns True for empty or single-element lists.\n"
            "Output ONLY the function definition."
        ),
        "test_code": """
import importlib.util
spec = importlib.util.spec_from_loader('soln', loader=None)
m = importlib.util.module_from_spec(spec)
exec(SOLUTION_CODE, m.__dict__)
f = m.pairwise_close
assert f([0.1+0.2, 0.3]) == True
assert f([1.0, 1.0, 1.0]) == True
assert f([]) == True
assert f([1.0]) == True
assert f([1.0, 2.0]) == False
print("PASS")
""",
        "bug_patterns": ["FLOAT_EQ", "EMPTY_EDGE"],
    },
    {
        "pid": "MIX_03",
        "prompt": (
            "Write a function `apply_offsets(values, offsets)` that returns a list of functions. "
            "The i-th returned function takes one argument and returns argument + offsets[i]. "
            "If offsets is shorter than values, pad with offset 0; if longer, ignore extras.\n"
            "Output ONLY the function definition."
        ),
        "test_code": """
import importlib.util
spec = importlib.util.spec_from_loader('soln', loader=None)
m = importlib.util.module_from_spec(spec)
exec(SOLUTION_CODE, m.__dict__)
f = m.apply_offsets
fns = f([10,20,30], [1,2,3])
assert fns[0](0) == 1, f"fns[0](0)={fns[0](0)}"
assert fns[1](0) == 2
assert fns[2](0) == 3
fns2 = f([10,20], [5])
assert fns2[1](0) == 0  # padded with 0
print("PASS")
""",
        "bug_patterns": ["LATE_BIND", "SLICE_BOUNDARY"],
    },
    {
        "pid": "MIX_04",
        "prompt": (
            "Write a function `parse_log_lines(lines)` that takes a list of strings and returns a dict "
            "mapping the log level (the first word, uppercased) to a list of full lines with that level. "
            "Skip lines that have no level (empty after strip). Lines like 'INFO: hello' have level 'INFO'. "
            "Lines like '[ERROR] fail' have level 'ERROR'. Empty list input must return {}.\n"
            "Output ONLY the function definition."
        ),
        "test_code": """
import importlib.util
spec = importlib.util.spec_from_loader('soln', loader=None)
m = importlib.util.module_from_spec(spec)
exec(SOLUTION_CODE, m.__dict__)
f = m.parse_log_lines
out = f(["INFO: starting", "ERROR: bad", "info: x", "[ERROR] fail", "  "])
assert "INFO" in out
assert any("starting" in s for s in out["INFO"])
assert any("bad" in s for s in out["ERROR"])
assert f([]) == {}
print("PASS")
""",
        "bug_patterns": ["EMPTY_EDGE", "TYPE_COERCE"],
    },
    {
        "pid": "MIX_05",
        "prompt": (
            "Write a function `flatten_lists(nested)` that takes a list whose elements may be lists or "
            "non-list values, and returns a new flat list. Only flatten one level (so [[1,2], 3] -> [1,2,3]). "
            "Empty input -> []. Do not mutate the input.\n"
            "Output ONLY the function definition."
        ),
        "test_code": """
import importlib.util
spec = importlib.util.spec_from_loader('soln', loader=None)
m = importlib.util.module_from_spec(spec)
exec(SOLUTION_CODE, m.__dict__)
f = m.flatten_lists
inp = [[1,2], 3, [4,5]]
out = f(inp)
assert out == [1,2,3,4,5], f"out={out}"
assert inp == [[1,2], 3, [4,5]], f"input mutated: {inp}"
assert f([]) == []
print("PASS")
""",
        "bug_patterns": ["EMPTY_EDGE", "COPY_SEMANTICS"],
    },
    {
        "pid": "MIX_06",
        "prompt": (
            "Write a function `running_average(values)` that returns a list of running averages: "
            "out[i] = mean(values[:i+1]). For empty input, return []. For single element [x], return [x].\n"
            "Output ONLY the function definition."
        ),
        "test_code": """
import importlib.util
spec = importlib.util.spec_from_loader('soln', loader=None)
m = importlib.util.module_from_spec(spec)
exec(SOLUTION_CODE, m.__dict__)
f = m.running_average
out = f([2, 4, 6])
assert abs(out[0] - 2.0) < 1e-9
assert abs(out[1] - 3.0) < 1e-9
assert abs(out[2] - 4.0) < 1e-9
assert f([]) == []
assert f([5]) == [5] or abs(f([5])[0] - 5.0) < 1e-9
print("PASS")
""",
        "bug_patterns": ["EMPTY_EDGE", "FLOAT_EQ"],
    },
    {
        "pid": "MIX_07",
        "prompt": (
            "Write a function `add_score(player, score, scores=None)` that adds the player to a "
            "dict scores with score as value, returns scores. If scores is None, start from a fresh dict "
            "(do NOT share state across calls). Multiple separate calls without scores must produce "
            "independent dicts.\n"
            "Output ONLY the function definition."
        ),
        "test_code": """
import importlib.util
spec = importlib.util.spec_from_loader('soln', loader=None)
m = importlib.util.module_from_spec(spec)
exec(SOLUTION_CODE, m.__dict__)
f = m.add_score
a = f("alice", 10)
b = f("bob", 20)
assert a == {"alice": 10}
assert b == {"bob": 20}, f"shared state: b={b}"
print("PASS")
""",
        "bug_patterns": ["MUT_DEFAULT"],
    },
    {
        "pid": "MIX_08",
        "prompt": (
            "Write a function `head_tail(lst)` that returns a tuple (head, tail) where:\n"
            "  - if lst is empty: head=None, tail=[]\n"
            "  - if lst has 1 element: head=that element, tail=[]\n"
            "  - else: head=lst[0], tail=remaining elements\n"
            "Output ONLY the function definition."
        ),
        "test_code": """
import importlib.util
spec = importlib.util.spec_from_loader('soln', loader=None)
m = importlib.util.module_from_spec(spec)
exec(SOLUTION_CODE, m.__dict__)
f = m.head_tail
assert f([1,2,3]) == (1, [2,3])
assert f([]) == (None, [])
assert f([5]) == (5, [])
print("PASS")
""",
        "bug_patterns": ["EMPTY_EDGE", "SLICE_BOUNDARY"],
    },
    {
        "pid": "MIX_09",
        "prompt": (
            "Write a function `safe_divide_each(numerators, denominators, default=0.0)` that returns a list "
            "where the i-th element is numerators[i] / denominators[i], or `default` if the denominator is 0. "
            "Both lists have the same length.\n"
            "Output ONLY the function definition."
        ),
        "test_code": """
import importlib.util
spec = importlib.util.spec_from_loader('soln', loader=None)
m = importlib.util.module_from_spec(spec)
exec(SOLUTION_CODE, m.__dict__)
f = m.safe_divide_each
out = f([10, 20, 30], [2, 0, 5])
assert abs(out[0] - 5.0) < 1e-9
assert out[1] == 0.0
assert abs(out[2] - 6.0) < 1e-9
print("PASS")
""",
        "bug_patterns": ["FLOAT_EQ", "EMPTY_EDGE"],
    },
    {
        "pid": "MIX_10",
        "prompt": (
            "Write a function `unique_in_order(lst)` that returns a NEW list containing each unique "
            "element of lst in the order of its first occurrence. Original lst not modified. "
            "Empty input -> [].\n"
            "Output ONLY the function definition."
        ),
        "test_code": """
import importlib.util
spec = importlib.util.spec_from_loader('soln', loader=None)
m = importlib.util.module_from_spec(spec)
exec(SOLUTION_CODE, m.__dict__)
f = m.unique_in_order
inp = [1, 2, 2, 3, 1, 4]
out = f(inp)
assert out == [1,2,3,4], f"out={out}"
assert inp == [1,2,2,3,1,4], f"input mutated"
print("PASS")
""",
        "bug_patterns": ["ITER_MUTATE", "COPY_SEMANTICS"],
    },
]

assert len({p["pid"] for p in PROBLEMS}) == len(PROBLEMS), "duplicate pid"
print(f"PROBLEMS module: {len(PROBLEMS)} tasks loaded")

# Pattern -> list of pids that mention this pattern (used for trigger detection)
def pids_for_pattern(pat):
    return [p["pid"] for p in PROBLEMS if pat in p["bug_patterns"]]

if __name__ == "__main__":
    from collections import Counter
    print(f"Total tasks: {len(PROBLEMS)}")
    bug_counter = Counter(b for p in PROBLEMS for b in p["bug_patterns"])
    print("Bug pattern coverage:")
    for pat, n in sorted(bug_counter.items(), key=lambda x: -x[1]):
        print(f"  {pat}: {n} tasks")
