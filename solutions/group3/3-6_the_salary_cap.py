"""
================================================================================
CHALLENGE: The Salary Cap (200 Points) [Slug: salary_cap]
CATEGORY ID: 6bf4f9f1-66ff-4ed9-a634-80dc9a2dd026
CHALLENGE ID: 5f8338ef-f142-4371-acff-0382ca9b5229
SUMMARY: Run the cost model backwards: given a runtime budget, how much QPE fits?
================================================================================

DESCRIPTION:
Resource estimation is sometimes presented in the following (simplified) flow: pick a quantum algorithm (represented by a circuit of a certain size), compile it, count the T gates, multiply by a per-T-gate time (based on some assumptions), and read off a runtime:

$$
T_{\mathrm{wall}} = N_T \cdot t_T .
$$

It is often helpful to think of resource estimation in the following alternative direction: given certain characteristics of a quantum computer (e.g, a fixed logical T-gate time, fixed by the code distance and the cycle time, a given number of logical qubits available, and a timeline for the project or for allocated compute time), how large of a problem, represented by a given algorithm, fits within those constraints?

Quantum phase estimation is a good place to ask at a high level, because the trade-off is fairly clean. With $m$ estimation wires the phase resolution is

$$
\epsilon = 2^{-m} ,
$$

so every extra wire gets you exactly one more bit of the phase. It also costs you roughly a doubling of the work, because QPE applies the controlled base unitary $\sum_{k=0}^{m-1} 2^k = 2^m - 1$ times. One more bit for twice the runtime is part of the bargain of phase estimation, and it is a far better bargain than sampling, where one more decimal digit costs a factor of one hundred extra samples/shots.

With (textbook) quantum phase estimation, two characteristics decide the answer to the earlier question of how large of a problem we might be able to run given certain hardware constraints.

The first is the base unitary. The T-count of QPE depends directly on what you are estimating the phase of. If you use a single-qubit rotation, the powers $U^{2^k}$ collapse, for example $R_Z(\theta)^{2^k} = R_Z(2^k \theta)$, so the controlled-powers block does not grow at all, and the cost you measure is in larger part the cost of the inverse QFT. A generic multi-qubit unitary on the other hand has no such structure, its powers genuinely accumulate, and the controlled-powers block dominates.

The second is the rotation synthesis precision. There is no exact Clifford+T circuit for an arbitrary rotation, so every rotation must be approximated, at a cost that grows roughly like $\log(1/\epsilon_{\mathrm{rot}})$ T gates apiece.

A key characteristic of resource estimation is that it is a symbolic calculation. It never builds a state vector, so it will easily cost a routine on hundreds of qubits that could not be simulated on any classical computer.

## Challenge code

In the code below you must complete this function:

- `affordable_precision`: returns the largest QPE size that fits a wall-clock budget. Its arguments are:

    - `t_gate_seconds` (`float`): the time to execute one logical T gate, in seconds.
    - `budget_seconds` (`float`): the total wall-clock budget, in seconds.
    - `max_estimation_wires` (`int`): a safety bound on the search. A correct model never reaches it.

Take the base unitary to be a generic two-qubit unitary, `qre.QubitUnitary(num_wires=2)`, and build phase estimation on it with `qre.QPE`. Cost it in the gate set `GATE_SET` provided in the code, with the single-qubit rotation precision set to `ROTATION_PRECISION` through a `qre.ResourceConfig`.

The runtime of a size is its T-count times `t_gate_seconds`, and a size is affordable when that runtime is no greater than `budget_seconds`. You want the largest affordable `m`. Because the T-count grows monotonically in `m`, you may stop at the first size that overruns.

There are two rules on how the answer is reached, both enforced by tests:

- The T-count you return must match what `qre.estimate` reports at the size you return. Reading it out of the result object is the point of the exercise, rather than reproducing it from a formula worked out beforehand.
- Your search must cost more than one size. Finding the largest size that fits means comparing sizes, so `qre.estimate` has to be called at least twice. How you search is up to you. For example, a scan upwards from `m = 1` and a binary search both pass.

### Input

A Python `list` of two floats, `[t_gate_seconds, budget_seconds]`:

- `t_gate_seconds` (`float`): the time to execute one logical T gate, in seconds.
- `budget_seconds` (`float`): the total wall-clock budget, in seconds.

### Output

A Python `tuple` of three values, `(num_estimation_wires, precision, t_count)`:

- `num_estimation_wires` (`int`): the largest affordable `m`.
- `precision` (`float`): the phase resolution `2 ** -m` that size buys.
- `t_count` (`int`): the T-count at that size, as reported by `qre.estimate`.


The wire count and the T-count are compared exactly, so cast them with `int(...)`; the resolution is compared to a relative tolerance of `1e-12`, which any of `2.0 ** -m`, `0.5 ** m` or `1 / 2 ** m` will satisfy. Return plain Python numbers rather than NumPy scalars, a NumPy integer or boolean cannot be serialised by the grader.

### Test cases
Two test cases are shown to you; the remaining cases are hidden and run when you submit.

As an example, for input `[1e-06, 1.0]`, your solution should output `(9, 0.001953125, 684108)`.

This means that a logical T gate takes one microsecond and you have one second of machine time, so the budget is a million T gates. Walking up the sizes, `m = 9` costs 684,108 T gates, which is 0.684 seconds and fits, while `m = 10` costs 1,366,956 T gates, which is 1.367 seconds and does not. So nine wires is the largest affordable size, and it gets you a phase resolution of $2^{-9} = 0.001953125$.

Good luck!

================================================================================

PUBLIC TEST CASES:
--- Test Case #1 ---
Input:
[1e-06, 1.0]
Expected Output:
[9, 0.001953125, 684108]

--- Test Case #2 ---
Input:
[0.001, 60.0]
Expected Output:
[5, 0.03125, 42252]

================================================================================

"""

# >>> BLOCK [SOLUTION_CONTEXT] ID=None
import json
import pennylane as qp
import pennylane.numpy as np

# <<< END BLOCK [SOLUTION_CONTEXT]

# >>> BLOCK [SOLUTION_CONTEXT] ID=685f1611-861e-4043-a565-ff40e7f6463e
qre = qp.estimator

GATE_SET = {"Hadamard", "S", "CNOT", "T"}
ROTATION_PRECISION = 1e-6


def affordable_precision(t_gate_seconds, budget_seconds, max_estimation_wires=40):
    """Finds the largest quantum phase estimation that fits a wall-clock budget.

    Args:
        t_gate_seconds (float): The time to execute one logical T gate, in seconds.
        budget_seconds (float): The total wall-clock budget, in seconds.
        max_estimation_wires (int): A safety bound on the search. A correct model never
            reaches it; a model that returns exactly this value has not found a limit,
            it has run out of loop.

    Returns:
        tuple[int, float, int]: The largest affordable number of estimation wires m,
            the phase resolution 2**-m that buys, and the T-count at that size.
    """

# <<< END BLOCK [SOLUTION_CONTEXT]

# >>> BLOCK [SOLUTION_TEMPLATE] ID=81b911c8-92f0-4a9c-a5b9-27df0e8a55dd
    config = qre.ResourceConfig()
    config.set_single_qubit_rot_precision(ROTATION_PRECISION)
    base = qre.QubitUnitary(num_wires=2)
    best = None
    for m in range(1, max_estimation_wires):
        qpe_op = qre.QPE(base, num_estimation_wires=m)
        resources = qre.estimate(qpe_op, gate_set=GATE_SET, config=config)
        t_count = int(resources.gate_counts.get("T", 0))
        if t_count * t_gate_seconds <= budget_seconds:
            best = (m, 2.0 ** -m, t_count)
        else:
            break
    return best

# <<< END BLOCK [SOLUTION_TEMPLATE]

# >>> BLOCK [TEST_SUPPORT] ID=eb96eece-fb55-4249-9753-c3715c97ce62
# These functions are responsible for testing the solution.
#
# Everything the grader needs is copied into local variables at the top of run(),
# before any of your code runs, and the verdict is decided there. Your solution
# body executes inside affordable_precision, in this same module namespace, so it
# could rebind any name defined at module level -- including check(). None of the
# names the grade depends on are read from that namespace after your code runs.
def run(test_case_input: str) -> str:
    loads, dumps, close = json.loads, json.dumps, np.isclose
    estimator = qre
    real_estimate = estimator.estimate
    make_qpe, make_base = estimator.QPE, estimator.QubitUnitary
    make_config = estimator.ResourceConfig
    gate_set, rot_precision = set(GATE_SET), ROTATION_PRECISION
    t_gate, budget = loads(test_case_input)
    calls = []

    def audited(*args, **kwargs):
        """Records the T-count of every size your search actually costs."""
        resources = real_estimate(*args, **kwargs)
        calls.append(int(resources.gate_counts.get("T", 0)))
        return resources

    def reference_t_count(wires):
        """Costs one QPE size exactly as the challenge specifies it."""
        config = make_config()
        config.set_single_qubit_rot_precision(rot_precision)
        resources = real_estimate(
            make_qpe(make_base(num_wires=2), num_estimation_wires=wires),
            gate_set=gate_set,
            config=config,
        )
        return int(resources.gate_counts.get("T", 0))

    estimator.estimate = audited
    try:
        best = affordable_precision(t_gate, budget)
    finally:
        estimator.estimate = real_estimate

    if best is None:
        raise AssertionError(
            "Your function returned None, so no size was ever recorded as "
            "affordable. Every test case here has an affordable size of at least "
            "m = 1, so check that your loop starts at m = 1 and that you record "
            "the current size as the best one before moving on to the next.")
    try:
        wires, precision, t_count = best
    except (TypeError, ValueError):
        raise AssertionError(
            "Return exactly three values: the number of estimation wires, the "
            "phase resolution, and the T-count.")
    wires = int(wires)
    if wires < 1:
        raise AssertionError(
            "You reported " + str(wires) + " estimation wires. The search starts "
            "at m = 1.")
    if wires > 40:
        raise AssertionError(
            "You reported " + str(wires) + " estimation wires, past the "
            "max_estimation_wires safety bound. A correct model never reaches it.")
    if not close(float(precision), 2.0 ** -wires, rtol=1e-12, atol=0.0):
        raise AssertionError(
            "You reported " + str(wires) + " wires with a resolution of "
            + str(float(precision)) + ", but the resolution that buys is 2**-m = "
            + str(2.0 ** -wires) + ".")
    if len(calls) < 2:
        raise AssertionError(
            "The estimator was called " + str(len(calls)) + " time(s). Finding the "
            "largest affordable size means comparing sizes, so call qre.estimate "
            "inside your search rather than obtaining the T-count another way.")
    here = reference_t_count(wires)
    if int(t_count) != here:
        raise AssertionError(
            "At m = " + str(wires) + " the T-count is " + str(here) + ", but you "
            "reported " + str(int(t_count)) + ". If you are within a fraction of a "
            "percent, the rotation precision is not set to ROTATION_PRECISION. If "
            "you are off by orders of magnitude, the base unitary is wrong -- it "
            "must be qre.QubitUnitary(num_wires=2).")
    if here * t_gate > budget:
        raise AssertionError(
            "m = " + str(wires) + " costs " + str(here) + " T gates, which run for "
            + str(here * t_gate) + " s against a budget of " + str(budget) + " s. "
            "That size does not fit -- keep the largest one that does.")
    if reference_t_count(wires + 1) * t_gate <= budget:
        raise AssertionError(
            "m = " + str(wires) + " fits, but so does m = " + str(wires + 1) + ". "
            "Return the largest affordable size, not the first one you find.")
    return dumps([wires, float(precision), int(t_count)])


def check(solution_output: str, expected_output: str) -> None:
    # run() has already decided the grade; this is a redundant second comparison
    # against the stored answer.
    got, exp = json.loads(solution_output), json.loads(expected_output)
    assert got[0] == exp[0], "Wrong number of estimation wires."
    assert np.isclose(got[1], exp[1], rtol=1e-12, atol=0.0), \
        "Wrong phase resolution -- it should be 2**-m."
    assert got[2] == exp[2], "T-count mismatch."

# <<< END BLOCK [TEST_SUPPORT]

# >>> BLOCK [TEST_CASES] ID=None
# These are the public test cases
test_cases = [
    ('[1e-06, 1.0]', '[9, 0.001953125, 684108]'),
    ('[0.001, 60.0]', '[5, 0.03125, 42252]')
]

# <<< END BLOCK [TEST_CASES]

# >>> BLOCK [TEST_SUPPORT] ID=None
# This will run the public test cases locally
for i, (input_, expected_output) in enumerate(test_cases):
    print(f"Running test case {i} with input '{input_}'...")

    try:
        output = run(input_)

    except Exception as exc:
        print(f"Runtime Error. {exc}")

    else:
        if message := check(output, expected_output):
            print(f"Wrong Answer. Have: '{output}'. Want: '{expected_output}'.")

        else:
            print("Correct!")

# <<< END BLOCK [TEST_SUPPORT]
