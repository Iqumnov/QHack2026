"""
================================================================================
CHALLENGE: Same Kit, Different Number (100 Points) [Slug: same_kit_diff_number]
CATEGORY ID: 6bf4f9f1-66ff-4ed9-a634-80dc9a2dd026
CHALLENGE ID: c52b0758-3cf5-4f7f-acdf-23976e3825d9
SUMMARY: A global phase is invisible - until you control the gate that carries it.
================================================================================

DESCRIPTION:
Two different matrices can be the same gate.

If $V = e^{i\alpha} U$ for some real $\alpha$, then $U$ and $V$ act identically on every state, and no measurement performed on that qubit can tell them apart. The reason is that measurement probabilities are built from squared magnitudes: for any state $|\psi\rangle$ and any observable, the phase $e^{i\alpha}$ appears once in the amplitude and once, conjugated, in its complex conjugate, so it cancels:

$$
\left| \langle \phi | V | \psi \rangle \right|^2 = \left| e^{i\alpha} \right|^2 \left| \langle \phi | U | \psi \rangle \right|^2 = \left| \langle \phi | U | \psi \rangle \right|^2 .
$$

This is why $-H$ and $H$ are the same gate, and why `qp.matrix` returning a matrix that differs from yours by a sign may not be an error.

## Turning a global phase into a relative phase

A global phase multiplies the whole state, so it drops out in the measurement process. A relative phase between two branches of a superposition does not. This is effectively what gives rise to interference effects in the measurement process at the end of a quantum computation. Controlling a gate converts the former into the latter.

Suppose $M = e^{i\alpha} I$ and put the control wire in $|+\rangle$ with the target in $|0\rangle$. The controlled operation acts on the $|1\rangle$ branch only:

$$
\mathrm{ctrl}(M) \, \frac{|0\rangle + |1\rangle}{\sqrt{2}} \otimes |0\rangle = \frac{|0\rangle + e^{i\alpha} |1\rangle}{\sqrt{2}} \otimes |0\rangle .
$$

The target remains untouched, but the control now carries a relative phase $\alpha$. A second Hadamard turns that phase into an observable measure of population,

$$
H \, \frac{|0\rangle + e^{i\alpha}|1\rangle}{\sqrt{2}} = \cos\left(\frac{\alpha}{2}\right) e^{i\alpha/2} |0\rangle - i \sin\left(\frac{\alpha}{2}\right) e^{i\alpha/2} |1\rangle ,
$$

so the probability of reading $1$ on the control is

$$
P(1) = \sin^2\left( \frac{\alpha}{2} \right) .
$$

This is the mechanism behind phase kickback, and behind quantum algorithms that read an eigenvalue out of a controlled unitary.

## Challenge statement

You are handed two $2 \times 2$ unitaries, $U$ and $V$. You will need to construct the quantity

$$
M = V U^{\dagger} ,
$$

and answer the following three questions.

1. Is $V$ the same gate as $U$ up to a global phase? 
2. If so, what is that phase $\alpha$, folded into the range $[0, 2\pi)$? If not, report `0.0`.
3. Implement the phase kickback circuit described above with the control and target wires initialised as |0>. What is the probability of measuring 1 on the control wire?

$$
\text{control: } |0\rangle \longrightarrow H \longrightarrow \bullet \longrightarrow H \longrightarrow \text{measure}
$$

$$
\text{target: } |0\rangle \longrightarrow M
$$

## Challenge code

You must complete this function:

- `same_gate_and_phase`: returns the verdict, the phase, and the control probability. Its arguments are:

    - `u_matrix` (`np.ndarray`): a $2 \times 2$ complex unitary.
    - `v_matrix` (`np.ndarray`): a $2 \times 2$ complex unitary.

### Input

Each test case is a Python `dict` with two keys, `"u"` and `"v"`. Each matrix is written as a list of 2 rows, each row a list of 2 entries, and each entry a `[real, imaginary]` pair of floats. The second public test case, $U = X$ and $V = Z$, looks like this:

```
{
    "u": [[[0.0, 0.0], [1.0, 0.0]],
          [[1.0, 0.0], [0.0, 0.0]]],
    "v": [[[1.0, 0.0], [0.0, 0.0]],
          [[0.0, 0.0], [-1.0, 0.0]]],
}
```

You never handle that pair format yourself. The grading code converts both matrices into single complex `np.ndarray`s before calling your function, so `u_matrix` and `v_matrix` arrive as ordinary $2 \times 2$ arrays of complex numbers.

The two public test cases are $U = H$ with $V = -H$ (the same gate, with $\alpha = \pi$), and $U = X$ with $V = Z$ (not the same gate). The remaining cases are hidden.

### Output

Return a `tuple` of three plain Python values:

```
(same_gate, alpha, p_one)
```

- `same_gate` (`bool`): whether $V$ is $U$ up to a global phase.
- `alpha` (`float`): the phase in $[0, 2\pi)$, or `0.0` when `same_gate` is `False`.
- `p_one` (`float`): the probability of measuring $1$ on the control wire.

The two floats are compared to within a tolerance of `1e-5`.

Good luck!

================================================================================

PUBLIC TEST CASES:
--- Test Case #1 ---
Input:
{"u": [[[0.7071067811865475, 0.0], [0.7071067811865475, 0.0]], [[0.7071067811865475, 0.0], [-0.7071067811865475, 0.0]]], "v": [[[-0.7071067811865475, 0.0], [-0.7071067811865475, 0.0]], [[-0.7071067811865475, 0.0], [0.7071067811865475, 0.0]]]}
Expected Output:
[true, 3.14159265359, 1.0]

--- Test Case #2 ---
Input:
{"u": [[[0.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [0.0, 0.0]]], "v": [[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [-1.0, 0.0]]]}
Expected Output:
[false, 0.0, 0.5]

================================================================================

"""

# >>> BLOCK [SOLUTION_CONTEXT] ID=None
import json
import pennylane as qp
import pennylane.numpy as np

# <<< END BLOCK [SOLUTION_CONTEXT]

# >>> BLOCK [SOLUTION_CONTEXT] ID=e5361f4b-db58-4b9f-8d09-62f7f3a62a77
def same_gate_and_phase(u_matrix, v_matrix):
    """Decides whether two unitaries are the same gate, and exposes the phase.

    Args:
        u_matrix (np.ndarray): A 2x2 unitary matrix with complex entries.
        v_matrix (np.ndarray): A 2x2 unitary matrix with complex entries.

    Returns:
        tuple[bool, float, float]: Whether v_matrix equals u_matrix up to a
            global phase; the phase alpha in [0, 2*pi) if so and 0.0 if not;
            and the probability of measuring 1 on the control wire of the
            phase kickback circuit described in the problem statement.
    """
    dev = qp.device("default.qubit", wires=2)

# <<< END BLOCK [SOLUTION_CONTEXT]

# >>> BLOCK [SOLUTION_TEMPLATE] ID=6ba09418-2fd5-4025-9976-ec26c0f3c355
    M = v_matrix @ u_matrix.conj().T
    cand_alpha = float(np.angle(M[0, 0])) % (2 * np.pi)
    if np.allclose(M, np.exp(1j * cand_alpha) * np.eye(2), atol=1e-5):
        same_gate = True
        alpha = cand_alpha
    else:
        same_gate = False
        alpha = 0.0

    @qp.qnode(dev)
    def circuit():
        qp.Hadamard(wires=0)
        qp.ctrl(qp.QubitUnitary(M, wires=1), control=0)
        qp.Hadamard(wires=0)
        return qp.probs(wires=0)

    probs = circuit()
    p_one = float(probs[1])
    return same_gate, alpha, p_one

# <<< END BLOCK [SOLUTION_TEMPLATE]

# >>> BLOCK [TEST_SUPPORT] ID=330d611c-323d-46d5-a666-7e4bfa126773
# These functions are responsible for testing the solution.
def to_matrix(rows):
    """Turns a 2x2 nested list of [real, imaginary] pairs into a complex matrix."""
    return np.array([[complex(entry[0], entry[1]) for entry in row] for row in rows])


AUDIT = {"tapes": []}
_real_device = qp.device
STATE_READOUTS = ("StateMP", "DensityMatrixMP", "PurityMP", "VnEntropyMP", "MutualInfoMP")


def audited_device(*args, **kwargs):
    """Wraps a device so the grader can see which circuits were actually run."""
    dev = _real_device(*args, **kwargs)
    inner = dev.execute

    def execute(circuits, *args, **kwargs):
        batch = [circuits] if isinstance(circuits, qp.tape.QuantumScript) else circuits
        for tape in batch:
            AUDIT["tapes"].append(tape)
        return inner(circuits, *args, **kwargs)

    dev.execute = execute
    return dev


def violation(p_one):
    """Returns a message if the reported probability did not come from the circuit."""
    tapes = AUDIT["tapes"]
    if len(tapes) != 1:
        return ("Expected exactly one circuit execution, saw " + str(len(tapes))
                + ". Part 3 asks for the probability read off the phase kickback "
                "circuit, so that circuit has to be built and run.")
    tape = tapes[0]
    measurements = tape.measurements
    for m in measurements:
        if type(m).__name__ in STATE_READOUTS:
            return ("Disallowed measurement: " + type(m).__name__ + ". Read the "
                    "control wire with qp.probs, not the full state.")
    if len(measurements) != 1 or type(measurements[0]) is not qp.measurements.ProbabilityMP:
        return ("Return exactly one qp.probs measurement on the control wire. "
                "Saw: " + str([type(m).__name__ for m in measurements]) + ".")
    if len(measurements[0].wires) != 1:
        return ("Measure qp.probs on the single control wire, not on "
                + str(len(measurements[0].wires)) + " wires.")
    control = measurements[0].wires[0]
    controlled = [op for op in tape.operations if len(op.control_wires) > 0]
    if len(controlled) == 0:
        return ("Your circuit contains no controlled operation. A global phase is "
                "only visible once the gate carrying it is controlled.")
    for op in controlled:
        if control not in list(op.control_wires):
            return ("The controlled operation " + type(op).__name__ + " is not "
                    "controlled on the wire you measured (wire " + str(control)
                    + "). The phase is kicked back onto the control wire.")
        targets = [w for w in op.wires if w not in list(op.control_wires)]
        if len(targets) != 1:
            return ("The controlled operation " + type(op).__name__ + " acts on "
                    + str(len(targets)) + " target wires. M is a single-qubit "
                    "unitary, so it has exactly one target.")
    replay = qp.tape.QuantumScript(tape.operations, tape.measurements)
    device = _real_device("default.qubit", wires=list(tape.wires))
    probs = np.array(qp.execute([replay], device)[0]).flatten()
    if not np.isclose(float(probs[1]), float(p_one), atol=1e-5):
        return ("The probability you returned is not the one your circuit produced. "
                "Your circuit gives " + str(round(float(probs[1]), 6)) + " for index 1.")
    return None


def inconsistent(same_gate, alpha, p_one):
    """Applies the free self-check from the problem statement: P(1) = sin^2(alpha/2)."""
    if not bool(same_gate):
        return None
    predicted = float(np.sin(float(alpha) / 2.0) ** 2)
    if np.isclose(predicted, float(p_one), atol=1e-5):
        return None
    return ("You reported the same gate with alpha = " + str(round(float(alpha), 6))
            + " and P(1) = " + str(round(float(p_one), 6)) + ", but when the two "
            "matrices are the same gate P(1) must equal sin^2(alpha/2) = "
            + str(round(predicted, 6)) + ". One of the two is wrong: check the fold "
            "of alpha into [0, 2*pi) and the order of the Hadamards in the circuit.")


def run(test_case_input: str) -> str:
    case = json.loads(test_case_input)
    u_matrix = to_matrix(case["u"])
    v_matrix = to_matrix(case["v"])
    AUDIT["tapes"] = []
    qp.device = audited_device
    try:
        same_gate, alpha, p_one = same_gate_and_phase(u_matrix, v_matrix)
    finally:
        qp.device = _real_device
    # Raised here rather than deferred to check(), so that a solution which rebinds
    # check() in the module namespace cannot swallow the verdict.
    message = violation(p_one) or inconsistent(same_gate, alpha, p_one)
    if message is not None:
        raise AssertionError(message)
    return json.dumps([bool(same_gate), float(alpha), float(p_one)])


def check(solution_output: str, expected_output: str) -> None:
    got = json.loads(solution_output)
    exp = json.loads(expected_output)
    assert bool(got[0]) == bool(exp[0]), \
        "Wrong verdict -- two matrices can be the same gate without being equal."
    assert np.isclose(got[1], exp[1], atol=1e-5), \
        "Wrong phase -- did you fold np.angle into [0, 2*pi)?"
    assert np.isclose(got[2], exp[2], atol=1e-5), \
        "Wrong control probability -- check the order of the Hadamards and that " \
        "you controlled M rather than V."

# <<< END BLOCK [TEST_SUPPORT]

# >>> BLOCK [TEST_CASES] ID=None
# These are the public test cases
test_cases = [
    ('{"u": [[[0.7071067811865475, 0.0], [0.7071067811865475, 0.0]], [[0.7071067811865475, 0.0], [-0.7071067811865475, 0.0]]], "v": [[[-0.7071067811865475, 0.0], [-0.7071067811865475, 0.0]], [[-0.7071067811865475, 0.0], [0.7071067811865475, 0.0]]]}', '[true, 3.14159265359, 1.0]'),
    ('{"u": [[[0.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [0.0, 0.0]]], "v": [[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [-1.0, 0.0]]]}', '[false, 0.0, 0.5]')
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
