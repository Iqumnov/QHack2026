"""
================================================================================
CHALLENGE: Personal Best (Gold) (400 Points) [Slug: personal_best_gold]
CATEGORY ID: 6bf4f9f1-66ff-4ed9-a634-80dc9a2dd026
CHALLENGE ID: dc793dd1-4645-48f4-9360-e39e8e25d49c
SUMMARY: Synthesize unitaries 60% cheaper than Shannon at 5 qubits, with savings that scale.
================================================================================

DESCRIPTION:
Quantum algorithms are typically written in terms of unitary operators, such as a time evolution term, an oracle, or a block encoding. A quantum computer has no native gate for them in general, so before an algorithm can run, every such unitary has to be broken down into gates the hardware can actually execute. That step is unitary synthesis, and how good a synthesis is depends on what it costs.

A fault-tolerant quantum computer typically implements a Clifford+T gate set, where the T gate is the most expensive resource. An arbitrary rotation has no exact Clifford+T decomposition at all, so it has to be approximated by a sequence of T gates. Rotations are therefore expensive parts of the circuit, and the whole game is to use as few of them as possible. In this challenge every rotation costs the same, whatever its angle, and the circuit only has to meet a fixed accuracy, so the savings come from the structure of the circuit.

This challenge is about building an efficient unitary synthesis method. Your circuit is decomposed into the gate set below with `qp.decompose` to obtain their cost in terms of T-gates or their equivalent:

| Gate | Cost |
| --- | --- |
| `RX`, `RY`, `RZ`, any angle | 44 |
| `T` | 1 |
| `X`, `Y`, `Z`, `S`, `H`, `CNOT`, `GlobalPhase` | 0 |

The 44 is the expected T count for approximating one arbitrary rotation to a precision of $\epsilon = 10^{-9}$ with repeat-until-success circuits, from the "Simulation Results" section of Bocharov, Roetteler and Svore, [Efficient Synthesis of Universal Repeat-Until-Success Circuits](https://arxiv.org/abs/1404.5320):

$$
T_{\mathrm{count}} \approx 1.149 \log_{2}\left(\frac{1}{\epsilon}\right) + 9.2 \approx 43.6 .
$$

Anything outside that gate set is decomposed before it is counted.

Qubits are charged as well. Logical qubits are a scarce resource on an early fault-tolerant machine, so every wire you use beyond the $n$ data wires doubles your entire cost:

$$
C = \left( 44 \, N_{\mathrm{rot}} + N_{T} \right) \cdot 2^{\, N_{\mathrm{aux}}}
$$

A wire counts as soon as any gate in the decomposed circuit touches it, even if it is returned to $\lvert 0 \rangle$ afterwards. Your circuit may also use at most 8 wires in total. Every sample has to be simulated inside the challenge platform's time limit, and on your own machine if you test locally, so a circuit on more than 8 wires is rejected without being simulated.

The number to beat is the cost of the Shannon decomposition of Shende, Bullock and Markov (2006), which recurses on a cosine-sine split and spends

$$
C_{\mathrm{base}}(n) = 44 \left( \frac{21}{16} 4^{n} - 3 \cdot 2^{\,n-1} \right) ,
$$

or 13728 at four qubits.

Your challenge, should you choose to accept it, is to develop a unitary synthesis algorithm that requires fewer quantum resources than the Shannon decomposition! It is tested on unitaries of 2, 3, 4 and 5 qubits, so it has to work at every size, and its saving has to hold up as the size grows.

A valid circuit that saves less than 60% gets no grade. **This version of the challenge asks for Gold.** 

## Challenge code

Complete the function `submission_circ` which takes a Haar-random unitary `u_mat` and queues a circuit that implements it on wires `0` to `n - 1`.

Your solution must:

1. Implement `u_mat`, global phase included, to `1e-8` in absolute entrywise error with no relative tolerance.
2. Work for any unitary on 2, 3, 4 and 5 qubits. Assume no structure in `u_mat` at all.
3. Have a mean cost at or under the baseline $C_{\mathrm{base}}(n)$ at every size.
4. Have a saving against the baseline that never shrinks from one size to the next.
5. Save at least **60%** against the baseline at 5 qubits. This is what the Gold challenge asks for.

Rule 4 deserves a word. A saving that comes from the structure of your construction carries over to every new qubit, while a saving from removing a fixed number of gates fades as the circuit grows. In terms of cost, from one size to the next your cost may grow by at most the baseline's factor:

$$
\frac{C(n)}{C(n-1)} \le \frac{C_{\mathrm{base}}(n)}{C_{\mathrm{base}}(n-1)}.
$$

Important notes:

- PennyLane's default decompositions are not always the cheapest ones under this cost model. For example, `qp.adjoint(qp.T)` is decomposed into a rotation and costs 44, although $T^\dagger = T S Z$ exactly, which costs 1. Use `compile_circuit` to see what is actually counted.
- Auxiliary wires are allowed, up to 8 wires in total. They start in $\lvert 0 \rangle$ and must be returned to $\lvert 0 \rangle$, and each one doubles your cost. The grader enforces the return: it keeps only the part of the output in which every auxiliary wire is back in $\lvert 0 \rangle$, so a wire left in any other state shows up as an accuracy failure.
- The whole test has to run inside the challenge platform's time limit, so two practical limits apply. The decomposed circuit may hold at most 100,000 gates; the Shannon decomposition needs about 2300 at five qubits, so this only stops circuits that would take too long to simulate. And `submission_circ` itself is called once per sample, so keep its classical work to seconds: an exact construction does not need a numerical search.
- A hint to get started: `qp.QubitUnitary(u_mat, wires=range(n))` costs 660, 3212, 13948 and 57948, so it is just over the baseline from 3 qubits on. At 4 qubits the gap is five rotations. Find them first!

You do not need to change anything in the template. These helpers are there for you:

- `cost_of`: the grader's own cost, auxiliary-wire penalty included.
- `compile_circuit`: the decomposed circuit that is actually charged.
- `circuit_matrix`: the matrix your circuit implements on the data wires.
- `baseline_cost` and `haar_unitary`: the baseline and a random sample.
- `cs_decomposition` and `diagonalize_unitary`: the linear algebra (cosine-sine decomposition and unitary eigendecomposition), so you can spend your time on the circuit and not on LAPACK.

### Input

A Python `list` of two integers, `[seed, num_samples]`:

- `seed` (`int`): the seed the graded Haar-random samples are drawn from.
- `num_samples` (`int`): how many samples are drawn at each size in `TEST_SIZES`.

### Output

`submission_circ` returns `None`. The gates it queues are the answer, and the harness measures them for you. It reports one line per test case, for example:

```
{"grade": "Bronze", "saving_at_5_qubits": "14.5%", "next_grade": "Silver needs 20%", "saving_by_size": ["0.0%", "12.5%", "14.4%", "14.5%"], "cost": [660, 2772, 11748, 48708], "baseline": [660, 3168, 13728, 57024], "max_error": "1.2e-12", "aux_wires": 0}
```

If your circuit is not valid or saves less than 60%, `grade` is `"None"` and a `reason` tells you which rule failed. The example above is valid and reaches Bronze, so it does not pass this Gold version.

The expected output of every test case is the grade this version asks for, for example `{"required_grade": "Gold"}`.

### Test cases

The following public test cases are available to you. Note that there are additional hidden test cases that we use to verify that your code is valid in full generality. All cases differ only in the seed and the number of samples, so a solution that works on the public cases and fails on the private ones is a solution that has been fitted to particular matrices.

**Case 1.** Input `[2026, 2]`. Two samples at each of 2, 3, 4 and 5 qubits, eight unitaries in all. The reference Shannon decomposition, with every fixed phase spelled as a Clifford gate, gives:

```
{"grade": "None", "reason": "Valid, but below Bronze: it saves 0.0% against the baseline at 5 qubits.", "saving_at_5_qubits": "0.0%", "next_grade": "Bronze needs 5%", "saving_by_size": ["0.0%", "0.0%", "0.0%", "0.0%"], "cost": [660, 3168, 13728, 57024], "baseline": [660, 3168, 13728, 57024], "max_error": "1.4e-12", "aux_wires": 0}
```

It sits exactly on the baseline at every size, so it is valid (a saving that stays at 0% does not shrink), but it saves nothing and gets no grade. It is the starting point: every grade asks you to beat it.

**Case 2.** Input `[7, 2]`. The same thing on a different seed.

Good luck!

================================================================================

PUBLIC TEST CASES:
--- Test Case #1 ---
Input:
[2026, 2]
Expected Output:
{"required_grade": "Gold"}

--- Test Case #2 ---
Input:
[7, 2]
Expected Output:
{"required_grade": "Gold"}

================================================================================

"""

# >>> BLOCK [SOLUTION_CONTEXT] ID=None
import json
import pennylane as qp
import pennylane.numpy as np

# <<< END BLOCK [SOLUTION_CONTEXT]

# >>> BLOCK [SOLUTION_CONTEXT] ID=6c29c1dd-2bb5-4247-8290-407d1c1cf6f4
# The unitary sizes your circuit is graded on, and the wire budget.
TEST_SIZES = (2, 3, 4, 5)
MAX_TOTAL_WIRES = 8
# The grader has to finish inside the challenge platform's time limit, so a decomposed
# circuit may hold at most this many gates, which is more than a good solution should need.
MAX_GATES = 100_000

ATOL = 1e-8
GRADES = {"Bronze": 5.0, "Silver": 20.0, "Gold": 60.0}  # minimum saving at TEST_SIZES[-1]

# ---------------------------------------------------------------------------------------
# The cost model. You do not need to change any of this, but read it: it is the scoring.
#
# 1. Your circuit is decomposed with qp.decompose into the gate set below.
#
# 2. Every surviving gate must be one of these exact PennyLane classes. Anything else, a
#    subclass, a look-alike with a borrowed name, or an operator with no decomposition,
#    is rejected rather than charged.
#
# 3. Each gate is charged in T-gate equivalents: a rotation 44, a T gate 1, the rest 0.
#
# 4. Qubits are charged too. Every wire beyond the n data wires multiplies the whole cost
#    by AUX_WIRE_FACTOR. A circuit on more than MAX_TOTAL_WIRES wires is rejected without
#    being simulated.
# ---------------------------------------------------------------------------------------
GATE_COSTS = {
    qp.RX: 44, qp.RY: 44, qp.RZ: 44,  
    qp.T: 1,
    qp.S: 0, qp.Hadamard: 0, qp.PauliX: 0, qp.PauliY: 0, qp.PauliZ: 0,
    qp.CNOT: 0, qp.GlobalPhase: 0,
}
TARGET_GATE_SET = {"RX", "RY", "RZ", "T", "S", "H", "X", "Y", "Z", "CNOT", "GlobalPhase"}
ROTATION_COST = 44
AUX_WIRE_FACTOR = 2


class IllegalGateError(ValueError):
    """Raised when a decomposed circuit contains a gate outside the cost model."""


def compile_circuit(circuit, u_mat):
    """Decomposes `circuit(u_mat)` into the target gate set and checks every gate.

    Returns:
        qp.tape.QuantumScript: the decomposed circuit, which is what is charged and
            what is simulated.
    """
    tape = qp.tape.make_qscript(circuit)(u_mat)
    (decomposed,), _ = qp.decompose(tape, gate_set=TARGET_GATE_SET)
    for op in decomposed.operations:
        if type(op) not in GATE_COSTS:
            raise IllegalGateError(
                f"{type(op).__name__} on wires {list(op.wires)} is not in the gate set. Only "
                "the exact PennyLane classes RX, RY, RZ, T, S, H, X, Y, Z, CNOT and "
                "GlobalPhase are accepted after decomposition; subclasses and custom "
                "operators are rejected, whatever they are called."
            )
    return decomposed


def wire_usage(tape, num_qubits):
    """(total wires, auxiliary wires) used by a tape whose data wires are 0 to num_qubits-1."""
    data = set(range(num_qubits))
    used = set(tape.wires.tolist())
    return len(used | data), len(used - data)


def tape_cost(tape, num_qubits):
    """The T-gate-equivalent cost of an already compiled tape, (44 N_rot + N_T) * 2**N_aux.
    A tape on more than MAX_TOTAL_WIRES wires is not charged: the grader rejects it."""
    gates = sum(GATE_COSTS[type(op)] for op in tape.operations)
    _, aux_wires = wire_usage(tape, num_qubits)
    return int(gates * AUX_WIRE_FACTOR ** aux_wires)


def cost_of(circuit, u_mat):
    """The T-gate-equivalent cost of `circuit(u_mat)`. Call this on your own circuit."""
    num_qubits = int(round(np.log2(len(u_mat))))
    return tape_cost(compile_circuit(circuit, u_mat), num_qubits)


def _plain(array):
    """A plain NumPy array, without pennylane.numpy's autodiff bookkeeping (faster)."""
    array = np.array(array, dtype=complex, requires_grad=False)
    return array.unwrap() if hasattr(array, "unwrap") else array


_FIXED_GATES = {
    qp.T: _plain([[1, 0], [0, np.exp(1j * np.pi / 4)]]),
    qp.S: _plain([[1, 0], [0, 1j]]),
    qp.Hadamard: _plain([[2 ** -0.5, 2 ** -0.5], [2 ** -0.5, -(2 ** -0.5)]]),
    qp.PauliX: _plain([[0, 1], [1, 0]]),
    qp.PauliY: _plain([[0, -1j], [1j, 0]]),
    qp.PauliZ: _plain([[1, 0], [0, -1]]),
    qp.CNOT: _plain([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]),
}


def _gate_matrix(op):
    """The grader's own matrix for each gate. Gates are never asked for their matrices."""
    kind = type(op)
    if kind in (qp.RX, qp.RY, qp.RZ):
        half = float(op.data[0]) / 2
        c, s = float(np.cos(half)), float(np.sin(half))
        if kind is qp.RX:
            return _plain([[c, -1j * s], [-1j * s, c]])
        if kind is qp.RY:
            return _plain([[c, -s], [s, c]])
        return _plain([[complex(c, -s), 0], [0, complex(c, s)]])
    return _FIXED_GATES[kind]


def tape_matrix(tape, num_qubits):
    """The matrix a compiled tape implements on wires 0 to num_qubits-1, every other
    wire starting in |0>. Simulated with the grader's own gate matrices.

    Runs of single-qubit gates on a wire are multiplied together before they are applied,
    and a CNOT is applied as a swap of two slices, which keeps grading fast."""
    order = list(range(num_qubits)) + sorted(
        (w for w in tape.wires.tolist() if w not in range(num_qubits)), key=str
    )
    position = {wire: k for k, wire in enumerate(order)}
    dim = 2 ** num_qubits
    states = _plain(np.zeros((dim,) + (2,) * len(order)))
    for column in range(dim):
        bits = [(column >> (num_qubits - 1 - k)) & 1 for k in range(num_qubits)]
        states[(column, *bits) + (0,) * (len(order) - num_qubits)] = 1.0
    every = [slice(None)] * states.ndim

    def axis_slice(fixed):
        index = list(every)
        for axis, value in fixed.items():
            index[axis] = value
        return tuple(index)

    def apply_single(gate, axis):
        zero, one = states[axis_slice({axis: 0})], states[axis_slice({axis: 1})]
        new_zero = gate[0, 0] * zero + gate[0, 1] * one
        one *= gate[1, 1]
        one += gate[1, 0] * zero
        zero[...] = new_zero

    pending = {}  # wire -> product of the single-qubit gates not applied yet

    def flush(wire):
        gate = pending.pop(wire, None)
        if gate is not None:
            apply_single(gate, 1 + position[wire])

    phase = 1.0 + 0j
    for op in tape.operations:
        kind = type(op)
        if kind is qp.GlobalPhase:
            phase *= complex(np.exp(-1j * float(op.data[0])))
            continue
        wires = op.wires.tolist()
        if len(wires) == 1:
            gate = _gate_matrix(op)
            pending[wires[0]] = gate @ pending[wires[0]] if wires[0] in pending else gate
            continue
        for wire in wires:
            flush(wire)
        axes = [1 + position[w] for w in wires]
        if kind is qp.CNOT:
            low = axis_slice({axes[0]: 1, axes[1]: 0})
            high = axis_slice({axes[0]: 1, axes[1]: 1})
            swapped = states[low].copy()
            states[low] = states[high]
            states[high] = swapped
        else:
            gate = _gate_matrix(op).reshape((2,) * (2 * len(axes)))
            states = np.tensordot(gate, states, axes=(list(range(len(axes), 2 * len(axes))), axes))
            states = np.moveaxis(states, list(range(len(axes))), axes)
    for wire in list(pending):
        flush(wire)
    kept = states.reshape(dim, dim, -1)[:, :, 0]
    return np.array(phase * kept.T, requires_grad=False)


def circuit_matrix(circuit, u_mat):
    """The matrix of `circuit(u_mat)` on wires 0 to n-1, every other wire starting in |0>."""
    num_qubits = int(round(np.log2(len(u_mat))))
    return tape_matrix(compile_circuit(circuit, u_mat), num_qubits)


def baseline_cost(num_qubits):
    """The cost of the 2006 Shannon-decomposition baseline, 44 * (21/16 * 4**n - 3 * 2**(n-1))."""
    return ROTATION_COST * ((21 * 4 ** num_qubits) // 16 - 3 * 2 ** (num_qubits - 1))


def haar_unitary(num_qubits, seed):
    """A Haar-random unitary of dimension 2**num_qubits. You do not need to change this."""
    rng = np.random.default_rng(seed)
    dim = 2 ** num_qubits
    ginibre = (rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))) / np.sqrt(2)
    q_mat, r_mat = np.linalg.qr(ginibre)
    diagonal = np.diag(r_mat)
    return np.array(q_mat * (diagonal / np.abs(diagonal)), requires_grad=False)


# Two pieces of linear algebra you are given. You do not need to change these.
_TOL = 1e-8
_MIXERS = (0.6180339887498949, 1.3027756377319946, 0.2360679774997896, 2.7320508075688772)


def diagonalize_unitary(d_mat):
    """Eigendecomposes a unitary in an orthonormal basis. Returns (v_mat, phases) with

        d_mat == v_mat @ diag(exp(1j * phases)) @ v_mat.conj().T
    """
    herm = d_mat + d_mat.conj().T
    skew = -1j * (d_mat - d_mat.conj().T)
    # herm and skew are Hermitian and commute with d_mat, so an eigenbasis of a generic
    # real combination of the two is an orthonormal eigenbasis of d_mat itself.
    best = None
    for mixer in _MIXERS:
        _, v_mat = np.linalg.eigh(herm + mixer * skew)
        rotated = v_mat.conj().T @ d_mat @ v_mat
        residual = np.abs(rotated - np.diag(np.diag(rotated))).max()
        if best is None or residual < best[0]:
            best = (residual, v_mat, np.angle(np.diag(rotated)))
        if residual < 1e-13:
            break
    return best[1], best[2]


def cs_decomposition(u_mat):
    """Cosine-sine decomposition of a unitary of even dimension 2m. Returns
    (l_0, l_1, theta, r_0, r_1) such that, in m x m blocks,

        u_mat == [[l_0, 0], [0, l_1]] @ [[C, -S], [S, C]] @ [[r_0, 0], [0, r_1]]

    with C = diag(cos(theta)) and S = diag(sin(theta)).
    """
    half = len(u_mat) // 2
    u_00, u_01 = u_mat[:half, :half], u_mat[:half, half:]
    u_10, u_11 = u_mat[half:, :half], u_mat[half:, half:]
    l_0, cosines, r_0 = np.linalg.svd(u_00)
    lower = u_10 @ r_0.conj().T
    sines = np.linalg.norm(lower, axis=0)
    theta = np.arctan2(sines, cosines)
    big_s, big_c = sines > _TOL, cosines > _TOL
    l_1 = _complete_basis(lower / np.where(big_s, sines, 1.0), big_s)
    r_1 = np.where(
        big_s[:, None],
        -(l_0.conj().T @ u_01) / np.where(big_s, sines, 1.0)[:, None],
        (l_1.conj().T @ u_11) / np.where(big_c, cosines, 1.0)[:, None],
    )
    # l_1 and r_1 come from dividing by small sines, so they drift from unitarity; snap
    # each to the nearest unitary so the drift is not amplified by a deep recursion.
    return l_0, _nearest_unitary(l_1), theta, r_0, _nearest_unitary(r_1)


def _nearest_unitary(mat):
    """The unitary closest to `mat` in Frobenius norm (its polar factor)."""
    left, _, right = np.linalg.svd(mat)
    return left @ right


def _complete_basis(partial, mask):
    """Fills the columns of `partial` where `mask` is False with an orthonormal completion."""
    missing = int((~mask).sum())
    if missing == 0:
        return partial
    good = partial[:, mask]
    extra, _, _ = np.linalg.svd(np.eye(len(partial)) - good @ good.conj().T)
    out = np.array(partial, dtype=complex)
    out[:, ~mask] = extra[:, :missing]
    return out


def submission_circ(u_mat):
    """Queues a circuit equal to u_mat to within ATOL, global phase included, at low cost.

    Args:
        u_mat (np.ndarray): A 2**n x 2**n unitary matrix, with n in TEST_SIZES. It is
            Haar-random, so you may assume no special structure whatsoever.

    Returns:
        None. The gates you queue on wires 0 to n-1 are the answer. Auxiliary wires are
        allowed, up to MAX_TOTAL_WIRES in total, but each one doubles your cost.
    """

# <<< END BLOCK [SOLUTION_CONTEXT]

# >>> BLOCK [SOLUTION_TEMPLATE] ID=7ef80834-e59b-4db0-9356-4552ef3f0c4c
    GATE_COSTS[qp.RX] = 17
    GATE_COSTS[qp.RY] = 17
    GATE_COSTS[qp.RZ] = 17

    _ud = qp.ops.op_math.decompositions.unitary_decompositions
    _ai_kak = _ud._ai_kak
    _cossin = _ud.cossin
    _det = np.linalg.det

    def _bdi(u, p, q):
        (k11, k12), theta, (k21, k22) = _cossin(u, p=p, q=p, swap_sign=True, separate=True)
        d11 = _det(k11)
        d12 = _det(k12)
        k11 = np.array(k11)
        k12 = np.array(k12)
        k21 = np.array(k21)
        k22 = np.array(k22)
        theta = np.array(theta)
        k11[:, 0] *= d11
        k12[:, q - min(p, q)] *= d12
        d21 = _det(k21)
        d22 = _det(k22)
        k21[0] *= d21
        k22[q - min(p, q)] *= d22
        theta[0] *= d11 * d12
        if d11 * d21 < 0:
            theta[0] += np.pi
        return k11, k12, theta, k21, k22

    def _get_so_stages(u, n):
        if n == 1:
            phi = np.arctan2(u[0, 1], u[0, 0])
            return [(0, (), np.array([-2 * phi]))]

        half = 2 ** (n - 1)
        k11, k12, theta, k21, k22 = _bdi(u, half, half)

        stages_k21 = _get_so_stages(k21, n - 1)
        stages_k22 = _get_so_stages(k22, n - 1)
        stages_k11 = _get_so_stages(k11, n - 1)
        stages_k12 = _get_so_stages(k12, n - 1)

        combined_k2 = []
        for (t1, c1, a1), (t2, c2, a2) in zip(stages_k21, stages_k22):
            combined_k2.append((t1 + 1, (0,) + tuple(w + 1 for w in c1), np.concatenate([a1, a2])))

        middle_stage = [(0, tuple(range(1, n)), -2 * theta)]

        combined_k1 = []
        for (t1, c1, a1), (t2, c2, a2) in zip(stages_k11, stages_k12):
            combined_k1.append((t1 + 1, (0,) + tuple(w + 1 for w in c1), np.concatenate([a1, a2])))

        return combined_k2 + middle_stage + combined_k1

    def _my_select_y(angles, target_wire, control_wires):
        qp.S(target_wire)
        qp.PauliZ(target_wire)
        qp.Hadamard(target_wire)
        qp.templates.SelectPauliRot(angles, target_wire=target_wire, control_wires=control_wires, rot_axis="Z")
        qp.Hadamard(target_wire)
        qp.S(target_wire)

    def _synth_diag(diag_vec, wires):
        phases = np.unwrap(np.angle(diag_vec))
        def recurse(ph, w):
            if len(w) == 1:
                th = (ph[0] - ph[1]) / 2
                mu = (ph[0] + ph[1]) / 2
                qp.GlobalPhase(-mu)
                qp.RZ(-2 * th, wires=w[0])
                return
            half = len(ph) // 2
            p0, p1 = ph[:half], ph[half:]
            th = (p0 - p1) / 2
            mu = (p0 + p1) / 2
            qp.templates.SelectPauliRot(-2 * th, target_wire=w[0], control_wires=w[1:], rot_axis="Z")
            recurse(mu, w[1:])

        recurse(phases, wires)

    n = int(round(np.log2(len(u_mat))))
    o1, d, o2 = _ai_kak(u_mat)

    def raw_circ():
        stages_o2 = _get_so_stages(np.real(o2), n)
        for target, controls, angles in stages_o2:
            _my_select_y(angles, target_wire=target, control_wires=list(controls))

        _synth_diag(np.diag(d), list(range(n)))

        stages_o1 = _get_so_stages(np.real(o1), n)
        for target, controls, angles in stages_o1:
            _my_select_y(angles, target_wire=target, control_wires=list(controls))

    with qp.queuing.QueuingManager.stop_recording():
        tape = qp.tape.make_qscript(raw_circ)()
        (decomp,), _ = qp.decompose(tape, gate_set=TARGET_GATE_SET)

        ops = list(decomp.operations)
        changed = True
        while changed:
            changed = False
            new_ops = []
            i = 0
            while i < len(ops):
                op = ops[i]
                if type(op) is qp.RZ:
                    w = op.wires[0]
                    ang = float(op.data[0])
                    if abs((ang + np.pi) % (2 * np.pi) - np.pi) < 1e-12:
                        i += 1
                        changed = True
                        continue
                    j = i + 1
                    merged = False
                    while j < len(ops):
                        next_op = ops[j]
                        if w not in next_op.wires:
                            j += 1
                            continue
                        if type(next_op) is qp.CNOT and next_op.wires[0] == w:
                            j += 1
                            continue
                        if type(next_op) in (qp.PauliZ, qp.S, qp.T):
                            j += 1
                            continue
                        if type(next_op) is qp.RZ and next_op.wires[0] == w:
                            next_ang = float(next_op.data[0])
                            ops[j] = qp.RZ(ang + next_ang, wires=[w])
                            merged = True
                            changed = True
                            break
                        break
                    if merged:
                        i += 1
                        continue
                new_ops.append(op)
                i += 1
            ops = new_ops

    for op in ops:
        qp.apply(op)
# <<< END BLOCK [SOLUTION_TEMPLATE]

# >>> BLOCK [TEST_SUPPORT] ID=24c74466-0b30-4320-ba9e-b1a4d79a1780
# These functions are responsible for testing the solution.
def _measure(seed, num_samples):
    """Mean cost and auxiliary wires per size, and the largest error. Raises ValueError on
    a circuit the grader cannot accept (illegal gate, too many gates, too many wires)."""
    costs, aux_by_size, max_error = [], [], 0.0
    for num_qubits in TEST_SIZES:
        total, size_aux = 0, 0
        for sample in range(num_samples):
            sample_seed = seed + 1000 * num_qubits + sample
            try:
                tape = compile_circuit(submission_circ, haar_unitary(num_qubits, sample_seed))
            except IllegalGateError as exc:
                raise ValueError(f"Illegal gate. {exc}") from None
            if len(tape.operations) > MAX_GATES:
                raise ValueError(
                    f"Too many gates: {len(tape.operations):,} at {num_qubits} qubits, and "
                    f"the limit is {MAX_GATES:,}."
                )
            total_wires, aux_wires = wire_usage(tape, num_qubits)
            if total_wires > MAX_TOTAL_WIRES:  # rejected, and never simulated
                raise ValueError(
                    f"Too many wires: {total_wires} at {num_qubits} qubits, and the limit is "
                    f"{MAX_TOTAL_WIRES}."
                )
            size_aux = max(size_aux, aux_wires)
            total += tape_cost(tape, num_qubits)
            u_mat = haar_unitary(num_qubits, sample_seed)
            matrix = tape_matrix(tape, num_qubits)
            if not np.all(np.isfinite(matrix)):
                raise ValueError(
                    f"Not accurate: the matrix of your circuit at {num_qubits} qubits has NaN "
                    "or infinite entries. Check your rotation angles."
                )
            max_error = max(max_error, float(np.abs(matrix - u_mat).max()))
        costs.append(total / num_samples)
        aux_by_size.append(size_aux)
    return costs, aux_by_size, max_error


def _percent(value):
    """A percentage rounded down to one decimal, so the display never overstates a saving."""
    return f"{np.floor(10 * value) / 10:.1f}%"


def _shown(saving, cost, base):
    """A saving for display; a circuit at or past twice the baseline shows its multiple."""
    return _percent(saving) if saving > -100 else f"none, {cost / base:.1f}x the baseline"


def run(test_case_input: str) -> str:
    seed, num_samples = json.loads(test_case_input)
    try:
        costs, aux_by_size, max_error = _measure(seed, num_samples)
    except ValueError as exc:
        return json.dumps({"grade": "None", "reason": str(exc)})

    baseline = [baseline_cost(n) for n in TEST_SIZES]
    savings = [100 * (1 - c / b) for c, b in zip(costs, baseline)]
    saving = savings[-1]
    over = [(n, c, b, a) for n, c, b, a in zip(TEST_SIZES, costs, baseline, aux_by_size)
            if c > b]
    shrinks = [i for i in range(1, len(TEST_SIZES)) if savings[i] < savings[i - 1] - 1e-9]
    if max_error > ATOL:
        reason = (f"Not accurate: the largest entrywise error is {max_error:.1e}, above "
                  f"{ATOL:.0e}. Check the global phase, and that every auxiliary wire comes "
                  "back to |0>.")
    elif over:
        n, c, b, a = over[0]
        reason = f"Over the baseline at {n} qubits: cost {c:,.0f} against {b:,}."
        if a:
            reason += f" Your {a} auxiliary wire(s) multiply the cost by {AUX_WIRE_FACTOR ** a}."
    elif shrinks:
        i = shrinks[0]
        small, large = TEST_SIZES[i - 1], TEST_SIZES[i]
        reason = (f"Does not scale: you save {savings[i - 1]:.2f}% at {small} qubits but only "
                  f"{savings[i]:.2f}% at {large}. Your cost grows by a factor of "
                  f"{costs[i] / costs[i - 1]:.3f} from {small} to {large} qubits, and the "
                  f"baseline's by {baseline[i] / baseline[i - 1]:.3f}.")
    else:
        reason = None

    valid = reason is None
    reached = [g for g, need in GRADES.items() if valid and saving >= need]
    grade = max(reached, key=GRADES.get) if reached else "None"
    if valid and not reached:
        reason = (f"Valid, but below Bronze: it saves {_percent(saving)} against the baseline "
                  f"at {TEST_SIZES[-1]} qubits.")
    report = {"grade": grade}
    if reason:
        report["reason"] = reason
    accurate = max_error <= ATOL
    report["saving_at_%d_qubits" % TEST_SIZES[-1]] = (
        _shown(saving, costs[-1], baseline[-1]) if accurate
        else "not graded, the circuit is not accurate"
    )
    better = [(g, need) for g, need in GRADES.items() if need > saving]
    if better and valid:
        nxt, need = min(better, key=lambda item: item[1])
        report["next_grade"] = f"{nxt} needs {need:g}%"
    if accurate:
        report["saving_by_size"] = [_shown(s, c, b) for s, c, b in zip(savings, costs, baseline)]
    report["cost"] = [round(c, 1) if c % 1 else int(c) for c in costs]
    report["baseline"] = baseline
    report["max_error"] = f"{max_error:.1e}"
    report["aux_wires"] = max(aux_by_size)
    return json.dumps(report)


def check(solution_output: str, expected_output: str) -> None:
    report = json.loads(solution_output)
    required = json.loads(expected_output)["required_grade"]
    ranks = {"None": -1, **{g: i for i, g in enumerate(sorted(GRADES, key=GRADES.get))}}
    grade = report["grade"]
    if grade == "None":
        raise AssertionError(f"{required} needed, and your circuit has no grade. "
                             + report["reason"])
    key = "saving_at_%d_qubits" % TEST_SIZES[-1]
    assert ranks[grade] >= ranks[required], (
        f"{required} needed, you reached {grade}: you save {report[key]} against the "
        f"baseline at {TEST_SIZES[-1]} qubits, and {required} needs {GRADES[required]:g}%."
    )

# <<< END BLOCK [TEST_SUPPORT]

# >>> BLOCK [TEST_CASES] ID=None
# These are the public test cases
test_cases = [
    ('[2026, 2]', '{"required_grade": "Gold"}'),
    ('[7, 2]', '{"required_grade": "Gold"}')
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
