import json
import pennylane as qp
import pennylane.numpy as np

# A three-qubit state is prepared for you. You do not need to change this.
def prepare(angles):
    """Prepares the state whose Pauli expectation values you have to measure.

    Args:
        angles (list[float]): four rotation angles.
    """
    qp.RY(angles[0], wires=0)
    qp.RX(angles[1], wires=1)
    qp.RY(angles[2], wires=2)
    qp.CNOT(wires=[0, 1])
    qp.RZ(angles[3], wires=1)
    qp.CNOT(wires=[1, 2])
    qp.RX(angles[3], wires=0)


def measure_pauli_word(word, angles):
    """Measures a Pauli word on the prepared state using only computational-basis
    probabilities.

    Args:
        word (str): three characters, each one of "I", "X", "Y", "Z". Character w
            is the Pauli factor acting on wire w.
        angles (list[float]): the four angles handed to prepare().

    Returns:
        tuple[list[str], list[int], float]:
            - the basis change applied to each wire, "none", "H" or "SH";
            - the eight signs, one per computational-basis outcome;
            - the expectation value of the Pauli word.
    """

    # --- Step 1: pick a basis change for each of the three wires -------------
    letter_to_basis = {"I": "none", "Z": "none", "X": "H", "Y": "SH"}
    basis = [letter_to_basis[letter] for letter in word]

    # --- Step 2: run one circuit and read the eight probabilities ------------
    dev = qp.device("default.qubit", wires=3)

    @qp.qnode(dev)
    def circuit():
        prepare(angles)
        for w, op in enumerate(basis):
            if op == "H":
                qp.Hadamard(wires=w)
            elif op == "SH":
                qp.adjoint(qp.S(wires=w))
                qp.Hadamard(wires=w)
        return qp.probs(wires=[0, 1, 2])

    probabilities = circuit()

    # --- Step 3: work out the sign of each of the eight outcomes -------------
    carried = [wire for wire, letter in enumerate(word) if letter != "I"]
    signs = [int((-1) ** sum((b >> (2 - wire)) & 1 for wire in carried)) for b in range(8)]

    # --- Step 4: combine them into the expectation value ---------------------
    value = float(sum(probability * sign for probability, sign in zip(probabilities, signs)))
    return basis, signs, value


# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:
    loads, dumps = json.loads, json.dumps
    make_device, script = qp.device, qp.tape.QuantumScript
    measure_probs, execute_tapes = qp.probs, qp.execute
    matrix_of, build_reference = qp.matrix, qp.tape.make_qscript
    allowed_basis = ("none", "H", "SH")
    word, angles = loads(test_case_input)
    log = []
    tally = {"recorded": 0}

    def watcher(real_execute):
        def execute(self, circuits, *inner_args, **inner_kwargs):
            results = real_execute(self, circuits, *inner_args, **inner_kwargs)
            single = isinstance(circuits, script)
            for tape in ([circuits] if single else list(circuits)):
                for measurement in tape.measurements:
                    log.append((type(measurement).__name__, tape.shots.total_shots,
                                measurement.obs is not None, tuple(measurement.wires),
                                len(tape.measurements), list(tape.operations)))
                    tally["recorded"] += 1
            return results
        return execute

    watched = []
    for holder in [qp.plugin_devices, qp.devices.__dict__]:
        for name in sorted(holder):
            try:
                candidate = holder[name]
                candidate = candidate.load() if hasattr(candidate, "load") else candidate
            except Exception:
                continue
            if not isinstance(candidate, type) or "execute" not in vars(candidate):
                continue
            real_execute = vars(candidate)["execute"]
            if callable(real_execute) and not any(candidate is seen for seen, _ in watched):
                watched.append((candidate, real_execute))
    for device_class, real_execute in watched:
        device_class.execute = watcher(real_execute)
    try:
        returned = measure_pauli_word(word, angles)
    finally:
        for device_class, real_execute in watched:
            device_class.execute = real_execute

    basis, signs, value = returned
    return dumps([basis, signs, float(value)])


def check(solution_output: str, expected_output: str) -> None:
    got = json.loads(solution_output)
    exp = json.loads(expected_output)
    assert got[0] == exp[0]
    assert got[1] == exp[1]
    assert np.isclose(got[2], exp[2], rtol=0.0, atol=1e-6)


test_cases = [
    ('["ZIZ", [0.79, 2.75, 2.89, 0.33]]', '[["none", "none", "none"], [1, -1, 1, -1, -1, 1, -1, 1], 0.931986472711103]'),
    ('["XYI", [1.45, 0.22, 1.72, 1.4]]', '[["H", "SH", "none"], [1, 1, -1, -1, -1, -1, 1, 1], 0.9673999542626387]')
]

if __name__ == "__main__":
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
