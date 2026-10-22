import json
import pennylane as qp
import pennylane.numpy as np


def qft_four_qubits():
    """
    Implements the four-qubit QFT circuit from elementary gates
    """
    # Wire 0
    qp.Hadamard(wires=0)
    qp.ControlledPhaseShift(np.pi / 2, wires=[1, 0])
    qp.ControlledPhaseShift(np.pi / 4, wires=[2, 0])
    qp.ControlledPhaseShift(np.pi / 8, wires=[3, 0])

    # Wire 1
    qp.Hadamard(wires=1)
    qp.ControlledPhaseShift(np.pi / 2, wires=[2, 1])
    qp.ControlledPhaseShift(np.pi / 4, wires=[3, 1])

    # Wire 2
    qp.Hadamard(wires=2)
    qp.ControlledPhaseShift(np.pi / 2, wires=[3, 2])

    # Wire 3
    qp.Hadamard(wires=3)

    # SWAP gates to restore natural qubit order
    qp.SWAP(wires=[0, 3])
    qp.SWAP(wires=[1, 2])


# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:
    return "No output"


def check(solution_output: str, expected_output: str) -> None:
    dev = qp.device("default.qubit", wires=4)
    wires = 4

    @qp.qnode(dev)
    def user_qft():
        qft_four_qubits()
        return qp.state()

    @qp.qnode(dev)
    def circuit_qft():
        qp.QFT(wires=range(wires))
        return qp.state()

    user_matrix = qp.matrix(user_qft)()
    pl_matrix = qp.matrix(circuit_qft)()

    names = list(qp.specs(user_qft)()["resources"].gate_types.keys())

    assert (
        names.count("QFT") == 0
    ), "You are not allowed to use PennyLane's built in QFT template"

    assert np.allclose(
        user_matrix, pl_matrix, atol=1e-3
    ), "Your code does not yield the correct probabilities"


# These are the public test cases
test_cases = [("No input", "No output")]

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
