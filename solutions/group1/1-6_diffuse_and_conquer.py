import json
import pennylane as qp
import pennylane.numpy as np

# DO NOT CHANGE THESE VARIABLES LATER IN THE CODE

phi = np.array([0.03, 0.04, 0.07, 0.09])
source = np.array([1., 2., 1., 0.5])
dt = 0.01
h = 0.2

# Phase angles for polynomial approximation of C/x (kappa = 4, degree 43)
# from PennyLane's official QSVT matrix inversion implementation
phi_pyqsp = np.array([
    -2.287, 2.776, -1.163, 0.408, -0.16, -0.387, 0.385, -0.726, 0.456, 0.062,
    -0.468, 0.393, 0.028, -0.567, 0.76, -0.432, -0.011, 0.323, -0.573, 0.82,
    -1.096, 1.407, -1.735, 2.046, -2.321, 2.569, -2.819, -0.011, 2.71, -2.382,
    2.574, 0.028, -2.749, 2.673, 0.062, -2.685, 2.416, 0.385, -0.387, -0.16,
    0.408, -1.163, -0.365, 2.426
])

phi_qsvt = qp.transform_angles(phi_pyqsp, "QSP", "QSVT")
qsvt_angles = phi_qsvt
C = 0.10145775

b = phi + dt * source

# Feel free to write any helper functions here

D2 = (1.0 / (h**2)) * np.array([
    [-2,  1,  0,  0],
    [ 1, -2,  1,  0],
    [ 0,  1, -2,  1],
    [ 0,  0,  1, -2]
], dtype=float)

device = qp.device("default.qubit", wires=4)


@qp.qnode(device)
def qsvt_diffusion(nu, k):
    """
    Implements QSVT to find the concentration phi after one time step

    Args:
    - nu (float): The diffusion coefficient
    - k (float): The reaction rate

    Returns:
    - (np.array): The concentation vector on the next time step.
    """
    # Construct matrix A
    A = (1 + k * dt) * np.eye(4) - nu * dt * D2

    # Prepare normalized |b> on system wires 2 and 3
    qp.StatePrep(b / np.linalg.norm(b), wires=[2, 3])

    # LCU ancilla on wire 0 to project onto the real polynomial Re(U) = P(A)
    qp.Hadamard(wires=0)

    def qsvt_block():
        be = qp.BlockEncode(A, wires=[1, 2, 3])
        projs = [qp.PCPhase(angle, dim=4, wires=[1, 2, 3]) for angle in phi_qsvt]
        qp.QSVT(be, projs)

    qp.ctrl(qsvt_block, control=0, control_values=[0])()
    qp.ctrl(qp.adjoint(qsvt_block), control=0, control_values=[1])()

    qp.Hadamard(wires=0)

    return qp.state()


# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:
    ins = json.loads(test_case_input)
    out_state = qsvt_diffusion(*ins)[:4]
    rescaled_out = out_state * np.linalg.norm(b) / C
    output = rescaled_out / np.linalg.norm(rescaled_out)

    return str(np.real(output).tolist())

def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    fidelity = np.dot(solution_output, expected_output)
    assert fidelity >= 0.99, "Your circuit doesn't look quite right "


# These are the public test cases
test_cases = [
    ('[0.2,0.5]', '[0.27959746, 0.42869397, 0.56892447, 0.64371709]'),
    ('[1.0, 1.2]', '[0.28550356, 0.4595721 , 0.59350887, 0.59584262]')
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
                sol = json.loads(output)
                exp = json.loads(expected_output)
                fid = np.dot(sol, exp)
                print(f"Correct! Fidelity: {fid:.6f}")
