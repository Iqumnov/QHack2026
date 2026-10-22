"""
================================================================================
CHALLENGE: My First Quantum Neural Network (100 Points) [Slug: my_first_qnn]
CATEGORY ID: 6bf4f9f1-66ff-4ed9-a634-80dc9a2dd026
CHALLENGE ID: a1e41998-fd28-467c-b43a-eceb0022adcf
SUMMARY: Build a one-qubit ansatz of a quantum neural network.
================================================================================

DESCRIPTION:
One of the primary focuses of PennyLane is quantum machine learning, in which quantum neural networks play a key role. In this coding challenge, we will construct the building blocks of a quantum neural network—the encoding block $\mathcal{S}(x)$ and the trainable block $\mathcal{A}(\vec{\theta})$:

<p align="center">
<img src="https://s3.us-east-1.amazonaws.com/swc-prod-event-content-qhack-2026/challenge-images/my_first_qnn/description_a_s.png" width="400"/>
</p>

When applied sequentially, they will constitute one layer of the quantum neural network. In this challenge, your task is to build the following single-qubit, single-layer quantum model (and perform a measurement at the end):

<p align="center">
<img src="https://s3.us-east-1.amazonaws.com/swc-prod-event-content-qhack-2026/challenge-images/my_first_qnn/s_a.png" width="400"/>
</p>

---
**NOTE**: After measurement, classical optimization is applied to update the trainable parameters $\vec{\theta}$ and potentially enable the model to learn how to solve classification or regression problems. However, this goes beyond the scope of this particular challenge.

---

## Challenge code

In the code template shown, you must complete the following quantum functions:

- `encoding_circuit`: A quantum function that applies the $\mathcal{S(x)}$ operator and returns nothing. The arguments of this function are:

    - `x` (`float`): variable that we want to encode in our quantum circuit.

- `training_circuit`: A quantum function that applies the $\mathcal{A(\vec{\theta})}$ operator and returns nothing. The arguments of this function are:

    - `theta_vec` (`np.ndarray(float)`): An array [theta_1, theta_2, theta_3] of angle parameters for the RZ, RY and RZ gates, respectively.

Then, you must complete the main function `quantum_neural_network` that combines both `encoding_circuit` and `training_circuit`:

- `quantum_neural_network`: A QNode that applies the functions `encoding_circuit` and the `training_circuit` and returns the expectation value $\langle\hat{\sigma}^z\rangle$. The arguments of this function are:

    - `x` (`float`): Variable that we want to encode in our quantum circuit.
    - `theta_vec` (`np.ndarray(float)`): An array [theta_1, theta_2, theta_3] of angle parameters for the RZ, RY and RZ gates, respectively.

### Input

As an input to this problem, you are given the variable $x$ as a `float` and the array $\vec{\theta}$  as an `np.ndarray(float)`.

### Output

The code must output an `np.tensor` that corresponds to the expectation value of the circuit `quantum_neural_network`.

### Test cases

The test cases are given as arrays of the form $[x, \vec{\theta}]$. The following **public test cases** are available for you to check your work. There are also some **hidden test cases**, which we will use to check that your solution works in full generality.

```python
test_input: [0.4859371, 1.32682562, -1.06015833, -4.41959946]
expected_output: 0.827549
test_input: [-0.72318627, -4.91395046, -5.0609724, 0.1125593]
expected_output: 0.865457
```

If your solution matches the correct one within the given tolerance specified in check (in this case it's a 1e-4 relative error tolerance), the output will be `"Success!"`. Otherwise, you will receive an `"Incorrect"` prompt.

All the best!


================================================================================

PUBLIC TEST CASES:
--- Test Case #1 ---
Input:
[0.4859371, 1.32682562, -1.06015833, -4.41959946]
Expected Output:
0.827549

--- Test Case #2 ---
Input:
[-0.72318627, -4.91395046, -5.0609724, 0.1125593]
Expected Output:
0.865457

================================================================================

"""

# >>> BLOCK [SOLUTION_CONTEXT] ID=None
import json
import pennylane as qp
import pennylane.numpy as np

# <<< END BLOCK [SOLUTION_CONTEXT]

# >>> BLOCK [SOLUTION_TEMPLATE] ID=966e7b3d-f65a-4b1d-8563-5111f38fb760
dev = qp.device("default.qubit", wires=1)

# <<< END BLOCK [SOLUTION_TEMPLATE]

# >>> BLOCK [SOLUTION_CONTEXT] ID=f94299c4-27cd-4b7b-95f7-7665a6477754
def training_circuit(theta_vec):
    """
    A function implementing the circuit A([theta_1, theta_2, theta_3]) shown in the upper part of the image. It should not return anything.

    Args:
        theta_vec (np.ndarray(float)): An array [theta_1, theta_2, theta_3] of angle
        parameters for the RZ, RY and RZ gates, respectively
        
    """

# <<< END BLOCK [SOLUTION_CONTEXT]

# >>> BLOCK [SOLUTION_TEMPLATE] ID=7c565f91-dc4f-49eb-b3ca-e339505b78b9
    qp.RZ(theta_vec[0], wires=0)
    qp.RY(theta_vec[1], wires=0)
    qp.RZ(theta_vec[2], wires=0)

# <<< END BLOCK [SOLUTION_TEMPLATE]

# >>> BLOCK [SOLUTION_CONTEXT] ID=11d8e619-eddd-4e4e-adc8-52a54b813826
def encoding_circuit(x):
    """
    A function implementing the circuit S(x) shown in the lower part of the image. It should not return anything.

    Args:
        x (float): A variable to be encoded as an angle parameter for the RX gate
    
    """

# <<< END BLOCK [SOLUTION_CONTEXT]

# >>> BLOCK [SOLUTION_TEMPLATE] ID=72cd785b-72c7-4501-8b24-7f2af2db9826
    qp.RX(x, wires=0)

# <<< END BLOCK [SOLUTION_TEMPLATE]

# >>> BLOCK [SOLUTION_CONTEXT] ID=04a9d3c0-57c9-489e-b7a4-487d17f5ad06
@qp.qnode(dev)
def quantum_neural_network(x, theta_vec):
    """
    A QNode implementing the circuits S(x) and A([theta_1, theta_2, theta_3]) subsequently and returning the  
    expectation value of the Pauli Z observable

    Args:
        x (float): A variable to be encoded as an angle parameter for the RX gate
        theta_vec (np.ndarray(float)): A list [theta_1, theta_2, theta_3] of angle parameters for the RZ, RY and RZ gates, respectively

    Returns: 
        (np.tensor): The expectation value of the Pauli Z observable
    """

# <<< END BLOCK [SOLUTION_CONTEXT]

# >>> BLOCK [SOLUTION_TEMPLATE] ID=86dc91f4-312b-4ab1-83f4-03a4672ae1e8
    encoding_circuit(x)
    training_circuit(theta_vec)
    return qp.expval(qp.PauliZ(0))

# <<< END BLOCK [SOLUTION_TEMPLATE]

# >>> BLOCK [TEST_SUPPORT] ID=e252eba9-af70-421c-8a52-d605d66bff06
# These functions are responsible for testing the solution
def run(test_case_input: str) -> str:
    ins = json.loads(test_case_input)
    output = quantum_neural_network(ins[0],ins[1:])

    return str(output)


def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    assert np.isclose(solution_output, expected_output, rtol=1e-4)

# <<< END BLOCK [TEST_SUPPORT]

# >>> BLOCK [TEST_CASES] ID=None
# These are the public test cases
test_cases = [
    ('[0.4859371, 1.32682562, -1.06015833, -4.41959946]', '0.827549'),
    ('[-0.72318627, -4.91395046, -5.0609724, 0.1125593]', '0.865457')
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
