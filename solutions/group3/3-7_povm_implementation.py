"""
================================================================================
CHALLENGE: Implementing POVMs (400 Points) [Slug: povm_implementation]
CATEGORY ID: 6bf4f9f1-66ff-4ed9-a634-80dc9a2dd026
CHALLENGE ID: c6d05117-b9ef-4bb8-8737-1ea719ce4458
SUMMARY: Use an auxiliary system to perform a generalized measurement.
================================================================================

DESCRIPTION:
## Challenge Statement

In introductory courses on quantum theory, we quickly get acquainted with the concept of a projective measurement. To get on the same page, let's quickly review the topic. 

When we measure an observable $B$ acting on a quantum system $S$, the possible outcomes are its eigenstates $\{\vert b_0\rangle, \vert b_1\rangle, \ldots, \vert b_N\rangle \},$  which form an orthonormal basis of the Hilbert space that represents $S$. Bohr's rule postulates that the probability of observing the eigenstate $\vert b_i \rangle$ when the quantum system is in the state $\vert \psi \rangle$ is 

$$ 
p(i) = \vert \langle b_i \vert \psi \rangle \vert^2.
$$

Alternatively, defining $P_i = \vert b_i\rangle\langle b_i \vert$, we can express

$$ 
p(i) = \langle \psi \vert P_i \vert \psi \rangle.
$$

Note that the operators $P_i$ are orthogonal projections, that is, they are Hermitian operators that satisfy 

- $P_i^2 = P_i,$
- $P_i P_j = 0$ when $i \neq j,$
- $\sum_{i=0}^{N}P_i = \mathbb{I}.$

We can model this type of measurement on $S$ by specifying the set $\{P_i\}$ of projectors. This set is known as a *projection-valued measure.*

In more advanced courses in quantum theory, we learn that there are more general measurements we can perform, some of which can be modelled through a *positive operator-valued measure,* or POVM for short. Generalizing the above, the measurement is specified via a set $\{E_i\}_{i=0}^{N}$ of positive hermitian operators such that

$$
\sum_{i=0}^{N}{E_i} = \mathbb{I}.
$$

The probabilities associated to the $i$-th measurement outcome are given by

$$ p(i) = \langle \psi \vert E_i \vert \psi \rangle.$$

In this challenge, we will construct a protocol that implements the measurement given by the POVM

$$
\begin{align*}
E_0 & = \alpha \vert 1 \rangle \langle 1 \vert, \\
E_1 & = \beta \vert - \rangle \langle - \vert, \\
E_2 & = \mathbb{I}-E_0 - E_1,
\end{align*}
$$

on one qubit. Here, $\alpha$ and $\beta$ are real numbers between $0$ and $1$. 

The protocol consists of coupling the qubit to an auxiliary qubit, evolving the two-qubit joint system, and finally measuring the joint system in the computational basis, as shown in the figure below. 

<p align="center">
<img src="https://s3.us-east-1.amazonaws.com/swc-prod-event-content-qhack-2026/challenge-images/povm_implementation/povm.png" width="400"/>
</p>

Denoting the probability of measuring the basis state $\vert i\rangle\vert j \rangle$ by $p_{ij}$ and the initial state of the qubit by $\vert \psi \rangle,$ the circuit should be such that

$$
\begin{align*}
p_{00} &= \langle \psi \vert E_0 \vert \psi \rangle, \\
p_{01} &= \langle \psi \vert E_1 \vert \psi \rangle, \\
p_{10} + p_{11} &= \langle \psi \vert E_2 \vert \psi \rangle.
\end{align*}
$$

Can you find the unitary that accomplishes this?

## Challenge code

In the code template shown, you must complete the circuit `implement_POVM`, which applies the unitary that implements the `alpha` and `beta`-dependent POVM. In `implement_POVM`, wire `0` must be the auxiliary qubit and wire `1` must be the system qubit.

You are also given some space to code any helper functions you may need. 
### Input

As input to this problem, you are given:

- `alpha` (`float`): Coefficient between $0$ and $1$ associated with the POVM element $E_0$
- `beta` (`float`): Coefficient between $0$ and $1$ associated with the POVM element $E_1$

### Output

Your circuit will be fed into a quantum device that measures the probabilities in the computational basis. This is done through the function `povm_circuit`, which is given for you in the challenge code.
Then, it will be run with ten random quantum states in order to verify that the output probabilities match the POVM probabilities. If they all match, the output will be `"Correct circuit"`. The relative error tolerance in the probabilities is `1e-4`.

### Test cases

The following **public test cases** are available to you. Note that there are additional **hidden test cases** that we use to verify that your code is valid in full generality.

```python
test_input: 0.25, 0.25
expected_output: "Correct circuit"

test_input: 0.2, 0.4
expected_output: "Correct circuit"
```

Good luck!


================================================================================

PUBLIC TEST CASES:
--- Test Case #1 ---
Input:
[0.25,0.25]
Expected Output:
Correct circuit

--- Test Case #2 ---
Input:
[0.2,0.4]
Expected Output:
Correct circuit

================================================================================

"""

# >>> BLOCK [SOLUTION_CONTEXT] ID=None
import json
import pennylane as qp
import pennylane.numpy as np

# <<< END BLOCK [SOLUTION_CONTEXT]

# >>> BLOCK [SOLUTION_TEMPLATE] ID=1afd4474-6f8d-41e5-9538-6d3f87b5a1a6
def get_povm_unitary(alpha, beta):
    v00 = 0.0
    v10 = np.sqrt(beta / 2.0)
    v20 = np.sqrt(1.0 - beta / 2.0)
    v30 = 0.0

    v01 = np.sqrt(alpha)
    v11 = -np.sqrt(beta / 2.0)
    v21 = (beta / 2.0) / np.sqrt(1.0 - beta / 2.0)
    rem = 1.0 - (v01**2 + v11**2 + v21**2)
    v31 = np.sqrt(max(0.0, rem))

    V = np.array([
        [v00, v01],
        [v10, v11],
        [v20, v21],
        [v30, v31]
    ], dtype=float)

    Q, _ = np.linalg.qr(V, mode='complete')
    for c in range(2):
        if np.dot(Q[:, c], V[:, c]) < 0:
            Q[:, c] = -Q[:, c]
    return Q

# <<< END BLOCK [SOLUTION_TEMPLATE]

# >>> BLOCK [SOLUTION_CONTEXT] ID=02057198-242b-436d-a1a7-2db5cc203c13
def implement_POVM(alpha,beta):
    """Quantum function that implements the protocol for the POVM

    args:
    - alpha (float): coefficient between 0 and 1 associated with the E_0 element
    - beta (float): coefficient between 0 and 1 associated with the E_1 element

    returns:
    - This function is a quantum function (a list of gates to be applied) and must not return anything
    """

# <<< END BLOCK [SOLUTION_CONTEXT]

# >>> BLOCK [SOLUTION_TEMPLATE] ID=dabbd322-1118-4beb-88f3-1eb7d3390f84
    U = get_povm_unitary(alpha, beta)
    qp.QubitUnitary(U, wires=[0, 1])

# <<< END BLOCK [SOLUTION_TEMPLATE]

# >>> BLOCK [SOLUTION_CONTEXT] ID=f5f80395-3480-48dc-9a3f-f0c8475db2c5
dev = qp.device("default.qubit", wires = [0,1])

@qp.qnode(dev)
def povm_circuit(values, state):
    """
    This circuit will apply your POVM to some initial state in the first wire and will be used to 
    verify that we obtain the right probabilities

    args:
    - values: An array [alpha, beta], containing the coefficients that define the POVM
    - state: A (not normalized) vector specifying the initial joint state for the circuit

    returns:
    - np.array: The probabilities on both wires to verify that your unitary gives the right output.
    
    """

    vec = np.array([state[0],state[1],0,0])
    vec = vec/np.linalg.norm(vec)

    qp.StatePrep(vec, wires = [0,1])

    implement_POVM(*values)

    return qp.probs(wires = [0,1])

# <<< END BLOCK [SOLUTION_CONTEXT]

# >>> BLOCK [TEST_SUPPORT] ID=67b74065-8235-4c91-acce-2fbc777bf5b5
# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:

    ins = json.loads(test_case_input)
    alpha = ins[0]
    beta = ins[1]
    def e0_probs(alpha, vec):

        vec = vec/np.linalg.norm(vec)
        e0 = np.kron([0,1],[[0],[1]])
        return alpha * (vec@e0@vec)

    def e1_probs(beta, vec):

        vec = vec/np.linalg.norm(vec)
        e1 = 1/2 * np.kron([1,-1],[[1],[-1]])
        return beta * (vec@e1@vec)

    for i in range(10):
        v = np.random.rand(1,2)[0]
        v = v/np.linalg.norm(v)
        if not np.allclose(povm_circuit(ins,v)[:2], [e0_probs(alpha,v), e1_probs(beta,v)]):
            return "Incorrect circuit"

    return "Correct circuit"

def check(solution_output: str, expected_output: str) -> None:
    solution_output = solution_output
    expected_output = expected_output

    assert solution_output == expected_output, "Not the correct POVM implementation."

# <<< END BLOCK [TEST_SUPPORT]

# >>> BLOCK [TEST_CASES] ID=None
# These are the public test cases
test_cases = [
    ('[0.25,0.25]', 'Correct circuit'),
    ('[0.2,0.4]', 'Correct circuit')
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
