"""
================================================================================
CHALLENGE: Pulse Optimization (400 Points) [Slug: pulse_programming]
CATEGORY ID: 6bf4f9f1-66ff-4ed9-a634-80dc9a2dd026
CHALLENGE ID: d4b32a28-ad23-490c-95d4-e1c74ac3f5b9
SUMMARY: Familiarize yourself with pulse programs.
================================================================================

DESCRIPTION:
## Challenge statement

Pulse programming is used as a method of controlling quantum operations in quantum systems. Essentially, a time-dependent electromagnetic field is applied to the quantum degrees of freedom, the state of whom are governed by time-dependent Schrödinger equation:

$$
\frac{\partial}{\partial t} \vert \psi \rangle = -i H(t) \vert \psi \rangle,
$$

where $H$ is the Hamiltonian that describes the system interacting with its environment. 

In this challenge, we're going to give you a Hamiltonian with tunable parameters (akin to tuning the electromagnetic field) and ask *you* to turn the knobs, let a quantum state evolve for some time, and generate a desired output state. You'll want to have the [documentation for `qp.pulse`](https://docs.pennylane.ai/en/stable/code/qml_pulse.html) handy if you're not familiar with it already!

Ok, so here's the deal. You're in the lab with a one-qubit pulse-programmable quantum computer. The Hamiltonian governing your system is this:

$$
H(t) = p_0t X + p_1t^2 Y + p_2t^3 Z,
$$

where $t = 2$ — this is hard-coded — and $p_i$ are tunable parameters. Your job is to evolve the $\vert 0 \rangle$ state for the given time and obtain a given target state, $\vert \psi \rangle$:

$$
\exp(-iH(t)t) \vert 0 \rangle = \vert \psi \rangle.
$$

To do this, you have to turn some knobs and figure out what the optimal values of $p_i$ are by maximizing the fidelity between $\vert \psi \rangle$ and $\exp(-iH(t)t) \vert 0 \rangle$ (or, equivalently, minimizing the *negative* of the fidelity) with respect to the parameters $p_i$:

$$
F = \vert \langle \psi \vert \exp(-iH(t)t) \vert 0 \rangle \ \vert^2.
$$

The maximum value of $F$ is $1$.

## Challenge code

In the code below you must complete the following functions:

1. `hamiltonian`: this function creates the time-dependent Hamiltonian provided in the description above. Create it similarly to what's outlined in the [`qp.pulse` documentation](https://docs.pennylane.ai/en/stable/code/qml_pulse.html).
2. `circuit`: this is a QNode that will only have one operation: an instance of `qp.evolve` that requires the Hamiltonian from the `hamiltonian` function. This QNode returns `qp.state()`.
3. `cost`: this function takes the output of `circuit` and `target_state` and calculates the fidelity. Check out the [`qp.math.fidelity_statevector` function in PennyLane's documentation](https://docs.pennylane.ai/en/stable/code/api/pennylane.math.fidelity_statevector.html).

### Input

As input to this challenge, you are given the target state (`list(float)`) that `circuit` must produce.

### Output

Your code will output the parameters $p_i$ (`list(float)`) such that $\exp(-iH(t)t) \vert 0 \rangle = \vert \psi \rangle$.

### Test cases

The following **public test cases** are available to you. Note that there are additional **hidden test cases** that we use to verify that your code is valid in full generality.

```python
test_input: [0.11775969, 0.99304212]
expected_output: [0.0, 0.062368193925808904, 0.0]
test_input: [0.7071067812, 0.7071067812]
sample_output:, [0.0, 0.26666667860250887, 0.0], 
```

Your `circuit` QNode must *only* contain 1 operation (an instance of `qp.evolve`). If your solution matches the correct one within the `1e-4` relative error tolerance, the output will be `"Success!"`. Otherwise, you will receive an `"Incorrect"` prompt.

Good luck!


================================================================================

PUBLIC TEST CASES:
--- Test Case #1 ---
Input:
[0.11775969, 0.99304212]
Expected Output:
[0.0, 0.062368193925808904, 0.0]

--- Test Case #2 ---
Input:
[0.7071067812, 0.7071067812]
Expected Output:
[0.0, 0.26666667860250887, 0.0]

================================================================================

"""

# >>> BLOCK [SOLUTION_CONTEXT] ID=None
import jax
import jax.numpy as jnp
import json
import pennylane as qp

# <<< END BLOCK [SOLUTION_CONTEXT]

# >>> BLOCK [SOLUTION_CONTEXT] ID=dda47230-9b1e-403c-aec6-ce3a826bde5d
jax.config.update("jax_enable_x64", True)
dev = qp.device("default.qubit", wires=1)
t = jnp.array(2)

@qp.qnode(dev)
def target_state(state):
    """ A QNode that prepares the target state.
    
    Args:
    state (numpy.tensor): The 1-qubit target state.
    Returns:
    (numpy.tensor): The same state.
    """
    qp.StatePrep(state, wires=0)
    return qp.state()


def hamiltonian():
    """Creates the time-dependent Hamiltonian: H(t) = p0 t X + p1 t^2 Y + p2 t^3 Z
    
    Returns:
    (qp.pulse.parametrized_hamiltonian.ParametrizedHamiltonian): the Hamiltonian.
    """

# <<< END BLOCK [SOLUTION_CONTEXT]

# >>> BLOCK [SOLUTION_TEMPLATE] ID=132dd0c0-d2df-4663-b5e5-c08eb7d6007b
    def f0(p, t):
        return p * t
    def f1(p, t):
        return p * (t ** 2)
    def f2(p, t):
        return p * (t ** 3)
    return f0 * qp.PauliX(0) + f1 * qp.PauliY(0) + f2 * qp.PauliZ(0)

# <<< END BLOCK [SOLUTION_TEMPLATE]

# >>> BLOCK [SOLUTION_CONTEXT] ID=a6d3d73d-6956-40d7-9013-4c9b913008ac
@qp.qnode(dev)
def circuit(params):
    """A QNode that evolves the |0> state under the time-dependent Hamiltonian.
    This QNode must contain one operation only: an instance of qp.evolve.
    
    Args:
    params (list(numpy.tensor)): The parameters p0, p1, and p2 (in that order).
    Returns:
    (numpy.tensor): The evolved state.
    """

# <<< END BLOCK [SOLUTION_CONTEXT]

# >>> BLOCK [SOLUTION_TEMPLATE] ID=4cbbec6c-ca36-4a4a-8b23-1246fe5c6b5c
    qp.evolve(hamiltonian())(params, t=2)
    return qp.state()

# <<< END BLOCK [SOLUTION_TEMPLATE]

# >>> BLOCK [SOLUTION_CONTEXT] ID=7d3bfc11-d70d-4d39-b8f2-f3e26ef8b2f7
def optimal_parameters(state):
    """Optimizes the parameters p0, p1, and p2.

    Args:
        state (numpy.tensor): The 1-qubit target state.

    Returns:
        params (list(float)): The optimized parameters p0, p1, and p2 (in that order).
    """

    def cost(params):
        """Computes the negative fidelity between the output of circuit and the target state
        
        args: 
            params (list(numpy.tensor)): The parameters p0, p1, and p2 (in that order).
        returns:
            (numpy.tensor): the negative fidelity.
        """

# <<< END BLOCK [SOLUTION_CONTEXT]

# >>> BLOCK [SOLUTION_TEMPLATE] ID=20a70eb4-72a4-4c92-84ea-145143177db4
        evolved = circuit(params)
        target = target_state(state)
        return -qp.math.fidelity_statevector(evolved, target)

# <<< END BLOCK [SOLUTION_TEMPLATE]

# >>> BLOCK [SOLUTION_CONTEXT] ID=3efccd4e-a6bb-4722-bb84-cf4aa547bf42
    def update(gamma, params):
        """Updates the parameters, params, via a simple gradient descent routine.
        
        Args:
        gamma (float): the learning
        params (list(numpy.tensor)): the parameters p0, p1, and p2 (in that order)
        Returns:
        params (list(numpy.tensor)): the updated parameters.
        """

        gradient = jax.grad(cost)(params)

        for i  in range(len(params)):
            params[i] -= gradient[i] * gamma

        return params

    params = [jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)] 
    gamma = 0.1

    for i in range(20):
        params = update(gamma, params)
        if i > 10:
            gamma *= 0.9

        params = [param.tolist() for param in params]

        return params

# <<< END BLOCK [SOLUTION_CONTEXT]

# >>> BLOCK [TEST_SUPPORT] ID=24d1c28f-a805-486a-816f-c027f01e10d1
# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:
    state = json.loads(test_case_input)
    output = optimal_parameters(jnp.array(state))

    return str(output)

def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)

    tape = qp.workflow.construct_tape(circuit)([jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)])
    op_names = [op.name for op in tape.operations]

    assert op_names.count("ParametrizedEvolution") == 1, "You may only use one operation."

    assert op_names[0] == "ParametrizedEvolution", "You can only use an instance of qp.evolve in your circuit."
    assert jnp.allclose(jnp.array(expected_output), jnp.array(solution_output), rtol=1e-4), "Your calculated parameters are incorrect."

# <<< END BLOCK [TEST_SUPPORT]

# >>> BLOCK [TEST_CASES] ID=None
# These are the public test cases
test_cases = [
    ('[0.11775969, 0.99304212]', '[0.0, 0.062368193925808904, 0.0]'),
    ('[0.7071067812, 0.7071067812]', '[0.0, 0.26666667860250887, 0.0]')
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
