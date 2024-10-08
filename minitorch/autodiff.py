from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    x0 = vals[arg]
    vals1 = list(vals)

    vals1[arg] = x0 + epsilon
    f_plus = f(*vals1)

    vals1[arg] = x0 - epsilon
    f_minus = f(*vals1)

    return (f_plus - f_minus) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulate the derivative value."""

    @property
    def unique_id(self) -> int:
        """Returns the unique identifier for the variable."""
        ...

    def is_leaf(self) -> bool:
        """Returns True if the variable is a leaf node."""
        ...

    def is_constant(self) -> bool:
        """Returns True if the variable is a constant."""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Returns the parent variables of the current variable."""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Computes the chain rule for backpropagation.

        Args:
        ----
            d_output: The derivative of the output with respect to some variable.

        Returns:
        -------
            An iterable of tuples containing parent variables and their corresponding derivatives.

        """
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    visited = set()
    topo_order = []

    def dfs(v: Variable) -> None:
        if v.unique_id in visited or v.is_constant():
            return
        visited.add(v.unique_id)
        for parent in v.parents:
            dfs(parent)
        topo_order.append(v)

    dfs(variable)
    return reversed(topo_order)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph to compute derivatives for the leaf nodes.

    Args:
    ----
        variable: The right-most variable.
        deriv: The derivative we want to propagate backward to the leaves.

    Returns:
    -------
        None: Updates the derivative values of each leaf through accumulate_derivative`.

    """
    topo_order = topological_sort(variable)
    derivatives = {variable.unique_id: deriv}

    for v in topo_order:
        if v.is_leaf():
            v.accumulate_derivative(derivatives[v.unique_id])
        else:
            for parent, parent_deriv in v.chain_rule(derivatives[v.unique_id]):
                if parent.unique_id in derivatives:
                    derivatives[parent.unique_id] += parent_deriv
                else:
                    derivatives[parent.unique_id] = parent_deriv


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Returns the saved tensors for backpropagation."""
        return self.saved_values
