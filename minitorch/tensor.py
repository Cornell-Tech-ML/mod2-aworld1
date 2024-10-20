"""Implementation of the core Tensor object for autodifferentiation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import operators
from .autodiff import Context, Variable, backpropagate
from .tensor_data import TensorData

# Comment these out if not yet implemented
from .tensor_functions import (
    EQ,
    LT,
    Add,
    All,
    Copy,
    Exp,
    Inv,
    IsClose,
    Log,
    MatMul,
    Mul,
    Neg,
    Permute,
    ReLU,
    Sigmoid,
    Sum,
    View,
    tensor,
)

if TYPE_CHECKING:
    from typing import Any, Iterable, List, Optional, Sequence, Tuple, Type, Union

    import numpy.typing as npt

    from .tensor_data import Shape, Storage, Strides, UserIndex, UserShape, UserStrides
    from .tensor_functions import Function
    from .tensor_ops import TensorBackend

    TensorLike = Union[float, int, "Tensor"]


@dataclass
class History:
    """`History` stores the history of `Function` operations that was
    used to construct the current Variable.
    """

    last_fn: Optional[Type[Function]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Tensor] = ()


_tensor_count = 0


class Tensor:
    """Tensor is a generalization of Scalar in that it is a Variable that
    handles multidimensional arrays.
    """

    backend: TensorBackend
    history: Optional[History]
    grad: Optional[Tensor]
    _tensor: TensorData
    unique_id: int
    name: str

    def __init__(
        self,
        v: TensorData,
        back: Optional[History] = None,
        name: Optional[str] = None,
        backend: Optional[TensorBackend] = None,
    ):
        global _tensor_count
        _tensor_count += 1
        self.unique_id = _tensor_count
        assert isinstance(v, TensorData)
        assert backend is not None
        self._tensor = v
        self.history = back
        self.backend = backend
        self.grad = None
        if name is not None:
            self.name = name
        else:
            self.name = str(self.unique_id)

        self.f = backend

    def requires_grad_(self, x: bool) -> None:
        """Set whether this tensor requires gradients."""
        self.history = History()

    def requires_grad(self) -> bool:
        """Returns whether this tensor requires gradients."""
        return self.history is not None

    def to_numpy(self) -> npt.NDArray[np.float64]:
        """Returns
        Converted to numpy array

        """
        return self.contiguous()._tensor._storage.reshape(self.shape)

    def _ensure_tensor(self, b: TensorLike) -> Tensor:
        """Turns a python number into a tensor with the same backend."""
        if isinstance(b, (int, float)):
            c = Tensor.make([b], (1,), backend=self.backend)
        else:
            b._type_(self.backend)
            c = b
        return c

    def item(self) -> float:
        """Convert a 1-element tensor to a float"""
        assert self.size == 1
        x: float = self._tensor._storage[0]
        return x

    def contiguous(self) -> Tensor:
        """Return a contiguous tensor with the same data"""
        return Copy.apply(self)

    def __repr__(self) -> str:
        return self._tensor.to_string()

    def __getitem__(self, key: Union[int, UserIndex]) -> float:
        key2 = (key,) if isinstance(key, int) else key
        return self._tensor.get(key2)

    def __setitem__(self, key: Union[int, UserIndex], val: float) -> None:
        key2 = (key,) if isinstance(key, int) else key
        self._tensor.set(key2, val)

    # Internal methods used for autodiff.
    def _type_(self, backend: TensorBackend) -> None:
        self.backend = backend
        if backend.cuda:  # pragma: no cover
            self._tensor.to_cuda_()

    def _new(self, tensor_data: TensorData) -> Tensor:
        return Tensor(tensor_data, backend=self.backend)

    @staticmethod
    def make(
        storage: Union[Storage, List[float]],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
        backend: Optional[TensorBackend] = None,
    ) -> Tensor:
        """Create a new tensor from data"""
        return Tensor(TensorData(storage, shape, strides), backend=backend)

    def expand(self, other: Tensor) -> Tensor:
        """Method used to allow for backprop over broadcasting.
        This method is called when the output of `backward`
        is a different size than the input of `forward`.


        Args:
        ----
            other : backward tensor (must broadcast with self)

        Returns:
        -------
            Expanded version of `other` with the right derivatives

        """
        # Case 1: Both the same shape.
        if self.shape == other.shape:
            return other

        # Case 2: Backward is a smaller than self. Broadcast up.
        true_shape = TensorData.shape_broadcast(self.shape, other.shape)
        buf = self.zeros(true_shape)
        self.backend.id_map(other, buf)
        if self.shape == true_shape:
            return buf

        # Case 3: Still different, reduce extra dims.
        out = buf
        orig_shape = [1] * (len(out.shape) - len(self.shape)) + list(self.shape)
        for dim, shape in enumerate(out.shape):
            if orig_shape[dim] == 1 and shape != 1:
                out = self.backend.add_reduce(out, dim)
        assert out.size == self.size, f"{out.shape} {self.shape}"
        # START CODE CHANGE (2021)
        return Tensor.make(out._tensor._storage, self.shape, backend=self.backend)
        # END CODE CHANGE (2021)

    def zeros(self, shape: Optional[UserShape] = None) -> Tensor:
        """Create a new tensor filled with zeros."""

        def zero(shape: UserShape) -> Tensor:
            return Tensor.make(
                [0.0] * int(operators.prod(shape)), shape, backend=self.backend
            )

        if shape is None:
            out = zero(self.shape)
        else:
            out = zero(shape)
        out._type_(self.backend)
        return out

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        """Get the tensor data info as a tuple."""
        return self._tensor.tuple()

    def detach(self) -> Tensor:
        """Detach from backprop"""
        return Tensor(self._tensor, backend=self.backend)

    # Variable elements for backprop

    def accumulate_derivative(self, x: Any) -> None:
        """Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
        ----
            x : value to be accumulated

        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.grad is None:
            self.grad = Tensor.make(
                [0.0] * int(operators.prod(self.shape)),
                self.shape,
                backend=self.backend,
            )
        self.grad += x

    def is_leaf(self) -> bool:
        """True if this variable created by the user (no `last_fn`)"""
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """True if this variable is a constant (no gradient required)"""
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        """Returns the parents of this variable."""
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Apply the chain rule to get the derivatives of the inputs."""
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        x = h.last_fn._backward(h.ctx, d_output)
        assert len(x) == len(h.inputs), f"Bug in function {h.last_fn}"
        return [
            (inp, inp.expand(self._ensure_tensor(d_in)))
            for inp, d_in in zip(h.inputs, x)
        ]

    def backward(self, grad_output: Optional[Tensor] = None) -> None:
        """Backpropagate gradients through the computation graph."""
        if grad_output is None:
            assert self.shape == (1,), "Must provide grad_output if non-scalar"
            grad_output = Tensor.make([1.0], (1,), backend=self.backend)
        backpropagate(self, grad_output)

    def __truediv__(self, b: TensorLike) -> Tensor:
        """Element-wise division with broadcasting support."""
        return Mul.apply(self, Inv.apply(self._ensure_tensor(b)))

    def __rtruediv__(self, b: TensorLike) -> Tensor:
        """Element-wise division with broadcasting support."""
        return Mul.apply(self._ensure_tensor(b), Inv.apply(self))

    def __matmul__(self, b: Tensor) -> Tensor:
        """Not used until Module 3"""
        return MatMul.apply(self, b)

    @property
    def shape(self) -> UserShape:
        """Returns
        shape of the tensor

        """
        return self._tensor.shape

    @property
    def size(self) -> int:
        """Returns the total number of elements in the tensor."""
        return self._tensor.size

    @property
    def dims(self) -> int:
        """Returns the number of dimensions of the tensor."""
        return len(self.shape)

    def __add__(self, other: TensorLike) -> Tensor:
        """Element-wise addition with broadcasting support."""
        other_tensor = self._ensure_tensor(other)
        return Add.apply(self, other_tensor)

    def __sub__(self, other: TensorLike) -> Tensor:
        """Element-wise subtraction with broadcasting support."""
        other_tensor = self._ensure_tensor(other)
        return Add.apply(self, Neg.apply(other_tensor))

    def __mul__(self, other: TensorLike) -> Tensor:
        """Element-wise multiplication with broadcasting support."""
        other_tensor = self._ensure_tensor(other)
        return Mul.apply(self, other_tensor)

    def __lt__(self, other: TensorLike) -> Tensor:
        """Element-wise less-than comparison."""
        other_tensor = self._ensure_tensor(other)
        return LT.apply(self, other_tensor)

    def __eq__(self, other: TensorLike) -> Tensor:
        """Element-wise equality comparison."""
        other_tensor = self._ensure_tensor(other)
        return EQ.apply(self, other_tensor)

    def __gt__(self, other: TensorLike) -> Tensor:
        """Element-wise greater-than comparison."""
        other_tensor = self._ensure_tensor(other)
        return LT.apply(other_tensor, self)

    def __neg__(self) -> Tensor:
        """Element-wise negation."""
        return Neg.apply(self)

    def __radd__(self, other: TensorLike) -> Tensor:
        """Right-hand side addition to support scalar + tensor."""
        return self.__add__(other)

    def __rmul__(self, other: TensorLike) -> Tensor:
        """Right-hand side multiplication to support scalar * tensor."""
        return self.__mul__(other)

    def is_close(self, other: TensorLike) -> Tensor:
        """Element-wise closeness comparison within a tolerance."""
        other_tensor = self._ensure_tensor(other)
        return IsClose.apply(self, other_tensor)

    def sigmoid(self) -> Tensor:
        """Applies the sigmoid function element-wise."""
        return Sigmoid.apply(self)

    def relu(self) -> Tensor:
        """Applies the ReLU function element-wise."""
        return ReLU.apply(self)

    def log(self) -> Tensor:
        """Computes the natural logarithm element-wise."""
        return Log.apply(self)

    def exp(self) -> Tensor:
        """Computes the exponential function element-wise."""
        return Exp.apply(self)

    def sum(self, dim: Optional[int] = None) -> Tensor:
        """Computes the sum over the specified dimension.

        Args:
        ----
            dim (int, optional): Dimension to reduce. If None, sums over all elements.

        Returns:
        -------
            Tensor: Summed tensor.

        """
        if dim is None:
            # Sum over all elements
            flattened = self.contiguous().view(int(operators.prod(self.shape)))
            return Sum.apply(flattened, tensor([0]))
        else:
            # Sum over the specified dimension
            return Sum.apply(self, tensor([dim]))

    def mean(self, dim: Optional[int] = None) -> Tensor:
        """Computes the mean over the specified dimension.

        Args:
        ----
            dim (int, optional): Dimension to reduce. If None, computes mean over all elements.

        Returns:
        -------
            Tensor: Tensor with mean values.

        """
        total = self.sum(dim=dim)
        if dim is None:
            count = operators.prod(self.shape)
        else:
            count = self.shape[dim]
        return total / count

    def permute(self, *order: int) -> Tensor:
        """Permutes the dimensions of the tensor according to the specified order.

        Args:
        ----
            *order (int): The desired ordering of dimensions.

        Returns:
        -------
            Tensor: Permuted tensor.

        """
        order_tensor = tensor(list(order))
        return Permute.apply(self, order_tensor)

    def view(self, *shape: int) -> Tensor:
        """Reshapes the tensor to the specified shape.

        Args:
        ----
            *shape (int): The desired shape.

        Returns:
        -------
            Tensor: Reshaped tensor.

        """
        shape_tensor = tensor(list(shape))
        return View.apply(self, shape_tensor)

    def all(self, dim: Optional[int] = None) -> Tensor:
        """Checks if all elements are true (non-zero) over the specified dimension.

        Args:
        ----
            dim (int, optional): Dimension to reduce. If None, checks all elements.

        Returns:
        -------
            Tensor: Tensor with boolean values indicating if all elements are true.

        """
        if dim is None:
            # Reduce over all elements
            flattened = self.contiguous().view(int(operators.prod(self.shape)))
            return All.apply(flattened, tensor([0]))
        else:
            # Reduce over the specified dimension
            return All.apply(self, tensor([dim]))

    def zero_grad_(self) -> None:
        """Sets the gradient of the tensor to None."""
        self.grad = None
