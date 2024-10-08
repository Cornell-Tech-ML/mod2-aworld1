"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(x: float, y: float) -> float:
    """Multiply two numbers"""
    return x * y


def id(x: float) -> float:
    """Return the input"""
    return x


def add(x: float, y: float) -> float:
    """Add two numbers"""
    return x + y


def neg(x: float) -> float:
    """Return the negative value of the input"""
    return -x


def lt(x: float, y: float) -> bool:
    """Return whether x is less than y"""
    return x < y


def eq(x: float, y: float) -> bool:
    """Return whether x is equal to y"""
    return x == y


def max(x: float, y: float) -> float:
    """Return the maximum of x and y"""
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Return whether x is within close in value to y"""
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Return the sigmoid of x"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Return the relu of x"""
    return max(0, x)


def log(x: float) -> float:
    """Return the natural logarithm of x"""
    return math.log(x)


def exp(x: float) -> float:
    """Return e^x"""
    return math.exp(x)


def log_back(x: float, y: float) -> float:
    """Return the derivative of log(x)"""
    return y / x


def inv(x: float) -> float:
    """Return the inverse of x"""
    return 1 / x


def inv_back(x: float, y: float) -> float:
    """Return the derivative of 1/x"""
    return -y / (x * x)


def relu_back(x: float, y: float) -> float:
    """Return the derivative of relu(x)"""
    return y if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(fn: Callable[[float], float], xs: Iterable[float]) -> Iterable[float]:
    """Map a function over a list"""
    return [fn(x) for x in xs]


def zipWith(
    fn: Callable[[float, float], float], xs: Iterable[float], ys: Iterable[float]
) -> Iterable[float]:
    """Apply a function to pairs of elements from two lists"""
    return [fn(x, y) for x, y in zip(xs, ys)]


def reduce(
    fn: Callable[[float, float], float], xs: Iterable[float], init: float
) -> float:
    """Reduce a list under a function"""
    acc = init
    for x in xs:
        acc = fn(acc, x)
    return acc


def negList(xs: Iterable[float]) -> Iterable[float]:
    """Negate a list of numbers"""
    return map(neg, xs)


def addLists(xs: Iterable[float], ys: Iterable[float]) -> Iterable[float]:
    """Add two lists elementwise"""
    return zipWith(add, xs, ys)


def sum(xs: Iterable[float]) -> float:
    """Sum a list of numbers"""
    return reduce(add, xs, 0)


def prod(xs: Iterable[float]) -> float:
    """Take the product of a list of numbers"""
    return reduce(mul, xs, 1)
