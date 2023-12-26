from math import tanh


def tanh_func(x: float) -> float:
    return tanh(x)


def tanh_prime_func(x: float) -> float:
    return 1 - tanh(x) ** 2
