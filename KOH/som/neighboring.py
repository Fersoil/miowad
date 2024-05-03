import numpy as np


class NeighboringFunc:
    def __init__(self) -> None:
        pass

    def __call__(self, x: np.ndarray, y: np.ndarray, t: int) -> np.ndarray:
        raise NotImplementedError

    def __str__(self) -> str:
        return "Abstract Neighboring Function"


class DistNeighboringFunc(NeighboringFunc):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x: np.ndarray, y: np.ndarray, t: int) -> np.ndarray:
        return np.dot(x, y)

    def __str__(self) -> str:
        return "L2 distance Neighboring Function"


class GaussianNeighboringFunc(NeighboringFunc):
    def __init__(self, sigma=1.0) -> None:
        # https://rdrr.io/cran/KRLS/man/gausskernel.html
        self.sigma = sigma

    def __call__(self, x: np.ndarray, y: np.ndarray, t: int) -> np.ndarray:
        dist = np.linalg.norm(x - y) * t
        return np.exp(-(dist**2) / (self.sigma**2))

    def __str__(self) -> str:
        return f"Gaussian Neighboring Function with sigma = {self.sigma}"


class MinusOneGaussianNeighboringFunc(NeighboringFunc):
    def __init__(self, sigma: float) -> None:
        super().__init__()

    def __call__(self, x: np.ndarray, y: np.ndarray, t: int) -> np.ndarray:
        dist = np.linalg.norm(x - y) * t
        return -np.exp(-(dist**2) / 2) * (dist**2 - 1) / (2 * np.pi) ** 2

    def __str__(self) -> str:
        return "Minus One Gaussian Neighboring Function"


class MexicanSombreroNeighboringFunc(NeighboringFunc):
    def __init__(self, sigma: float) -> None:
        super().__init__()

    def __call__(self, x: np.ndarray, y: np.ndarray, t: int) -> np.ndarray:
        dist = np.linalg.norm(x - y)
        return (2 - 4 * dist**2) * np.exp(-(dist**2))

    def __str__(self) -> str:
        return "Mexican Sombrero Neighboring Function"
