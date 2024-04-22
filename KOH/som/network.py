import numpy as np
from typing import Literal


from som.neighboring import DistNeighboringFunc, GaussianNeighboringFunc, MinusOneGaussianNeighboringFunc, MexicanSombreroNeighboringFunc


class KohonenNetwork:
    def __init__(self, M: int, N: int, neighboring_func: Literal["L2", "gaussian", "minus_one_gaussian", "mexican"] = "gaussian", 
                 vec_dim: int = 2, lambda_param: float = 10) -> None:
        
        self.M = M
        self.N = N
        self.vec_dim = vec_dim
        self.lambda_param = lambda_param

        if neighboring_func == "L2":
            self.neighboring_func = DistNeighboringFunc()
        elif neighboring_func == "gaussian":
            self.neighboring_func = GaussianNeighboringFunc()
        elif neighboring_func == "minus_one_gaussian":
            self.neighboring_func = MinusOneGaussianNeighboringFunc()
        else:
            self.neighboring_func = MexicanSombreroNeighboringFunc()    

        self.cells = np.random.rand(M, N, vec_dim)

    def fit(self, X: np.ndarray, epochs: int) -> None:
        for t in range(epochs):
            print(f"Epoch {t}")
            for x in X:
                i, j = self.find_best_matching_unit(x)
                self.update_cells(x, i, j, t)
    
    def dist(self, x: np.ndarray, y: np.ndarray) -> float:
        return np.linalg.norm(x - y)

    def find_best_matching_unit(self, x: np.ndarray) -> tuple:
        min_dist = np.inf
        best_i, best_j = 0, 0
        for i in range(self.M):
            for j in range(self.N):
                dist = self.dist(x, self.cells[i, j])
                if dist < min_dist:
                    min_dist = dist
                    best_i, best_j = i, j

        return best_i, best_j
    
    def update_cells(self, x: np.ndarray, i: int, j: int, t: int) -> None:
        for m in range(self.M):
            for n in range(self.N):
                self.cells[m, n] += self.alpha(t) * self.neighboring_func(np.array([m, n]), np.array([i, j]), t) * (x - self.cells[m, n])


    def alpha(self, t: int) -> float:
        return np.exp(-t / self.lambda_param)
