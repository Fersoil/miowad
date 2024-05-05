import numpy as np
from typing import Literal
import networkx as nx
import matplotlib.pyplot as plt


from som.neighboring import NeighboringFunc, DistNeighboringFunc, GaussianNeighboringFunc, MinusOneGaussianNeighboringFunc, MexicanSombreroNeighboringFunc


class KohonenNetwork:
    def __init__(self, M: int, N: int, neighboring_func: NeighboringFunc = GaussianNeighboringFunc(), grid: Literal["rectangular", "hexagonal"]="rectangular",
                init_method: Literal["uniform", "random", "dataset"]="random", initial_learning_rate: float=1.0, lambda_param: float=1.0, vec_dim: int = 2, dataset: np.ndarray = None) -> None:
        
        self.M = M
        self.N = N
        self.vec_dim = vec_dim
        
        # learning rate parameters
        self.initial_learning_rate = initial_learning_rate
        self.lambda_param = lambda_param
        
        self.neighboring_func = neighboring_func
        
        self.grid = grid
        
        # plotting and positioning
        self._initial_graph = nx.grid_2d_graph(M, N)
        self.graph = nx.grid_2d_graph(M, N)
        for i in range(M):
            for j in range(N):
                if grid == "rectangular":
                    self._initial_graph.nodes[(i, j)]['pos'] = [i, j]
                if grid == "hexagonal":
                    self._initial_graph.nodes[(i, j)]['pos'] = [i * np.sqrt(3)/2, j + (j % 2) * np.sqrt(3)/2]


        self._pos = np.zeros((M, N, 2))
        for i in range(M):
            for j in range(N):
                self._pos[i, j] = np.array(self._initial_graph.nodes[(i, j)]['pos'])

        # initialize cells
        if dataset is not None:
            if dataset.shape[-1] != self.vec_dim:
                raise ValueError("Given shape of network and dataset does not match")

            self.init_cells(method=init_method, dataset=dataset)
        else:
            self.init_cells(method = init_method)


    def init_cells(self, method: Literal["random", "uniform", "dataset"], dataset: np.ndarray = None) -> None:
        if method == "random":
            self.cells = np.random.rand(self.M, self.N, self.vec_dim)
        elif method == "uniform":
            self.cells = np.zeros((self.M, self.N, self.vec_dim))
            for node in self._initial_graph.nodes:
                    node = tuple(node)
                    self.cells[node] = self._initial_graph.nodes[node]["pos"]
        elif method == "dataset":
            assert dataset is not None, "Dataset must be provided in case of method='dataset'"
            self.cells = np.zeros((self.M, self.N, self.vec_dim))
            index_array = np.arange(dataset.shape[0])
            
            sample_indexes = np.random.choice(index_array, self.M * self.N, replace=False)
            
            for i in range(self.M):
                for j in range(self.N):
                    self.cells[i, j] = dataset[sample_indexes[i * self.N + j]]

    def fit(self, X: np.ndarray, epochs: int, verbose = False, history = True) -> None:
        self.neighboring_func.total_epochs = epochs
        if history:
            labels = []
        for t in range(epochs):
            if verbose:
                print(f"Epoch {t}")
            for x in np.random.permutation(X):
                i, j = self.find_best_matching_unit(x)
                self.update_cells(x, i, j, t)
            if history:
                labels.append(self.predict(X, return_labels=True))
        
        if history:
            return labels

    def predict(self, X: np.ndarray, return_labels: bool = True) -> np.ndarray:
        res = np.array([self.find_best_matching_unit(x) for x in X])
        if not return_labels:
            return res
        else:
            return np.array([[i * self.M + j for i, j in res]]).flatten()

    
    def dist(self, x: np.ndarray, y: np.ndarray) -> float:
        return np.linalg.norm(x - y)

    def find_best_matching_unit(self, x: np.ndarray) -> tuple:

        distances = np.linalg.norm(self.cells.reshape(-1, 2) - x, axis=1)
        best_i, best_j = np.unravel_index(np.argmin(distances, axis=None), (self.M, self.N))

        return best_i, best_j
    
    def update_cells(self, x: np.ndarray, i: int, j: int, t: int) -> None:
        best_unit = self._pos[i, j]
        for m in range(self.M):
            for n in range(self.N):
                current_unit = self._pos[m, n]
                influence = self.neighboring_func(current_unit, best_unit, t)
                self.cells[m, n] += self.learning_rate(t) * influence * (x - self.cells[m, n])

    def learning_rate(self, t: int) -> float:
        # this is the alpha param (or L)
        return np.exp(-t / self.lambda_param) * self.initial_learning_rate
    
    def update_graph(self):
        for i in range(self.cells.shape[0]):
            for j in range(self.cells.shape[1]):
                if self.vec_dim == 2:
                    self.graph.nodes[(i, j)]['pos'] = [self.cells[i, j, 0], self.cells[i, j, 1]]
                elif self.vec_dim == 3:
                    self.graph.nodes[(i, j)]['pos'] = [self.cells[i, j, 0], self.cells[i, j, 1], self.cells[i, j, 2]]
                else:
                    raise ValueError("vec_dim must be 2 or 3 to plot the graph.")


    def plot_graph(self):
        self.update_graph()
        pos = nx.get_node_attributes(self.graph, 'pos') 
        if self.vec_dim == 2:
            nx.draw(self.graph, pos, with_labels=True, node_size=1000, node_color="skyblue", font_size=10, font_weight="bold", font_color="black")
        elif self.vec_dim == 3:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')

            
            nodes = np.array([pos[v] for v in self.graph.nodes])
            edges = np.array([[pos[u], pos[v]] for u, v in self.graph.edges])

            ax.scatter3D(*nodes.T, alpha=0.2, s=100, color="blue")

            for edge in edges:
                ax.plot(*edge.T, color="black")

            ax.grid(False)
            ax.set_axis_off()
            plt.tight_layout()
        else:
            raise ValueError("vec_dim must be 2 or 3 to plot the graph.")

        plt.show()
