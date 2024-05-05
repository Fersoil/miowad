import numpy as np


    
class NeighboringFunc:
    """
    Neighboring Function for Kohonen Network. It simply defines which nodes' weights should be updated during the training process.
    """
    def __init__(self, initial_neighbouring_radius: float=1.0, radius_decay: bool = True, total_epochs: int=0) -> None:
        """function initializes the abstract neighbouring function

        Args:
            initial_neighbouring_radius (float, optional): The parameter of neighbouring radius in the first epoch. In the first epoch, or if there is no decay turned on, it is essentially the `sigma`. Defaults to 1.0.
            radius_decay (bool, optional): Whether the neighbouring radius should be decreased over time. Defaults to True.
            total_epochs (int, optional): total number of epochs in the training process, usefull parameter for the decay. Defaults to None.
        """        
        self.initial_neighbouring_radius = initial_neighbouring_radius
        
        if 1 - 1e-5 < initial_neighbouring_radius < 1 + 1e-5: # clipping the value to avoid numerical issues
            if self.initial_neighbouring_radius == 1:
                self.initial_neighbouring_radius = 1 + 1e-5
            else:
                self.initial_neighbouring_radius = 1 + np.sign(initial_neighbouring_radius - 1) * 1e-5
        self.radius_decay = radius_decay
        self.total_epochs = total_epochs

    def __call__(self, x: np.ndarray, y: np.ndarray, t: int) -> np.ndarray:
        raise NotImplementedError

    def __str__(self) -> str:
        return "Abstract Neighboring Function"
    
    def neighbouring_radius(self, current_epoch: int) -> float:        
        if not self.radius_decay:
            return self.initial_neighbouring_radius
        
        if current_epoch > self.total_epochs:
            raise ValueError("Incorrect initialization of Neighboring Function, total_epochs is not set")
        return self.initial_neighbouring_radius * np.exp(-current_epoch / self.get_time_constant())
    
    def get_time_constant(self) -> float:
        return self.total_epochs / np.log(self.initial_neighbouring_radius)



class DistNeighboringFunc(NeighboringFunc):
    def __init__(self, initial_neighbouring_radius: float=1.0, radius_decay: bool = True, total_epochs: int=0) -> None:
        super().__init__(initial_neighbouring_radius, radius_decay,  total_epochs)

    def __call__(self, x: np.ndarray, y: np.ndarray, current_epoch: int) -> np.ndarray:
        dist = np.linalg.norm(x - y)
        
        if dist <= self.neighbouring_radius(current_epoch):
            return 1
        return 0

    def __str__(self) -> str:
        return "L2 circle Neighboring Function "


class GaussianNeighboringFunc(NeighboringFunc):
    def __init__(self, initial_neighbouring_radius: float=0.5, radius_decay: bool = True, total_epochs: int=0) -> None:
        super().__init__(initial_neighbouring_radius, radius_decay,  total_epochs)
        
                
    def __call__(self, x: np.ndarray, y: np.ndarray, current_epoch: int) -> np.ndarray:
        dist = np.linalg.norm(x - y) 
        return np.exp(-(dist**2) / (2 * self.neighbouring_radius(current_epoch)**2))

    def __str__(self) -> str:
        return f"Gaussian Neighboring Function"
    

class MinusOneGaussianNeighboringFunc(NeighboringFunc):
    # https://hannibunny.github.io/orbook/preprocessing/04gaussianDerivatives.html
    def __init__(self, initial_neighbouring_radius: float=0.5, radius_decay: bool = False, total_epochs: int=0) -> None:
        super().__init__(initial_neighbouring_radius, radius_decay,  total_epochs)

    def __call__(self, x: np.ndarray, y: np.ndarray, current_epoch: int) -> np.ndarray:
        dist = np.linalg.norm(x - y)
        gaussian = np.exp(-(dist**2) / (2 * self.neighbouring_radius(current_epoch)**2))
        # need to do some scaling to make it work
        return - gaussian * (dist**2 / (self.neighbouring_radius(current_epoch)**2) - 1)

        
        return - gaussian * (dist**2 / (self.neighbouring_radius(current_epoch)**4) - 1/(self.neighbouring_radius(current_epoch)**2))

    def __str__(self) -> str:
        return "Minus One Gaussian Neighboring Function"


class MexicanSombreroNeighboringFunc(NeighboringFunc):
    def __init__(self, initial_neighbouring_radius: float=1.0, radius_decay: bool = True, total_epochs: int=0) -> None:
        super().__init__(initial_neighbouring_radius, radius_decay,  total_epochs)
        if radius_decay:
            print("The decay is not implemented yet for this function")


    def __call__(self, x: np.ndarray, y: np.ndarray, t: int) -> np.ndarray:
        dist = np.linalg.norm(x - y)
        return (2 - 4 * dist**2) * np.exp(-(dist**2))

    def __str__(self) -> str:
        return "Mexican Sombrero Neighboring Function"
