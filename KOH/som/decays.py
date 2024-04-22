import numpy as np



def base_decay_func(t: int, lambda_param: int) -> float:
    return np.exp(-t / lambda_param)


