import numpy as np
import pandas as pd
from typing import Callable, Any, List, Dict, Optional
class Bootstrapper():
    def __init__(
        self,
        data: pd.DataFrame,
        compute_func: Callable[[pd.DataFrame], Any],
        n_samples: int=100,
        sample_size: Optional[int]=None,
    ):
        self.data = data
        self.compute_func = compute_func
        self.n_samples = n_samples
        self.sample_size = sample_size or len(data)
        
    def run(self):
        pass
    
    def summary(self):
        pass