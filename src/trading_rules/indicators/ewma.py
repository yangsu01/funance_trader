from typing import List
import pandas as pd

class EWMA():
    def __init__(
        self,
        window: int,
    ):
        self.window = window
        self.ewma = pd.Series(dtype=float)
        self.alpha = 2 / (window + 1)
    
    
    def update(self, price: float, date: pd.Timestamp) -> float:
        if not self.ewma.empty:
            self.ewma.loc[date] = self.alpha*price + (1-self.alpha) * self.ewma.iloc[-1]
            
        else:
            self.ewma.loc[date] = price
        
        return self.ewma.loc[date]
    
    
    def get_history(self) -> pd.Series:
        return self.ewma
    

    def check_enough_data(self) -> bool:
        return len(self.ewma) >= self.window