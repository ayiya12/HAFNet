from .model import HAFNet  
from .trainer import HAFNetTrainer as Trainer  
  
import os.path  
import json  
  
with open(os.path.join(os.path.dirname(__file__), 'default.json'), 'r') as file:  
    cfg = json.load(file)  
  
__all__ = ['HAFNet', 'Trainer', 'cfg']