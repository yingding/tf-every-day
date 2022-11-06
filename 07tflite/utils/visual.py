import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass


def compare_logits(logits: dict, plt_func: callable == None):
  width = 0.35
  offset = width/2
  assert len(logits)==2

  keys = list(logits.keys())
  plt.bar(x = np.arange(len(logits[keys[0]]))-offset,
      height=logits[keys[0]], width=0.35, label=keys[0])
  plt.bar(x = np.arange(len(logits[keys[1]]))+offset,
      height=logits[keys[1]], width=0.35, label=keys[1])
  plt.legend()
  plt.grid(True)
  plt.ylabel('Logit')
  plt.xlabel('ClassID')

  delta = np.sum(np.abs(logits[keys[0]] - logits[keys[1]]))
  plt.title(f"Total difference: {delta:.3g}")
  
  if plt_func is not None:
    assert callable(getattr(plt, plt_func.__name__))
    plt_func()

@dataclass
class PlotLineData():
    x_values: list
    y_values: list
    label: str  

def display_training_loss(lines: list[PlotLineData], xlabel: str="Epoch", ylabel: str = 'Loss [Cross Entropy]',
    plt_func: callable = None
):
    # assert len(epochs) == len(losses)
    
    for line in lines:
        plt.plot(line.x_values, line.y_values, label=line.label)
    plt.ylim([0, max(plt.ylim())])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    if plt_func is not None:
        assert callable(getattr(plt, plt_func.__name__))
        plt_func()