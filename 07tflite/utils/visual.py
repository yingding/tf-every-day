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


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# def plot_images(images, predictions, true_labels, title="", plt_func: callable = None):
#     """
#     https://matplotlib.org/3.6.2/gallery/subplots_axes_and_figures/figure_title.html
#     """
#     plt.figure(figsize=(10,10))
    
#     for i in range(25):
#         plt.subplot(5,5,i+1)
#         plt.xticks([])
#         plt.yticks([])
#         plt.grid(False)
#         plt.imshow(images[i], cmap=plt.cm.binary)
#         color = 'b' if predictions[i] == true_labels[i] else 'r'
#         plt.xlabel(class_names[predictions[i]], color=color)
#         if (i == 2): # the third plot at the first row, use the title
#             plt.title(title)
#     if plt_func is not None:
#         assert callable(getattr(plt, plt_func.__name__))
#         plt_func()


def plot_images(images, predictions, true_labels, title: str, plt_func: callable = None):
    """
    https://matplotlib.org/3.6.2/gallery/subplots_axes_and_figures/figure_title.html
    """
    
    fig, axs = plt.subplots(5, 5, figsize=(6, 6), sharex=True, sharey=True, constrained_layout=True)
    fig.suptitle(title, fontsize=16)
    for i, ax in enumerate(axs.flat):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.imshow(images[i], cmap=plt.cm.binary)
        name = class_names[true_labels[i]]
        if predictions[i] == true_labels[i]:
            color = 'b'
        else:
            color = 'r'
            name = f"t: {name} p: {class_names[predictions[i]]}"  
        ax.set_xlabel(name, color=color)

    if plt_func is not None:
        assert callable(getattr(plt, plt_func.__name__))
        plt_func()        
