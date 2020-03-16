import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def plot_summaries(history, plot_name1):
    # Plot loss and Accuracy 
    fig, ax = plt.subplots(1, 2, figsize = (16, 8))
    df = pd.DataFrame(history)

    ax[0].plot(df[['root_loss','vowel_loss','consonant_loss', 'val_root_loss','val_vowel_loss','val_consonant_loss']])
    ax[0].set_ylim(0, 2)
    ax[0].set_title('Loss')
    ax[0].legend(['train_root_loss','train_vowel_loss','train_conso_loss', 'val_root_loss','val_vowel_loss','val_conso_loss'],
                loc='upper right')
    ax[0].grid()
    ax[1].plot(df[['root_accuracy','vowel_accuracy','consonant_accuracy', 'val_root_accuracy','val_vowel_accuracy','val_consonant_accuracy']])
    ax[1].set_ylim(0.5, 1)
    ax[1].set_title('Accuracy')
    ax[1].legend(['train_root_acc','train_vowel_acc','train_conso_acc', 'val_root_acc','val_vowel_acc','val_conso_acc'],
                loc='lower right')
    ax[1].grid()
    fig.savefig(plot_name1)
