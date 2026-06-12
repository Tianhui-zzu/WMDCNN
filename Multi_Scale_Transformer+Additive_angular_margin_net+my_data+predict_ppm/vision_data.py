import os.path

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from mpl_toolkits.mplot3d import Axes3D
path = 't3'
target_ppm = 'extract_y.csv'
predict_ppm = 'final_predict_results.csv'

def vision_data02():
    X = np.arange(0, 1.1, 0.1)
    Y = X
    print(X)
    plt.plot(X, Y, color='red')
    X1 = pd.read_csv(os.path.join(target_ppm))
    X2 = pd.read_csv(os.path.join(predict_ppm))
    plt.scatter(X1, X2, marker='o', alpha=.8)
    plt.show()


