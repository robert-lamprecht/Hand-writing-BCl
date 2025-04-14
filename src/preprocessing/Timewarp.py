#Time Warping & PCA
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import pandas as pd

# Note: the filepath i use here is unique to my google drive, i couldnt figure out how to get
# the google drive mount to work on shared folders so i copied the mat files to my drive for access
dat = scipy.io.loadmat('/content/drive/MyDrive/Emory_Year_2/COMP NEURO/t5.2019.05.08_singleLetters.mat')
letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
           'greaterThan','comma','apostrophe','tilde','questionMark']

# Normalize the neural activity by blockwise z-scoring
for letter in letters:
    norm_cube = np.array(dat[f'neuralActivityCube_{letter}'], dtype=np.float32)

    t_idx = np.arange(3)
    for y in range(9):
        mn = np.zeros((3, 1, 192))
        mn[0, 0, :] = dat['meansPerBlock'][y, :]
        mn[1, 0, :] = dat['meansPerBlock'][y, :]
        mn[2, 0, :] = dat['meansPerBlock'][y, :]

        sd = np.zeros((1, 1, 192))
        sd[0, 0, :] = dat['stdAcrossAllData']

        norm_cube[t_idx, :, :] -= mn
        norm_cube[t_idx, :, :] /= sd
        t_idx += 3

    dat[f'neuralActivityCube_{letter}'] = norm_cube
# Compute trial-averaged activity for each character
all_data = np.zeros((2000, 27264))
all_spatial = np.zeros((200000, 192))
all_labels = np.zeros(2000, dtype=int)
all_avg = []
c_idx = 0
spatial_idx = 0

