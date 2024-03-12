import os

import numpy as np
import torch

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from scipy import stats

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

#
# Normalization and helper functions
#

def normalized_max(x):
    result = x / x.max(axis=1).reshape((x.shape[0], 1))
    return result

def normalized_l2(x, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(x, order, axis))
    l2[l2==0] = 1
    return x / np.expand_dims(l2, axis)

def normalized_z_score(x):
    scaler = StandardScaler()
    scaler.fit(x)
    return scaler.transform(x)

def small_large(a, b):
    smaller = a if a.shape[0] <= b.shape[0] else b
    larger = a if a.shape[0] > b.shape[0] else b
    return (smaller, larger)

def remove_min_max(data, iterations=1, replace=1):
    cdata = np.copy(data)
    for _ in range(iterations):
        cdata[cdata == np.max(cdata)] = replace
        cdata[cdata == np.min(cdata)] = replace
    return cdata

def slide(x, window=None):
    x = np.asarray(x)
    if window is None:
        window = x.shape[0]
    start_idx = np.arange(len(x) - window + 1)
    return np.array([x[i:(i + window)] for i in start_idx])

def slice_of_slice(data, slice1, slice2):
    return np.array([data[i, slice2[0]:slice2[1]] for i in range(slice1[0], slice1[1])])

def filter(data, threshold, value=0):
    cdata = np.copy(data)
    cdata[cdata < threshold] = value
    return cdata

def highlight(data, slice):
    mask = np.zeros_like(data)
    mask[slice[0]:slice[1]+1, :] = 1
    mask[:, slice[0]:slice[1]+1] = 1
    
    masked_matrix = data * mask
    return masked_matrix

#
# Protein/Embeddings simmilarity evaluation functions
#

def autocorr(x, window=None, normalize=False):
    if window is None:
        window = x.shape[0]
    results = [np.correlate(x[i:window + i], x[i:window + i], mode='full') for i in range((x.shape[0] - window) + 1)]
    results = np.array([r[r.size // 2:] for r in results])
    if normalize:
        results = normalized_max(results)
    return results

def crosscorr(a, b, window=None, normalize=False):
    smaller, larger = small_large(a, b)
    if window is None:
        window = smaller.shape[0]
    results = [np.correlate(smaller[i:window + i], larger, mode='full') for i in range((smaller.shape[0] - window) + 1)]
    results = np.array([r[r.size // 2:] for r in results])
    if normalize:
        results = normalized_max(results)
    return results

def project(values, mask):
    numbers_flat = values.flatten()
    bools_flat = mask.flatten()
    output_flat = np.zeros_like(bools_flat, dtype=values.dtype)
    true_indices = np.where(bools_flat)[0]
    repeat_times = -(-len(true_indices) // len(numbers_flat))
    repeated_numbers = np.tile(numbers_flat, repeat_times)[:len(true_indices)]
    output_flat[true_indices] = repeated_numbers
    return output_flat.reshape(mask.shape)

def embedding_correlation(a, b, window=None):
    smaller, larger = small_large(a, b)
    _window = smaller.shape[0] if window is None else window
    autocorr_smaller = autocorr(smaller, _window, normalize=True)
    autocorr_larger = autocorr(larger, _window, normalize=True)
    # emb_corr = [cosine_similarity(np.vstack((autocorr_slice, autocorr_larger)))[0][1:] for autocorr_slice in autocorr_smaller]
    emb_corr = cosine_similarity(autocorr_smaller, autocorr_larger)
    return np.array(emb_corr)

def protein_correlation(a, b, window=None):
    corr = [embedding_correlation(emb1, emb2, window) for (emb1, emb2) in zip(a, b)]
    return np.array(corr)

#
# Correlation/Distance functions
#

def pearsonr(a, b):
    smaller, larger = small_large(a, b)
    corr = [np.corrcoef(np.vstack((emb, larger)))[0][1:] for emb in smaller]
    return np.array(corr)
    
def spearmanr(a, b):
    smaller, larger = small_large(a, b)
    corr = [stats.spearmanr(np.vstack((emb, larger)), axis=1)[0][0][1:] for emb in smaller]
    return np.array(corr)    

def cossim3d(data):
    return np.array([cosine_similarity(e) for e in data])

def cossim_partial(a, b, window=None):
    smaller, larger = small_large(a, b)
    if window is None:
        window = smaller.shape[0]
    smaller_slides = slide(smaller, window=window)
    larger_slides = slide(larger, window=window)
    results = [cosine_similarity(np.vstack((slide, larger_slides)))[0] for slide in smaller_slides]
    return np.array(results)

def cossim(compared_set, comparison_set):
    return np.array([cosine_similarity(np.vstack((compared, comparison_set)))[0] for compared in compared_set])

#
# Statistic analysis functions
#

def variance2d(data):
    return np.var(data, axis=1)

def variance3d(data):
    return np.var(data, axis=2)

def std2d(data):
    return np.std(data, axis=1)

def std3d(data):
    return np.std(data, axis=1)

#
# Dimentionality reduction functions
#
    
def pca(data, components=3):
    pca = PCA(n_components=components)
    pca_result = pca.fit_transform(data)
    
    print('Shape: {}'.format(pca_result.shape))
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    print('Cumulative explained variation for {} principal components: {}'.format(components, np.sum(pca.explained_variance_ratio_)))
    
    return pca_result

def tsne(data, components=3):
    tsne = TSNE(n_components=components, verbose=0, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(data)
    return tsne_results

def umap(data, components=3):
    umap = UMAP(n_components=components, init='random', random_state=0)
    umap_results = umap.fit_transform(data)
    return umap_results