import numpy as np
import torch

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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

def autocorr(x, window=None, normalize=False):
    if window is None:
        window = x.shape[0]
    results = [np.correlate(x[i:window + i], x[i:window + i], mode='full') for i in range((x.shape[0] - window) + 1)]
    results = np.array([r[r.size // 2:] for r in results])
    if normalize:
        results = normalized_max(results)
    return results

def corr(a, b, normalize=False):
    result = np.correlate(a, b, mode='full')
    result = result[result.size // 2:]
    if normalize:
        result = result / float(result.max())
    return result
    
def remove_min_max(data, iterations=1):
    cdata = np.copy(data)
    for _ in range(iterations):
        cdata[cdata == np.max(cdata)] = 0
        cdata[cdata == np.min(cdata)] = 0
    return cdata

def embedding_correlation(a, b, window=None):
    smaller = a if a.shape[0] <= b.shape[0] else b
    larger = a if a.shape[0] > b.shape[0] else b
    _window = smaller.shape[0] if window is None else window
    autocorr_smaller = autocorr(smaller, _window, normalize=True)
    autocorr_larger = autocorr(larger, _window, normalize=True)
    emb_corr = [np.corrcoef(np.vstack((autocorr_slice, autocorr_larger)))[0][1:] for autocorr_slice in autocorr_smaller]
    return np.array(emb_corr)

def protein_correlation(a, b, window=None):
    corr = [embedding_correlation(emb1, emb2, window) for (emb1, emb2) in zip(a, b)]
    return np.array(corr)

# a and b are 2d arrays
def multidim_corr(a, b):
    min_dim = min(a.shape[1], b.shape[1])
    max_dim = max(a.shape[1], b.shape[1])
    corr = np.zeros(shape=((max_dim - min_dim) + 1, a.shape[0]))
    for i in range((max_dim - min_dim) + 1):
        corr[i] = np.array([np.corrcoef(a[j, :min_dim], b[j, i:min_dim + i])[0][1] for j in range(a.shape[0])])
    return corr

def magnitude(x):
    return np.sqrt(np.tensordot(x, x, axes=-1))

def save(name="default.txt", data=[], mode="w"):
    with open(name, mode=mode) as f:
        np.savetxt(f, data, fmt="%-.5f")
    
def layer_norm(data):
    layer_norm = torch.nn.LayerNorm(data.shape[0], dtype=torch.float64)
    return layer_norm(torch.from_numpy(data.reshape(1, 1, -1))).detach().numpy().reshape(-1)

def cossim3d(data):
    return np.array([cosine_similarity(e) for e in data])
    
def cossim(compared_set, comparison_set):
    return np.array([cosine_similarity(np.vstack((compared, comparison_set)))[0] for compared in compared_set])

def slice_of_slice(data, slice1, slice2):
    return np.array([data[i, slice2[0]:slice2[1]] for i in range(slice1[0], slice1[1])])

def pca(data, components=3):
    pca = PCA(n_components=components)
    pca_result = pca.fit_transform(data)
    
    print('Shape: {}'.format(pca_result.shape))
    
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    
    print('Cumulative explained variation for {} principal components: {}'.format(components, np.sum(pca.explained_variance_ratio_)))
    
    return pca_result

def tsne(data, components=3):
    tsne = TSNE(n_components=components, verbose=0, perplexity=40, n_iter=300)
    tsne_pca_results = tsne.fit_transform(data)
    return tsne_pca_results