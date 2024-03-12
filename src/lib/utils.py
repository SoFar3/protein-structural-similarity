import numpy as np
import glob
import os

def shape_to_str(shape, delim='_'):
    s = ""
    for d in shape:
        s = s + str(d) + delim
    s = s[:-1]
    return s

def str_to_shape(s, delim='_'):
    dims = [int(p) for p in s.split(delim)]
    return tuple(dims)

def embeddings_path(seq_name, shape, type="seq"):
    protein_dir = os.environ.get('PROTEIN_DB_PATH', default=os.path.abspath(os.path.join(os.pardir, "protein_db")))
    emb_dir = f'esm/embeddings/{seq_name}/'
    return os.path.join(protein_dir, emb_dir, f'{type}_{shape_to_str(shape)}.txt')

def read_embeddings(seq_name, type="seq"):
    emb_files = find_embeddings(seq_name, type)
    if len(emb_files) == 0:
        raise Exception(f'No embedding file found for sequence {seq_name}')
    return np.fromfile(emb_files[0], sep=' ').reshape(str_to_shape(os.path.splitext(os.path.basename(emb_files[0]))[0].replace(type + '_', '')))

def find_embeddings(seq_name, type="seq"):
    protein_dir = os.environ.get('PROTEIN_DB_PATH', default=os.path.abspath(os.path.join(os.getcwd(), "protein_db")))
    emb_dir = f'esm/embeddings/{seq_name}/'
    protein_path = f'{os.path.join(protein_dir, emb_dir, f"{type}*")}'
    print(f'Looking for embeddings in path: {protein_path}')
    return glob.glob(protein_path)

def embeddings_exist(seq_name, type="seq"):
    return len(find_embeddings(seq_name, type)) != 0

def save(name="default.txt", data=[], mode="w"):
    os.makedirs(os.path.dirname(name), exist_ok=True)
    with open(name, mode=mode) as f:
        np.savetxt(f, data, fmt="%-.5f")