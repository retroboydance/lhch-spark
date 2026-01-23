import os
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap, grad, random
from jax.tree_util import tree_map, tree_leaves, tree_reduce

def random_split_like_tree(rng_key, target=None, treedef=None):
    if treedef is None: treedef = jax.tree_util.tree_structure(target)
    keys = random.split(rng_key, treedef.num_leaves)
    return jax.tree_util.tree_unflatten(treedef, keys)

def norm(x):
    rms = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    return x / jnp.sqrt(rms + 1e-6)

def kvq_project(x, target):
    return tree_map(lambda w: jnp.dot(w, x.T), target)

def apply_rotary_embeddings(x, cos, sin):
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return jnp.concatenate([y1, y2], axis=-1)

def precompute_rotary_embeddings(seq_len, head_dim, base=10000):
    freqs = 1.0 / (base ** (jnp.arange(0, head_dim, 2) / head_dim))
    t = jnp.arange(seq_len)
    freqs = jnp.outer(t, freqs)
    return dict(cos = jnp.cos(freqs), sin = jnp.sin(freqs))

def att_tree(dims: int, scale=1e-2):
    return dict(
        key = scale * jnp.ones((dims, dims)),
        query = scale * jnp.ones((dims, dims)),
        value = scale * jnp.ones((dims, dims)),
    )

def init_dense_nobias_params(key, meta_params):
    keys_tree = random_split_like_tree(key, meta_params)
    return tree_map(lambda l, k: l * random.normal(k, l.shape), meta_params, keys_tree)


if __name__ == '__main__':
    test_key = random.key(420)
    
    embed_dim = 16
    seq_length = 32
    vocab_size = 10000
    num_heads = 2
    num_kv_heads = num_heads

    test_input = np.random.normal(size=(seq_length, embed_dim))
    param_tree = att_tree(embed_dim)
    params = init_dense_nobias_params(test_key, param_tree)
    cos_sin = precompute_rotary_embeddings(seq_length, embed_dim // num_heads)

    output = kvq_project(test_input, params)

    q = apply_rotary_embeddings(output['query'], cos_sin['cos'], cos_sin['sin'])
    k = apply_rotary_embeddings(output['key'], cos_sin['cos'], cos_sin['sin'])

    q = norm(q)
    k = norm(k)