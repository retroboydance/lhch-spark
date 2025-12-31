import os
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap, grad, random
from jax.tree_util import tree_map, tree_leaves, tree_reduce
import time
from jax.scipy.special import logsumexp
from torch.utils.data import DataLoader, default_collate
from torchvision.datasets import MNIST

def numpy_collate(batch):
  return tree_map(np.asarray, default_collate(batch))

def flatten_and_cast(pic):
  return np.ravel(np.array(pic, dtype=jnp.float32))

def random_split_like_tree(rng_key, target=None, treedef=None):
    if treedef is None: treedef = jax.tree_util.tree_structure(target)
    keys = jax.random.split(rng_key, treedef.num_leaves)
    return jax.tree_util.tree_unflatten(treedef, keys)

def create_params_tree(in_dim, out_dim, scale=1e-2):
    return dict(weight = scale * jnp.ones((in_dim, out_dim)), bias = scale * jnp.ones((out_dim,)))

def init_layer_params(key, meta_params):
    ##assert len(meta_params) == 2
    keys_tree = random_split_like_tree(key, meta_params)
    return tree_map(lambda l, k: l * random.normal(k, l.shape), meta_params, keys_tree)

def init_network_layout(layer_config: list):
    in_dim = layer_config[:-1]
    out_dim = layer_config[1:]
    layer_names = ['layer' + str(idx) for idx in range(len(layer_config) - 1)]
    return dict(zip(layer_names, zip(in_dim, out_dim)))

def jax_relu(raw_output: jax.Array):
    return jnp.maximum(0, raw_output)

def forward(params, x):
    *hidden, final = params
    for layer in hidden:
        x = jnp.dot(x, params[layer]['weight']) + params[layer]['bias']
        x = jax_relu(x)
    logits = jnp.dot(x, params[final]['weight']) + params[final]['bias']
    return logits - logsumexp(logits, axis=-1, keepdims=True)

batched_forward = vmap(forward, in_axes=(None, 0))

def one_hot(x, k, dtype=jnp.float32):
  return jnp.array(x[:, None] == jnp.arange(k), dtype)

def accuracy(params, images, targets):
  target_class = jnp.argmax(targets, axis=1)
  predicted_class = jnp.argmax(batched_forward(params, images), axis=1)
  return jnp.mean(predicted_class == target_class)

def loss(params, images, targets):
  preds = batched_forward(params, images)
  return -jnp.mean(preds * targets)

@jax.jit
def update(params, x, y, lr=3e-3):
    grads = jax.grad(loss)(params, x, y)
    return tree_map(lambda p, g: p - lr * g, params, grads)

if __name__ == '__main__':
    test_key = random.key(420)

    image_size = (28, 28)

    step_size = 0.01
    num_epochs = 10
    batch_size = 16
    n_targets = 10

    layerz = [image_size[0] * image_size[0], 128, 128, n_targets]

    param_tree = init_network_layout(layerz)

    for key, value in param_tree.items():
        param_tree[key] = create_params_tree(value[0], value[1])

    params = init_layer_params(test_key, param_tree)

    mnist_dataset = MNIST('/tmp/mnist/', download=True, transform=flatten_and_cast)
    training_generator = DataLoader(mnist_dataset, batch_size=batch_size, collate_fn=numpy_collate)

    train_images = np.array(mnist_dataset.train_data).reshape(len(mnist_dataset.train_data), -1)
    train_labels = one_hot(np.array(mnist_dataset.train_labels), n_targets)

    mnist_dataset_test = MNIST('/tmp/mnist/', download=True, train=False)
    test_images = jnp.array(mnist_dataset_test.test_data.numpy().reshape(len(mnist_dataset_test.test_data), -1), dtype=jnp.float32)
    test_labels = one_hot(np.array(mnist_dataset_test.test_labels), n_targets)

    for epoch in range(num_epochs):
        start_time = time.time()
        for x, y in training_generator:
            y = one_hot(y, n_targets)
            params = update(params, x, y)
        epoch_time = time.time() - start_time

        train_acc = accuracy(params, train_images, train_labels)
        test_acc = accuracy(params, test_images, test_labels)
        print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
        print("Training set accuracy {}".format(train_acc))
        print("Test set accuracy {}".format(test_acc))