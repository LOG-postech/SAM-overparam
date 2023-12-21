import jax
from jax import lax
from jax.tree_util import tree_map, tree_flatten, tree_unflatten
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp

"""
Functions for sparsifying the networks.
"""

# Various saliency scores.
# -----------------------------------------------------------------------------

def snip_score(params, batch, **kwargs):
  def loss_fn(params):
    """loss function used for training."""
    logits, _= kwargs['apply_fn'](
        {'params': params, 'batch_stats': kwargs['batch_stats']},
        batch['image'],
        train=True,
        mutable=['batch_stats'])
    loss = kwargs['loss_fn'](logits, batch['label'])
    return loss
  grad = jax.grad(loss_fn)(params)
  grad = jax.lax.pmean(grad, axis_name='batch')
  return tree_map(lambda w, g: lax.abs(w*g), params, grad)


def magnitude_score(params, batch, **kwargs):
  return tree_map(lambda w: lax.abs(w), params)


def random_score(params, batch, **kwargs):
  f_param, unravel = ravel_pytree(params)
  f_rand = jax.random.normal(kwargs['key'], f_param.shape)
  return unravel(f_rand)


def compute_score(sc_type, params, batch, **kwargs):
  return globals()[f'{sc_type}_score'](params, batch, **kwargs)  


# Mask Utilities.
# -----------------------------------------------------------------------------

def compute_mask(scores, sp, pruner):
  """Generate pruning mask based on given scores, keep highest (1-sp)-weights"""
  
  assert 0 <= sp <= 1

  # mask computing function given score and threshold
  def _mask_dict(sc, thr):
    if 'kernel' not in sc: return jnp.full(sc.shape, True)
    
    mask_dict = {'kernel': sc['kernel'] > thr}
    if 'bias' in sc:
      mask_dict['bias'] = jnp.full(sc['bias'].shape, True)

    return mask_dict

  if pruner == 'snip':
    scope = 'global'
  elif pruner == 'random':
    scope = 'local'

  # flatten scores pytree, leaf being dict containing 'kernel' instead of jnp.arrays   
  flat_tr, trdef = tree_flatten(scores, lambda tr: 'kernel' in tr)


  # sort by scores, use only kernel/weight parameters
  if scope=='global':
    flat_sc, _  = ravel_pytree([sc['kernel'] for sc in flat_tr if 'kernel' in sc])
    sort_sc = jnp.sort(flat_sc)
    thr = sort_sc[int(sp*len(sort_sc))] # compute global threshold

    _mask_dict_g = lambda sc: _mask_dict(sc, thr)
    flat_mask = [*map(_mask_dict_g, flat_tr)] # compute mask

  elif scope=='local':
    sort_scs = [(jnp.sort(sc['kernel'].ravel()) if 'kernel' in sc else None) for sc in flat_tr]
    thrs = [sc if sc==None else sc[int(sp*len(sc))] for sc in sort_scs] # compute layer thresholds

    flat_mask = [*map(_mask_dict, flat_tr, thrs)] # compute mask

  mask = tree_unflatten(trdef, flat_mask)

  return mask

@jax.jit
def apply_mask(params, mask):
  """Apply pruning mask to the parameters"""
  return tree_map(lambda p, m: p*m, params, mask)

def weight_sparsity(params):
  """Calculate the overall sparsity of the model (only for the kernels)"""
  flat_tr, _ = tree_flatten(params, lambda tr: 'kernel' in tr)
  flat_w, _ = ravel_pytree([m['kernel'] for m in flat_tr if 'kernel' in m])
  return (flat_w == 0).sum().item() / len(flat_w) // jax.device_count()