import gin
import optax
from absl import logging

@gin.configurable
def create_opt(name='adamw', learning_rate=6.25e-5, beta1=0.9, beta2=0.999,
                     eps=1.5e-4, weight_decay=0.0, centered=False):
  """Create an optimizer for training.

  Currently, only the Adam and RMSProp optimizers are supported.

  Args:
    name: str, name of the optimizer to create.
    learning_rate: float, learning rate to use in the optimizer.
    beta1: float, beta1 parameter for the optimizer.
    beta2: float, beta2 parameter for the optimizer.
    eps: float, epsilon parameter for the optimizer.
    weight_decay: float, the weight decay magnitude
    centered: bool, centered parameter for RMSProp.

  Returns:
    A flax optimizer.
  """
  if name == 'adamw':
    logging.info('Creating AdamW optimizer with settings lr=%f, beta1=%f, '
                 'beta2=%f, eps=%f, weight decay=%f', learning_rate, beta1, beta2, eps, weight_decay)
    return optax.adamw(learning_rate, b1=beta1, b2=beta2, eps=eps, weight_decay=weight_decay)
  elif name == 'adam':
    logging.info('Creating Adam optimizer with settings lr=%f, beta1=%f, '
                 'beta2=%f, eps=%f', learning_rate, beta1, beta2, eps)
    return optax.adam(learning_rate, b1=beta1, b2=beta2, eps=eps)
  elif name == 'rmsprop':
    logging.info('Creating RMSProp optimizer with settings lr=%f, beta2=%f, '
                 'eps=%f', learning_rate, beta2, eps)
    return optax.rmsprop(learning_rate, decay=beta2, eps=eps,
                         centered=centered)
  else:
    raise ValueError('Unsupported optimizer {}'.format(name))