from tf_env import tf
import numpy as np
from tensorflow.python.ops import array_ops

def _conv_weight_variable(shape, name, trainable=True):
    initializer = tf.orthogonal_initializer()
    return tf.get_variable(name, shape=shape, initializer=initializer, trainable=trainable)

def _fc_weight_variable(shape, name, trainable=True):
    initializer = tf.orthogonal_initializer()
    return tf.get_variable(name, shape=shape, initializer=initializer, trainable=trainable)

def _bias_variable(shape, name, trainable=True):
    initializer = tf.constant_initializer(0.0)
    return tf.get_variable(name, shape=shape, initializer=initializer, trainable=trainable)

def maxpooling(input_list, unit_num, unit_size, name):
    concat_result = tf.concat(input_list, axis=1, name="%s_concat_result" % name)
    reshape_concat_result = tf.reshape(concat_result, shape=[-1, unit_num, unit_size, 1], name="reshape_%s" % name)
    pool_result = tf.nn.max_pool(reshape_concat_result, [1, unit_num, 1, 1], [1, 1, 1, 1], padding='VALID',
                                 name="pool_%s" % name)
    output_dim = int(np.prod(pool_result.get_shape()[1:]))
    reshape_pool_result = tf.reshape(pool_result, shape=[-1, output_dim], name="reshape_pool_%s" % name)
    return reshape_pool_result

def topk_softmax(logits, top_k):
    top, top_idx = tf.nn.top_k(logits, top_k, sorted=False)
    top_sm = tf.nn.softmax(top)

    input_shape = tf.shape(logits)
    input_row_idx = tf.tile(tf.range(input_shape[0])[:, tf.newaxis], (1, top_k))
    scatter_idx = tf.stack([input_row_idx, top_idx], axis=-1)
    result = tf.scatter_nd(scatter_idx, top_sm, input_shape)
    return result

def sample_from_prob(prob):
    dist = tf.distributions.Categorical(probs=prob)
    one_hot_ourput = tf.one_hot(dist.sample(), prob.get_shape()[-1])
    return one_hot_ourput

def focal_loss(target_tensor, prediction_tensor, weights=None, alpha=0.25, gamma=2, is_reduce=True):
    r"""Compute focal loss for predictions.
                Multi-labels Focal loss formula:
                    FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                         ,which alpha = 0.25, gamma = 2, p = pred_prob, z = target_tensor.
            Args:
             prediction_tensor: A float tensor of shape [batch_size, num_anchors,
                num_classes] representing the predicted prob for each class
             target_tensor: A float tensor of shape [batch_size, num_anchors,
                num_classes] representing one-hot encoded classification targets
             weights: A float tensor of shape [batch_size, num_anchors]
             alpha: A scalar tensor for focal loss alpha hyper-parameter
             gamma: A scalar tensor for focal loss gamma hyper-parameter
            Returns:
                loss: A (scalar) tensor representing the value of the loss function
            """
    zeros = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)

    # For poitive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
    pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - prediction_tensor, zeros)

    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, prediction_tensor)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(prediction_tensor, 1e-8, 1.0)) - (
                1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - prediction_tensor, 1e-8, 1.0))
    if is_reduce:
        return tf.reduce_mean(per_entry_cross_ent)
    else:
        return per_entry_cross_ent