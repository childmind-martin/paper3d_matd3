"""Network builders and policy math for the MAPPO baseline."""

from __future__ import annotations

import math

import numpy as np
import tensorflow as tf

from paper3d_train_optimized import build_continuous_action_network


LOG_STD_MIN = -5.0
LOG_STD_MAX = 1.0
TANH_EPS = 1e-6


def build_shared_mappo_actor_network(
    input_shape,
    action_dim=7,
    hidden_units=(256, 256, 256),
    use_residual=True,
    use_fr_feature=False,
    use_pf_feature=False,
    pf_feature_dim=3,
):
    """Reuse the current actor trunk so MAPPO keeps the same observation/action style."""
    return build_continuous_action_network(
        input_shape=input_shape,
        action_dim=action_dim,
        hidden_units=hidden_units,
        use_residual=use_residual,
        use_fr_feature=use_fr_feature,
        use_pf_feature=use_pf_feature,
        pf_feature_dim=pf_feature_dim,
    )


def build_centralized_value_network(
    state_shape,
    hidden_units=(512, 512, 512),
    use_residual=True,
    use_fr_feature=False,
    use_pf_feature=False,
    pf_feature_dim=3,
    n_agents=3,
):
    """Centralized scalar value critic V(s) with the same conditioning style as the current codebase."""
    state_input = tf.keras.layers.Input(shape=state_shape, name="state_input")
    fr_input = tf.keras.layers.Input(shape=(1,), name="fr_input") if use_fr_feature else None
    pf_input = (
        tf.keras.layers.Input(shape=(pf_feature_dim * n_agents,), name="pf_input")
        if (use_pf_feature and pf_feature_dim > 0)
        else None
    )

    x = state_input
    for i, units in enumerate(hidden_units):
        residual = x
        x = tf.keras.layers.Dense(
            units,
            kernel_initializer=tf.keras.initializers.HeUniform(),
            kernel_regularizer=tf.keras.regularizers.l2(2e-5),
            name=f"value_dense_{i}",
        )(x)
        x = tf.keras.layers.LayerNormalization(name=f"value_ln_{i}")(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
        if use_residual and i > 0 and x.shape[-1] == residual.shape[-1]:
            x = tf.keras.layers.Add(name=f"value_residual_{i}")([x, residual])

    if use_fr_feature and fr_input is not None:
        fr_emb = tf.keras.layers.Dense(16, activation="relu", name="value_fr_emb")(fr_input)
        x = tf.keras.layers.Concatenate(name="value_concat_fr")([x, fr_emb])
    if pf_input is not None:
        pf_emb = tf.keras.layers.Dense(
            64,
            kernel_initializer=tf.keras.initializers.HeUniform(),
            kernel_regularizer=tf.keras.regularizers.l2(2e-5),
            name="value_pf_dense",
        )(pf_input)
        pf_emb = tf.keras.layers.LayerNormalization(name="value_pf_ln")(pf_emb)
        pf_emb = tf.keras.layers.LeakyReLU(alpha=0.01)(pf_emb)
        x = tf.keras.layers.Concatenate(name="value_concat_pf")([x, pf_emb])

    value_output = tf.keras.layers.Dense(
        1,
        kernel_initializer=tf.keras.initializers.GlorotUniform(),
        name="value_output",
    )(x)

    inputs = [state_input]
    if fr_input is not None:
        inputs.append(fr_input)
    if pf_input is not None:
        inputs.append(pf_input)
    return tf.keras.Model(inputs=inputs, outputs=value_output, name="centralized_value")


def raw_action_to_pre_tanh(action: tf.Tensor, eps: float = TANH_EPS) -> tf.Tensor:
    clipped = tf.clip_by_value(action, -1.0 + eps, 1.0 - eps)
    return 0.5 * (tf.math.log1p(clipped) - tf.math.log1p(-clipped))


def policy_mean_to_pre_tanh(mean_action: tf.Tensor, eps: float = TANH_EPS) -> tf.Tensor:
    return raw_action_to_pre_tanh(mean_action, eps=eps)


def _expand_log_std(log_std: tf.Tensor, target: tf.Tensor) -> tf.Tensor:
    clipped = tf.clip_by_value(tf.cast(log_std, target.dtype), LOG_STD_MIN, LOG_STD_MAX)
    return tf.reshape(clipped, [1, -1]) + tf.zeros_like(target)


def squashed_gaussian_sample(
    mean_action: tf.Tensor,
    log_std: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    mean_pre = policy_mean_to_pre_tanh(mean_action)
    log_std_expanded = _expand_log_std(log_std, mean_pre)
    std = tf.exp(log_std_expanded)
    noise = tf.random.normal(tf.shape(mean_pre), dtype=mean_pre.dtype)
    pre_tanh = mean_pre + noise * std
    action = tf.tanh(pre_tanh)
    log_prob_components, entropy_components = squashed_gaussian_log_prob_components_from_pre_tanh(
        mean_pre,
        log_std_expanded,
        pre_tanh,
        action,
    )
    log_prob = tf.reduce_sum(log_prob_components, axis=-1)
    entropy = tf.reduce_sum(entropy_components, axis=-1)
    return action, log_prob, entropy


def squashed_gaussian_log_prob(
    mean_action: tf.Tensor,
    log_std: tf.Tensor,
    action: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    log_prob_components, entropy_components = squashed_gaussian_log_prob_components(
        mean_action,
        log_std,
        action,
    )
    log_prob = tf.reduce_sum(log_prob_components, axis=-1)
    entropy = tf.reduce_sum(entropy_components, axis=-1)
    return log_prob, entropy


def squashed_gaussian_log_prob_from_pre_tanh(
    mean_pre: tf.Tensor,
    log_std: tf.Tensor,
    pre_tanh: tf.Tensor,
    action: tf.Tensor,
) -> tf.Tensor:
    var = tf.exp(2.0 * log_std)
    log_prob = -0.5 * (
        tf.square(pre_tanh - mean_pre) / (var + TANH_EPS)
        + 2.0 * log_std
        + math.log(2.0 * math.pi)
    )
    log_prob = tf.reduce_sum(log_prob, axis=-1)
    correction = tf.reduce_sum(tf.math.log(1.0 - tf.square(action) + TANH_EPS), axis=-1)
    return log_prob - correction


def squashed_gaussian_log_prob_components(
    mean_action: tf.Tensor,
    log_std: tf.Tensor,
    action: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    mean_pre = policy_mean_to_pre_tanh(mean_action)
    log_std_expanded = _expand_log_std(log_std, mean_pre)
    action_pre = raw_action_to_pre_tanh(action)
    return squashed_gaussian_log_prob_components_from_pre_tanh(
        mean_pre,
        log_std_expanded,
        action_pre,
        action,
    )


def squashed_gaussian_log_prob_components_from_pre_tanh(
    mean_pre: tf.Tensor,
    log_std: tf.Tensor,
    pre_tanh: tf.Tensor,
    action: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    var = tf.exp(2.0 * log_std)
    gaussian_terms = -0.5 * (
        tf.square(pre_tanh - mean_pre) / (var + TANH_EPS)
        + 2.0 * log_std
        + math.log(2.0 * math.pi)
    )
    correction_terms = tf.math.log(1.0 - tf.square(action) + TANH_EPS)
    log_prob_components = gaussian_terms - correction_terms
    entropy_components = log_std + 0.5 * math.log(2.0 * math.pi * math.e)
    return log_prob_components, entropy_components


def repeated_advantages(advantages: np.ndarray, n_agents: int) -> np.ndarray:
    tiled = np.repeat(np.asarray(advantages, dtype=np.float32)[:, None], int(n_agents), axis=1)
    return tiled.reshape(-1)
