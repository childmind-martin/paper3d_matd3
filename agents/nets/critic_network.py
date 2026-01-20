import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, BatchNormalization, Concatenate
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Reshape, Lambda
from tensorflow.keras.layers import TimeDistributed

# A very simple actor network implementations for testing
def generate_critic_network(
        num_states,
        num_actions
        ):
    """生成评论家网络
    
    Args:
        num_states: 观察空间总维度
        num_actions: 动作空间总维度
    """
    print(f"创建Critic网络: 观察维度={num_states}, 动作维度={num_actions}")
    
    # 状态输入
    state_input = tf.keras.layers.Input(shape=(num_states,), dtype=tf.float32)
    # 动作输入
    action_input = tf.keras.layers.Input(shape=(num_actions,), dtype=tf.float32)
    
    # 状态处理路径 - 使用更复杂的网络结构
    state_out = tf.keras.layers.Dense(
        384, 
        activation="elu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-5),
        kernel_initializer='glorot_normal'
    )(state_input)
    state_out = tf.keras.layers.BatchNormalization()(state_out)
    
    state_out = tf.keras.layers.Dense(
        256, 
        activation="elu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-5),
        kernel_initializer='glorot_normal'
    )(state_out)
    state_out = tf.keras.layers.BatchNormalization()(state_out)
    
    # 动作处理路径 - 添加独立的处理层
    action_out = tf.keras.layers.Dense(
        128, 
        activation="elu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-5),
        kernel_initializer='glorot_normal'
    )(action_input)
    
    # 合并状态和动作
    concat = tf.keras.layers.Concatenate()([state_out, action_out])
    
    # 共享路径 - 使用更复杂的网络结构
    out = tf.keras.layers.Dense(
        256, 
        activation="elu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-5)
    )(concat)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.Dropout(0.1)(out)  # 添加少量dropout防止过拟合
    
    out = tf.keras.layers.Dense(
        128, 
        activation="elu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-5)
    )(out)
    out = tf.keras.layers.BatchNormalization()(out)
    
    # 最后的处理层 - 更小的网络接近输出
    out = tf.keras.layers.Dense(
        64, 
        activation="elu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-5)
    )(out)
    
    # Q值输出 - 使用很小的初始化范围
    outputs = tf.keras.layers.Dense(
        1, 
        kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.003, maxval=0.003),
        bias_initializer=tf.keras.initializers.Constant(0.0)
    )(out)
    
    # 创建模型
    model = tf.keras.Model([state_input, action_input], outputs)
    
    # 编译模型以启用正则化
    model.compile(optimizer='adam', loss='mse')
    
    return model

def generate_baseline_critic_network(
        num_float_states,
        shape_img_states,
        num_actions
        ):

    initializer = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)

    # State as input
    state_float_input = Input(shape=num_float_states)
    state_out = Dense(16, activation="relu")(state_float_input)
    state_out = Dense(32, activation="relu")(state_out)

    # For a baseline, just use a dense network
    state_img_input = Input(shape=(shape_img_states))
    flatten = Flatten()(state_img_input)
    img_state_out   = Dense(32)(flatten)

    # Action as input
    action_input = Input(shape=(num_actions))
    action_out = Dense(32, activation="relu")(action_input)

    # Both are passed through seperate layer before concatenating
    concat = Concatenate()([state_out, img_state_out, action_out])

    out = Dense(64, activation="relu")(concat)
    out = Dense(64, activation="relu")(out)
    outputs = Dense(1)(out)

    # Outputs single value for given state-action
    model = tf.keras.Model([state_float_input, state_img_input,  action_input], outputs)

    return model



def generate_cnnlstm_critic_network(
        num_float_states,
        shape_img_states,
        num_actions
        ):

    initializer = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)

    # State as input
    state_float_input = Input(shape=num_float_states)
    state_out = Dense(16, activation="relu")(state_float_input)
    state_out = Dense(32, activation="relu")(state_out)

    # CNNLSTM for the image input
    # CNN Portion First
    state_img_input = Input(shape=(shape_img_states))
    conv_2d         = TimeDistributed(Convolution2D(32, (3, 3))) (state_img_input)
    max_pool        = TimeDistributed(MaxPooling2D(pool_size=(2,2))) (conv_2d)
    #conv_2d         = TimeDistributed(Convolution2D(32, (3, 3)))(max_pool)
    #max_pool        = TimeDistributed(MaxPooling2D(pool_size=(2,2))) (conv_2d)
    flatten         = TimeDistributed(Flatten())(max_pool)
    # LSTM Portion Now
    lstm            = LSTM(units=32)(flatten)
    #lstm            = LSTM(units=64)(lstm)
    img_state_out   = Dense(32)(lstm)

    # Action as input
    action_input = Input(shape=(num_actions))
    action_out = Dense(32, activation="relu")(action_input)

    # Both are passed through seperate layer before concatenating
    concat = Concatenate()([state_out, img_state_out, action_out])

    out = Dense(64, activation="relu")(concat)
    out = Dense(64, activation="relu")(out)
    outputs = Dense(1)(out)

    # Outputs single value for given state-action
    model = tf.keras.Model([state_float_input, state_img_input,  action_input], outputs)

    return model

