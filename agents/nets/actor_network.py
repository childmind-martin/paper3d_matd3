import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, BatchNormalization, Concatenate, Dropout
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Reshape, Lambda
from tensorflow.keras.layers import TimeDistributed

# A very simple actor network implementations for testing
def generate_actor_network(num_obs, num_act, max_action=1.0):
    """生成Actor网络模型
    
    Args:
        num_obs: 观察空间维度
        num_act: 动作空间维度
        max_action: 动作最大值
    """
    print(f"创建Actor网络: 输入维度={num_obs}, 输出维度={num_act}")
    
    # 使用float32数据类型，避免类型不匹配问题
    inputs = tf.keras.layers.Input(shape=(num_obs,), dtype=tf.float32, name='actor_input')
    
    # 使用更深、更宽的网络结构
    # 第一层 - 更宽的隐藏层，使用ELU激活函数
    x = tf.keras.layers.Dense(
        384, 
        activation='elu', 
        kernel_initializer='glorot_normal',
        kernel_regularizer=tf.keras.regularizers.l2(1e-6),  # 减少L2正则化强度
        name='actor_dense1'
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name='actor_bn1')(x)
    x = tf.keras.layers.Dropout(0.05)(x)  # 减少dropout比例，避免影响输出幅度
    
    # 第二层 - 中等大小隐藏层
    x = tf.keras.layers.Dense(
        256, 
        activation='elu', 
        kernel_initializer='glorot_normal',
        kernel_regularizer=tf.keras.regularizers.l2(1e-6),  # 减少L2正则化强度
        name='actor_dense2'
    )(x)
    x = tf.keras.layers.BatchNormalization(name='actor_bn2')(x)
    
    # 第三层 - 较小隐藏层，使用更小的初始化范围
    x = tf.keras.layers.Dense(
        128, 
        activation='elu', 
        kernel_initializer='glorot_normal',
        kernel_regularizer=tf.keras.regularizers.l2(1e-6),  # 减少L2正则化强度
        name='actor_dense3'
    )(x)
    x = tf.keras.layers.BatchNormalization(name='actor_bn3')(x)
    
    # 输出层使用tanh激活，限制在[-1,1]范围，使用更大的初始化范围
    outputs = tf.keras.layers.Dense(
        num_act, 
        activation='tanh', 
        kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.01, maxval=0.01),  # 增大初始化范围
        bias_initializer=tf.keras.initializers.Constant(0.1),  # 初始化偏置为小正值，促进网络输出
        name='actor_output'
    )(x)
    
    # 缩放到指定范围，增大输出幅度
    outputs = outputs * max_action * 1.2  # 增大最大动作范围
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # 编译模型以启用正则化，使用更小的学习率
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse')
    
    return model

def generate_baseline_actor_network(
        num_float_states,
        shape_img_states,
        num_actions,
        actions_max
        ):

    # 使用更大的初始化范围，使梯度更容易传播
    initializer = tf.random_uniform_initializer(minval=-0.01, maxval=0.01)

    # Float state network - A simple dense network
    float_input     = Input(shape=num_float_states)
    dense           = Dense(64)(float_input)
    #dense           = Dense(64)(dense)
    float_state_out = Dense(32)(dense)

    # We treat the inputs like a collection of floats
    img_input      = Input(shape=shape_img_states)
    flatten      = Flatten()(img_input)
    img_state_out  = Dense(32)(flatten)

    concat = Concatenate()([float_state_out, img_state_out])

    out = Dense(64, activation="relu")(concat)
    out = Dense(64, activation="relu")(out)
    outputs = Dense(num_actions, activation="tanh", kernel_initializer=initializer)(out)
    # Assumes the actions are equal in each direction
    outputs = outputs * actions_max * 1.2  # 增大动作范围
    model = tf.keras.Model([float_input, img_input], outputs)

    return model


# A very simple actor network implementations for testing
def generate_cnnlstm_actor_network(
        num_float_states,
        shape_img_states,
        num_actions,
        actions_max
        ):

    # 使用更大的初始化范围
    initializer = tf.random_uniform_initializer(minval=-0.01, maxval=0.01)

    # Float state network - A simple dense network
    float_input     = Input(shape=num_float_states)
    dense           = Dense(64)(float_input)
    #dense           = Dense(64)(dense)
    float_state_out = Dense(32)(dense)

    # CNNLSTM for the image input
    # CNN Portion First
    img_input      = Input(shape=shape_img_states)
    conv_2d        = TimeDistributed(Convolution2D(3, (3, 3))) (img_input)
    max_pool       = TimeDistributed(MaxPooling2D(pool_size=(2,2))) (conv_2d)
    #conv_2d        = TimeDistributed(Convolution2D(3, (3, 3))) (max_pool)
    #max_pool       = TimeDistributed(MaxPooling2D(pool_size=(2,2))) (conv_2d)
    flatten        = TimeDistributed(Flatten())(max_pool)
    # LSTM Portion Now
    lstm           = LSTM(units=32)(flatten)
    #lstm           = LSTM(units=64)(lstm)
    img_state_out  = Dense(32)(lstm)

    concat = Concatenate()([float_state_out, img_state_out])

    out = Dense(64, activation="relu")(concat)
    out = Dense(64, activation="relu")(out)
    outputs = Dense(num_actions, activation="tanh", kernel_initializer=initializer)(out)
    # Assumes the actions are equal in each direction
    outputs = outputs * actions_max * 1.2  # 增大动作范围
    model = tf.keras.Model([float_input, img_input], outputs)

    return model

