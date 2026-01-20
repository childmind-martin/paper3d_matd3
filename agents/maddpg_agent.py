class DDPGAgent:
    def __init__(self, env, agent_id, buffer_size=100000, batch_size=1024, gamma=0.99, tau=0.01, actor_lr=0.00025, critic_lr=0.0005, is_3d=False, noise_std_dev=0.2, build_actor_fn=None, build_critic_fn=None):
        # ... existing code ...
        
        # 更新学习率参数
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        
        # 保存自定义网络构建函数
        self.build_actor_fn = build_actor_fn
        self.build_critic_fn = build_critic_fn
        
        # ... existing code ...
        
        # 模型创建
        self._create_networks(is_3d=is_3d)
        
    def _create_networks(self, is_3d=False):
        """创建DDPG所需的深度神经网络
        
        Args:
            is_3d: 是否为3D环境，将影响动作空间维度
        """
        # 设置动作维度
        action_dim = 3 if is_3d else 2
        self.action_dim = action_dim
        self.is_3d = is_3d
        
        # 获取观察空间和动作空间维度
        obs_dim = self.obs_dim
        
        # 性能优化：创建更高效的网络架构
        # 采用较窄但更深的网络，添加批归一化以加速训练
        
        # 创建Actor网络 - 输入: 观察 | 输出: 动作
        if self.build_actor_fn is not None:
            # 使用自定义网络构建函数
            print("使用自定义Actor网络构建函数")
            self.actor = self.build_actor_fn(input_shape=(obs_dim,), action_dim=action_dim)
        else:
            # 使用默认网络构建函数
            self.actor = self._build_actor(obs_dim, action_dim)
            
        # 判断actor是否是多输出模型
        self.is_multi_output = hasattr(self.actor, 'outputs') and len(self.actor.outputs) > 1
        if self.is_multi_output:
            print(f"检测到多输出Actor网络，输出数量: {len(self.actor.outputs)}")
            # 默认第一个输出为动作输出
            self.actor_action_output_index = 0
            # 第二个输出为力参数输出
            self.actor_force_params_index = 1
        
        # 创建目标Actor网络
        if self.build_actor_fn is not None:
            # 重新使用自定义构建函数以确保结构一致
            self.target_actor = self.build_actor_fn(input_shape=(obs_dim,), action_dim=action_dim)
        else:
            # 使用默认网络构建函数
            self.target_actor = self._build_actor(obs_dim, action_dim)
        
        # 复制权重到目标网络
        self.target_actor.set_weights(self.actor.get_weights())
        
        # 创建Critic网络 - 输入: 观察+所有智能体动作 | 输出: Q值
        if self.build_critic_fn is not None:
            # 使用自定义网络构建函数
            print("使用自定义Critic网络构建函数")
            self.critic = self.build_critic_fn(state_shape=(obs_dim,), action_dim=action_dim * self.n_agents)
        else:
            # 使用默认网络构建函数
            self.critic = self._build_critic(obs_dim, action_dim * self.n_agents)
            
        # 创建目标Critic网络
        if self.build_critic_fn is not None:
            # 重新使用自定义构建函数以确保结构一致
            self.target_critic = self.build_critic_fn(state_shape=(obs_dim,), action_dim=action_dim * self.n_agents)
        else:
            # 使用默认网络构建函数
            self.target_critic = self._build_critic(obs_dim, action_dim * self.n_agents)
        
        # 复制权重到目标网络
        self.target_critic.set_weights(self.critic.get_weights())
        
        # 优化器
        self._actor_opt = tf.keras.optimizers.Adam(learning_rate=self.actor_lr)
        self._critic_opt = tf.keras.optimizers.Adam(learning_rate=self.critic_lr)
        
        # 动作噪声
        self._noise = OUNoise(mean=np.zeros(action_dim), std_dev=float(noise_std_dev) * np.ones(action_dim), decay=0.9995)
    
    def _build_actor(self, obs_dim, action_dim):
        """构建优化后的Actor网络
        
        Args:
            obs_dim: 观察空间的维度
            action_dim: 动作空间的维度
        Returns:
            actor_model: Keras模型
        """
        inputs = tf.keras.layers.Input(shape=(obs_dim,))
        
        # 第一层 - 较大的隐藏层捕捉输入特征
        x = tf.keras.layers.Dense(256, activation=None)(inputs) 
        x = tf.keras.layers.BatchNormalization()(x)  # 添加批归一化
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(0.1)(x)  # 轻微的dropout防止过拟合
        
        # 第二层 - 抽取更高级特征
        x = tf.keras.layers.Dense(128, activation=None)(x)
        x = tf.keras.layers.BatchNormalization()(x)  # 添加批归一化
        x = tf.keras.layers.Activation('relu')(x)
        
        # 第三层 - 进一步抽象
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        
        # 输出层 - tanh激活函数约束动作到[-1,1]范围
        outputs = tf.keras.layers.Dense(action_dim, activation='tanh')(x)
        
        # 创建模型
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        return model
    
    def _build_critic(self, obs_dim, action_dim):
        """构建优化后的Critic网络
        
        Args:
            obs_dim: 观察空间的维度
            action_dim: 所有智能体的动作空间总维度
        Returns:
            critic_model: Keras模型
        """
        # 状态输入
        state_input = tf.keras.layers.Input(shape=(obs_dim,))
        state_out = tf.keras.layers.Dense(128, activation='relu')(state_input)
        
        # 动作输入
        action_input = tf.keras.layers.Input(shape=(action_dim,))
        
        # 合并状态和动作
        concat = tf.keras.layers.Concatenate()([state_out, action_input])
        
        # 添加网络层
        x = tf.keras.layers.Dense(256, activation=None)(concat)
        x = tf.keras.layers.BatchNormalization()(x)  # 添加批归一化
        x = tf.keras.layers.Activation('relu')(x)
        
        x = tf.keras.layers.Dense(128, activation=None)(x)
        x = tf.keras.layers.BatchNormalization()(x)  # 添加批归一化
        x = tf.keras.layers.Activation('relu')(x)
        
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        
        # 输出层 - 没有激活函数，因为Q值可以是任何实数
        outputs = tf.keras.layers.Dense(1)(x)
        
        # 创建模型
        model = tf.keras.Model(inputs=[state_input, action_input], outputs=outputs)
        
        return model
    
    def policy(self, obs, add_noise=True, training=False):
        """根据观察选择动作

        Args:
            obs: 观察值
            add_noise: 是否添加噪声（用于探索）
            training: 是否处于训练模式

        Returns:
            action: 选择的动作
        """
        # 转换观察值为tensor并确保维度正确
        if isinstance(obs, np.ndarray) and len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)  # 添加批次维度
            
        obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        
        # 获取Actor网络输出
        outputs = self.actor(obs, training=training)
        
        # 处理多输出情况
        if self.is_multi_output:
            # 获取动作输出（第一个输出）
            action = outputs[self.actor_action_output_index]
        else:
            # 单输出模型
            action = outputs
            
        # 从tensor转换为numpy数组
        if isinstance(action, tf.Tensor):
            action = action.numpy()
            
        # 添加噪声（如果需要）
        if add_noise:
            action = action + self._noise.sample()
            # 确保动作在[-1, 1]范围内
            action = np.clip(action, -1.0, 1.0)
            
        # 去掉批次维度（如果只有一个样本）
        if action.shape[0] == 1:
            action = action[0]
            
        return action
        
    def get_force_params(self, obs):
        """获取力参数输出

        Args:
            obs: 观察值

        Returns:
            force_params: 力参数值，如果不是多输出模型则返回None
        """
        if not self.is_multi_output:
            return None
            
        # 转换观察值为tensor并确保维度正确
        if isinstance(obs, np.ndarray) and len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)  # 添加批次维度
            
        obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        
        # 获取Actor网络输出
        outputs = self.actor(obs, training=False)
        
        # 获取力参数输出（第二个输出）
        force_params = outputs[self.actor_force_params_index]
            
        # 从tensor转换为numpy数组
        if isinstance(force_params, tf.Tensor):
            force_params = force_params.numpy()
            
        # 去掉批次维度（如果只有一个样本）
        if force_params.shape[0] == 1:
            force_params = force_params[0]
            
        return force_params
    
    def target_policy(self, obs, add_noise=False, training=False):
        """使用目标Actor网络根据观察选择动作

        Args:
            obs: 观察值
            add_noise: 是否添加噪声（用于探索）
            training: 是否处于训练模式

        Returns:
            action: 选择的动作
        """
        # 转换观察值为tensor并确保维度正确
        if isinstance(obs, np.ndarray) and len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)  # 添加批次维度
            
        obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        
        # 使用目标Actor网络获取输出
        outputs = self.target_actor(obs, training=training)
        
        # 处理多输出情况
        if self.is_multi_output:
            # 获取动作输出（第一个输出）
            action = outputs[self.actor_action_output_index]
        else:
            # 单输出模型
            action = outputs
            
        # 从tensor转换为numpy数组
        if isinstance(action, tf.Tensor):
            action = action.numpy()
            
        # 添加噪声（如果需要）
        if add_noise:
            action = action + self._noise.sample()
            # 确保动作在[-1, 1]范围内
            action = np.clip(action, -1.0, 1.0)
            
        # 去掉批次维度（如果只有一个样本）
        if action.shape[0] == 1:
            action = action[0]
            
        return action
    
    def update(self, obs, action, reward, next_obs, done, next_actions):
        """使用更高效的批处理策略更新策略网络，支持多输出网络
        
        Args:
            obs: 当前观察
            action: 当前动作
            reward: 奖励
            next_obs: 下一个观察
            done: 是否完成
            next_actions: 所有智能体的下一步动作
        
        Returns:
            critic_loss: Critic网络的损失
            actor_loss: Actor网络的损失
        """
        print(f"_update_critic - 输入形状: states={obs.shape if hasattr(obs, 'shape') else 'unknown'}, actions={action.shape if hasattr(action, 'shape') else 'unknown'}, next_states={next_obs.shape if hasattr(next_obs, 'shape') else 'unknown'}, next_actions={[a.shape if hasattr(a, 'shape') else 'unknown' for a in next_actions]}")
        
        # 确保obs是正确的形状
        if isinstance(obs, np.ndarray) and len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)
            
        # 确保action是正确的形状
        if isinstance(action, np.ndarray) and len(action.shape) == 1:
            action = np.expand_dims(action, axis=0)
            
        # 确保next_obs是正确的形状
        if isinstance(next_obs, np.ndarray) and len(next_obs.shape) == 1:
            next_obs = np.expand_dims(next_obs, axis=0)
            
        # 将所有输入转换为tensors
        obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        action = tf.convert_to_tensor(action, dtype=tf.float32)
        reward = tf.convert_to_tensor([reward], dtype=tf.float32)
        next_obs = tf.convert_to_tensor(next_obs, dtype=tf.float32)
        done = tf.convert_to_tensor([float(done)], dtype=tf.float32)
        
        # 处理next_actions，确保它们都是正确的形状
        processed_next_actions = []
        for i, next_action in enumerate(next_actions):
            # 确保是numpy数组
            if isinstance(next_action, tf.Tensor):
                next_action = next_action.numpy()
                
            # 确保形状正确
            if len(next_action.shape) == 1:  # 如果是(action_dim,)
                next_action = np.expand_dims(next_action, axis=0)  # 变为(1, action_dim)
                
            # 转换为tensor
            next_action_tensor = tf.convert_to_tensor(next_action, dtype=tf.float32)
            processed_next_actions.append(next_action_tensor)
        
        # 检查每个张量的形状是否相同，不同则调整
        action_shapes = [a.shape for a in processed_next_actions]
        if len(set(str(s) for s in action_shapes)) > 1:
            # 有不同形状的张量，需要统一
            target_shape = (1, self.action_dim)  # 目标形状：(batch_size, action_dim)
            for i, a in enumerate(processed_next_actions):
                if a.shape != target_shape:
                    processed_next_actions[i] = tf.reshape(a, target_shape)
        
        # 将下一步动作连接为一个张量
        try:
            all_next_actions = tf.concat(processed_next_actions, axis=1)
        except Exception as e:
            print(f"连接next_actions失败: {e}, 形状: {[a.shape for a in processed_next_actions]}")
            # 尝试替代方案
            all_next_actions = tf.concat([tf.reshape(a, (1, -1)) for a in processed_next_actions], axis=1)
        
        # 更新Critic
        with tf.GradientTape() as tape:
            # 获取目标Q值
            target_q = self.target_critic([next_obs, all_next_actions])
            target_q = reward + self.gamma * target_q * (1 - done)
            
            # 获取当前Q值
            all_actions = []
            for i, a in enumerate(processed_next_actions):
                if i == self.agent_id:
                    all_actions.append(action)
                else:
                    # 确保动作维度匹配
                    if a.shape != action.shape:
                        a = tf.reshape(a, action.shape)
                    all_actions.append(a)
            
            try:
                all_actions = tf.concat(all_actions, axis=1)
            except Exception as e:
                print(f"连接all_actions失败: {e}, 形状: {[a.shape for a in all_actions]}")
                # 尝试替代方案
                all_actions = tf.concat([tf.reshape(a, (1, -1)) for a in all_actions], axis=1)
                
            current_q = self.critic([obs, all_actions])
            
            # 计算TD误差和损失
            critic_loss = tf.reduce_mean(tf.square(target_q - current_q))
            
            # 性能优化：添加L2正则化，防止过拟合
            for var in self.critic.trainable_variables:
                critic_loss += 0.001 * tf.reduce_sum(tf.square(var))
            
        # 计算梯度并更新Critic网络权重
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self._critic_opt.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        
        # 更新Actor
        with tf.GradientTape() as tape:
            # 使用当前策略选择动作
            actor_outputs = self.actor(obs)
            
            # 处理多输出情况
            if self.is_multi_output:
                current_actions = actor_outputs[self.actor_action_output_index]
            else:
                current_actions = actor_outputs
            
            # 组合所有动作
            all_current_actions = []
            
            # 增加额外的安全检查，防止迭代标量张量
            if isinstance(processed_next_actions, list) and len(processed_next_actions) > 0:
                for i, a in enumerate(processed_next_actions):
                    if i == self.agent_id:
                        all_current_actions.append(current_actions)
                    else:
                        # 确保动作维度匹配
                        if hasattr(a, 'shape') and hasattr(current_actions, 'shape') and a.shape != current_actions.shape:
                            # 尝试重新形状
                            if hasattr(a, 'shape') and len(a.shape) > 0:
                                a = tf.reshape(a, current_actions.shape)
                            else:
                                # 如果a是标量，将其扩展为与current_actions相同形状的常量
                                a = tf.ones_like(current_actions) * tf.cast(a, dtype=tf.float32)
                        all_current_actions.append(a)
            else:
                # 如果processed_next_actions不是列表或为空，直接使用current_actions
                all_current_actions = [current_actions]
            
            # 安全地连接所有动作
            try:
                # 确保所有元素都是张量并且具有相同的形状
                valid_actions = []
                for a in all_current_actions:
                    if isinstance(a, tf.Tensor):
                        # 检查形状是否匹配
                        if len(a.shape) < 2:
                            # 扩展维度以匹配批次维度
                            a = tf.expand_dims(a, axis=0)
                        if a.shape[0] != obs.shape[0]:
                            # 调整批次大小
                            a = tf.broadcast_to(a, [obs.shape[0], a.shape[-1]])
                        valid_actions.append(a)
                    else:
                        # 转换为张量
                        a_tensor = tf.convert_to_tensor(a, dtype=tf.float32)
                        if len(a_tensor.shape) < 2:
                            a_tensor = tf.expand_dims(a_tensor, axis=0)
                        if a_tensor.shape[0] != obs.shape[0]:
                            a_tensor = tf.broadcast_to(a_tensor, [obs.shape[0], a_tensor.shape[-1]])
                        valid_actions.append(a_tensor)
                
                # 连接有效的动作
                if len(valid_actions) > 1:
                    all_current_actions_tensor = tf.concat(valid_actions, axis=1)
                elif len(valid_actions) == 1:
                    all_current_actions_tensor = valid_actions[0]
                else:
                    # 如果没有有效动作，使用零张量
                    all_current_actions_tensor = tf.zeros((obs.shape[0], self.action_dim), dtype=tf.float32)
                    
            except Exception as e:
                print(f"连接all_current_actions失败: {e}, 形状: {[a.shape if hasattr(a, 'shape') else 'unknown' for a in all_current_actions]}")
                # 使用零张量作为后备
                all_current_actions_tensor = tf.zeros((obs.shape[0], self.action_dim), dtype=tf.float32)
            
            # 使用单个输入调用critic，避免维度不匹配问题
            try:
                # 检查critic模型的输入格式
                if len(self.critic.inputs) == 2:
                    # 标准格式：[obs, actions]
                    q_value = self.critic([obs, all_current_actions_tensor])
                else:
                    # 试图将观察和动作连接起来作为单一输入
                    combined_input = tf.concat([obs, all_current_actions_tensor], axis=1)
                    q_value = self.critic(combined_input)
                
                # 确保q_value是可减均值的
                if len(tf.shape(q_value)) == 0:  # 如果是标量
                    # 直接使用负值
                    actor_loss = -q_value
                else:
                    # 计算均值
                    actor_loss = -tf.reduce_mean(q_value)
            
            except Exception as e:
                print(f"计算actor_loss时出错: {e}")
                # 使用默认损失值
                actor_loss = tf.constant(1.0, dtype=tf.float32)
            
            # 性能优化：添加动作正则化，鼓励更多探索
            actor_loss += 0.001 * tf.reduce_mean(tf.square(current_actions))
            
            # 对于多输出网络，添加额外的正则化以确保力参数的稳定性
            if self.is_multi_output and len(actor_outputs) > 1:
                force_params = actor_outputs[self.actor_force_params_index]
                # 添加软约束，防止力参数变化过快
                actor_loss += 0.0005 * tf.reduce_mean(tf.square(force_params))
        
        # 计算梯度并更新Actor网络权重
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self._actor_opt.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        
        return critic_loss.numpy(), actor_loss.numpy()
    
    # ... existing code ... 