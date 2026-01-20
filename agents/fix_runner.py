import re

def main():
    """修复MADDPGRunner类，添加缺失的run_episode和plot_3d_trajectory方法"""
    print("开始修复MADDPGRunner类...")
    
    # 读取原始文件
    filepath = 'maddpg_runner.py'
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查run_episode方法是否定义在类外部
    run_episode_outside = False
    if 'def run_episode(self,' in content and '\ndef run_episode(self,' in content:
        run_episode_outside = True
        print("检测到run_episode方法定义在类外部")
    
    # 如果run_episode在类外部，我们需要将其添加到类内部
    if run_episode_outside:
        # 提取外部的run_episode方法
        run_episode_pattern = re.compile(r'\ndef run_episode\(self,.*?return states, reward_episode, info, images', re.DOTALL)
        match = run_episode_pattern.search(content)
        
        if match:
            # 提取方法并正确缩进
            run_episode_code = match.group(0)
            run_episode_code = run_episode_code.lstrip('\n')  # 移除开头的换行符
            run_episode_code = '    ' + run_episode_code.replace('\n', '\n    ')  # 添加缩进
            
            # 从内容中删除外部方法
            content = run_episode_pattern.sub('', content)
            
            # 添加方法到类中的适当位置
            class_end_pattern = re.compile(r'class MADDPGRunner\(.*?):.*?(?=\n\n\n|\n\ndef|\Z)', re.DOTALL)
            match = class_end_pattern.search(content)
            if match:
                class_content = match.group(0)
                # 在类末尾添加方法
                new_class_content = class_content + '\n' + run_episode_code
                content = content.replace(class_content, new_class_content)
                print("run_episode方法已移动到类内部")
            else:
                print("无法定位类结束位置")
        else:
            print("无法提取run_episode方法")
    
    # 检查plot_3d_trajectory方法是否存在
    if 'def plot_3d_trajectory(self,' not in content:
        print("添加缺失的plot_3d_trajectory方法")
        
        # 创建plot_3d_trajectory方法
        plot_3d_code = '''
    def plot_3d_trajectory(self, info, output_path=None):
        """绘制3D轨迹图并保存"""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        try:
            # 避免中文字体问题，直接使用英文
            plt.rcParams['font.sans-serif'] = ['Arial']
            # 使用ASCII负号替代Unicode负号
            plt.rcParams['axes.unicode_minus'] = False
            
            # 提取轨迹数据
            trajectories = []
            for i in range(self.n_agents):
                traj = []
                for step_info in info:
                    if 'positions' in step_info and i < len(step_info['positions']):
                        traj.append(step_info['positions'][i])
                trajectories.append(traj)
            
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # 绘制智能体轨迹
            for i, trajectory in enumerate(trajectories):
                if not trajectory:
                    continue
                
                # 确保数据是浮点数
                x = [float(point[0]) for point in trajectory]
                y = [float(point[1]) for point in trajectory]
                z = [float(point[2]) if len(point) > 2 else 0 for point in trajectory]
                
                # 使用鲜明颜色
                line_color = ['red', 'blue', 'green', 'purple', 'orange'][i % 5]
                
                # 绘制完整轨迹线
                ax.plot(x, y, z, color=line_color, label=f'Agent {i}', linewidth=3.5)
                
                # 标记起点和终点
                ax.scatter(x[0], y[0], z[0], color='green', s=150, marker='^', label='Start' if i == 0 else "")
                ax.scatter(x[-1], y[-1], z[-1], color='red', s=150, marker='o', label='End' if i == 0 else "")
            
            # 设置图表属性
            ax.set_xlabel('X Axis', fontsize=14)
            ax.set_ylabel('Y Axis', fontsize=14) 
            ax.set_zlabel('Z Axis', fontsize=14)
            ax.legend(loc='upper right', fontsize=12)
            ax.set_title('Agent Trajectories', fontsize=16)
            ax.grid(True)
            
            # 保存图表
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"轨迹图已保存到: {output_path}")
            
            return fig
        except Exception as e:
            print(f"绘制3D轨迹图失败: {e}")
            import traceback
            traceback.print_exc()
            return None
'''
        
        # 将方法添加到类内部
        class_end_pattern = re.compile(r'class MADDPGRunner\(.*?):.*?(?=\n\n\n|\n\ndef|\Z)', re.DOTALL)
        match = class_end_pattern.search(content)
        if match:
            class_content = match.group(0)
            # 在类末尾添加方法
            new_class_content = class_content + plot_3d_code
            content = content.replace(class_content, new_class_content)
            print("plot_3d_trajectory方法已添加到类内部")
        else:
            print("无法定位类结束位置")
    
    # 如果run_episode方法不存在，添加它
    if 'def run_episode(self,' not in content:
        print("添加缺失的run_episode方法")
        
        # 创建run_episode方法
        run_episode_code = '''
    def run_episode(self, max_episode_len=200, policy_param='last', render=True, waitTime=0.0, mode=None, test_mode=True):
        """运行一个回合（用于评估）"""
        # 设置使用哪个策略
        for agent in self.agents:
            if hasattr(agent, 'set_policy_for_execution'):
                agent.set_policy_for_execution(policy_param)
        
        # 初始化状态
        states = self.env.reset()
        reward_episode = 0
        images = []
        info = []
        
        # 初始化智能体轨迹数组
        trajectories = [[] for _ in range(self.n_agents)]
        
        # 记录初始位置
        for i in range(self.n_agents):
            if hasattr(self.env.world.agents[i], 'state') and hasattr(self.env.world.agents[i].state, 'p_pos'):
                pos = self.env.world.agents[i].state.p_pos.copy()
                trajectories[i].append(pos)
                
                # 如果智能体有轨迹属性，更新它
                if not hasattr(self.env.world.agents[i], '_trajectory'):
                    self.env.world.agents[i]._trajectory = []
                self.env.world.agents[i]._trajectory.append(pos.copy())
        
        # 执行回合
        for step in range(max_episode_len):
            # 每个智能体选择动作
            actions = []
            for i, agent in enumerate(self.agents):
                action = agent.policy(tf.expand_dims(states[i], 0))
                action_np = action.numpy()[0]
                actions.append(action_np)
            
            # 执行动作
            next_states, reward, dones, _ = self.env.step(actions)
            
            # 记录轨迹
            for i in range(self.n_agents):
                if hasattr(self.env.world.agents[i], 'state') and hasattr(self.env.world.agents[i].state, 'p_pos'):
                    pos = self.env.world.agents[i].state.p_pos.copy()
                    trajectories[i].append(pos)
                    
                    # 更新智能体轨迹属性
                    if hasattr(self.env.world.agents[i], '_trajectory'):
                        self.env.world.agents[i]._trajectory.append(pos.copy())
                    
            # 记录信息
            step_info = {
                'step': step,
                'reward': reward,
                'positions': [agent.state.p_pos.copy() for agent in self.env.world.agents if hasattr(agent, 'state') and hasattr(agent.state, 'p_pos')]
            }
            info.append(step_info)
            
            # 累计奖励
            reward_episode += sum(reward)
            
            # 渲染
            if render:
                try:
                    if mode == 'rgb_array':
                        img = self.env.render(mode='rgb_array')
                        if img is not None:
                            images.append(img.copy())
                    else:
                        # 使用 rgb_array 作为后备方案
                        try:
                            self.env.render(mode='human')
                        except Exception as e:
                            img = self.env.render(mode='rgb_array')
                            if img is not None and len(images) > 0:
                                images.append(img.copy())
                except Exception as e:
                    print(f'渲染失败: {e}')
                
                if waitTime > 0:
                    time.sleep(waitTime)
            
            states = next_states
            if all(dones):
                break
        
        # 打印轨迹信息
        for i, traj in enumerate(trajectories):
            print(f'智能体 {i} 轨迹点数量: {len(traj)}')
            if len(traj) >= 2:
                print(f'  起点: {traj[0]}')
                print(f'  终点: {traj[-1]}')
                print(f'  移动距离: {np.linalg.norm(np.array(traj[-1]) - np.array(traj[0]))}')
        
        return states, reward_episode, info, images
'''
        
        # 将方法添加到类内部
        class_end_pattern = re.compile(r'class MADDPGRunner\(.*?):.*?(?=\n\n\n|\n\ndef|\Z)', re.DOTALL)
        match = class_end_pattern.search(content)
        if match:
            class_content = match.group(0)
            # 在类末尾添加方法
            new_class_content = class_content + run_episode_code
            content = content.replace(class_content, new_class_content)
            print("run_episode方法已添加到类内部")
        else:
            print("无法定位类结束位置")
    
    # 保存修改后的文件
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("MADDPGRunner类修复完成！")

if __name__ == "__main__":
    main()
