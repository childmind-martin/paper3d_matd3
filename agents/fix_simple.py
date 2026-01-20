import os

def main():
    """添加缺失的run_episode和plot_3d_trajectory方法到MADDPGRunner类"""
    print("开始修复MADDPGRunner类...")
    
    # 读取原始文件
    filepath = 'maddpg_runner.py'
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.readlines()
    
    # 找到类的最后一行
    last_line = 0
    class_start = 0  # 初始化class_start变量
    for i, line in enumerate(content):
        if line.startswith('class MADDPGRunner'):
            class_start = i
        if line.strip().startswith('def ') and i > class_start:
            last_line = i - 1
            break
    
    if last_line == 0:  # 如果没有找到类之后的函数定义，假设它是文件的最后
        last_line = len(content) - 1
    
    # 创建要添加的方法内容
    run_episode_code = """
    def run_episode(self, max_episode_len=200, policy_param='last', render=True, waitTime=0.0, mode=None, test_mode=True):
        \"\"\"运行一个回合（用于评估）\"\"\"
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
                        self.env.render(mode='human')
                except Exception as e:
                    print(f'渲染失败: {e}')
                
                if waitTime > 0:
                    import time
                    time.sleep(waitTime)
            
            states = next_states
            if all(dones):
                break
        
        return states, reward_episode, info, images

    def plot_3d_trajectory(self, info, output_path=None):
        \"\"\"绘制3D轨迹图并保存\"\"\"
        import matplotlib.pyplot as plt
        
        try:
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
"""
    
    # 在适当位置插入新的方法
    new_content = content[:last_line+1] + [run_episode_code] + content[last_line+1:]
    
    # 保存修改后的文件
    with open(filepath + '.new', 'w', encoding='utf-8') as f:
        f.writelines(new_content)
    
    # 备份原文件
    os.rename(filepath, filepath + '.bak')
    # 将新文件重命名为原文件名
    os.rename(filepath + '.new', filepath)
    
    print("MADDPGRunner类修复完成！")

if __name__ == "__main__":
    main() 