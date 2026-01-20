import pygame
import numpy as np
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import time
import cv2
import math

# 全局常量，避免重复计算和魔法数字
DOUBLEBUF_OPENGL = DOUBLEBUF | OPENGL  # 预先计算标志组合
COLOR_BIT_DEPTH_BIT = GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT  # 预先计算缓冲区位掩码

# 3D实体基类
class Entity3D:
    def __init__(self, position, size=0.1, color=[0.5, 0.5, 0.8], entity_type='agent', name=None):
        self.position = np.array(position, dtype=np.float32)
        self.size = size
        self.color = color
        self.entity_type = entity_type
        self.name = name
        
    def set_position(self, position):
        """设置实体位置"""
        self.position = np.array(position, dtype=np.float32)
        
    def render(self):
        """渲染实体"""
        glPushMatrix()
        # 移动到实体位置
        glTranslatef(self.position[0], self.position[1], self.position[2])
        
        # 设置颜色和材质属性
        glColor3f(self.color[0], self.color[1], self.color[2])
        
        # 设置材质属性
        mat_ambient = [self.color[0] * 0.4, self.color[1] * 0.4, self.color[2] * 0.4, 1.0]
        mat_diffuse = [self.color[0], self.color[1], self.color[2], 1.0]
        mat_specular = [0.5, 0.5, 0.5, 1.0]
        mat_shininess = 30.0
        
        glMaterialfv(GL_FRONT, GL_AMBIENT, mat_ambient)
        glMaterialfv(GL_FRONT, GL_DIFFUSE, mat_diffuse)
        glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular)
        glMaterialf(GL_FRONT, GL_SHININESS, mat_shininess)
        
        # 根据实体类型选择渲染形状
        if self.entity_type == 'agent':
            # 智能体用球体表示
            self._draw_sphere(self.size)
        elif 'obstacle' in self.entity_type or 'obstacl' in self.entity_type or 'barrier' in self.entity_type or 'obstacle' in (self.name or ''):
            # 障碍物使用球体表示
            self._draw_sphere(self.size * 1.2)  # 稍微大一些
        elif 'goal' in self.entity_type or 'goal' in (self.name or ''):
            # 目标点用球体表示
            self._draw_sphere(self.size * 1.1)  # 稍微大一些以便识别
        elif 'landmark' in self.entity_type:
            # 路标也用球体表示
            self._draw_sphere(self.size * 1.0)
        else:
            # 默认使用球体
            self._draw_sphere(self.size)
            
        glPopMatrix()  # 只保留一个pop，与push对应
        
    def _draw_sphere(self, radius):
        """绘制球体"""
        # 使用更高精度的球体
        sphere = gluNewQuadric()
        gluQuadricNormals(sphere, GLU_SMOOTH)  # 平滑法线
        gluQuadricTexture(sphere, GL_TRUE)     # 启用纹理坐标
        gluSphere(sphere, radius, 32, 32)      # 增加精度到32x32
        gluDeleteQuadric(sphere)
        
    def _draw_cube(self, size):
        """绘制立方体"""
        # 一个简单的立方体
        vertices = [
            [-size, -size, -size],
            [size, -size, -size],
            [size, size, -size],
            [-size, size, -size],
            [-size, -size, size],
            [size, -size, size],
            [size, size, size],
            [-size, size, size]
        ]
        
        # 立方体的面
        faces = [
            [0, 1, 2, 3],  # 底面
            [4, 5, 6, 7],  # 顶面
            [0, 1, 5, 4],  # 前面
            [2, 3, 7, 6],  # 后面
            [0, 3, 7, 4],  # 左面
            [1, 2, 6, 5]   # 右面
        ]
        
        glBegin(GL_QUADS)
        for face in faces:
            for idx in face:
                glVertex3f(vertices[idx][0], vertices[idx][1], vertices[idx][2])
        glEnd()
        
    def _draw_cone(self, size):
        """绘制圆锥体表示目标点"""
        quadric = gluNewQuadric()
        gluQuadricDrawStyle(quadric, GLU_FILL)
        # 圆锥底部
        gluCylinder(quadric, size*1.5, 0, size*2, 16, 16)
        gluDeleteQuadric(quadric)

# 球体实体
class Sphere(Entity3D):
    def __init__(self, position, size, color):
        super().__init__(position, size, color)
        self.quad = gluNewQuadric()
        
    def render(self):
        """渲染球体"""
        glPushMatrix()
        
        # 设置位置
        glTranslatef(self.position[0], self.position[1], self.position[2])
        
        # 设置颜色
        glColor3f(self.color[0], self.color[1], self.color[2])
        
        # 渲染球体
        gluSphere(self.quad, self.size, 20, 20)
        
        glPopMatrix()

# 创建3D世界查看器
class Viewer3D:
    def __init__(self, width=800, height=600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height), DOUBLEBUF_OPENGL)
        pygame.display.set_caption("3D Multi-Agent Environment")
        
        # 性能优化：设置PyGame相关
        pygame.event.set_allowed([QUIT, KEYDOWN, KEYUP, MOUSEBUTTONDOWN, MOUSEBUTTONUP, MOUSEMOTION])  # 只允许我们关心的事件类型
        pygame.key.set_repeat(1, 10)  # 提高按键重复率，使键盘操作更流畅
        
        # 添加场景中心点，用于旋转
        self.center = [0.0, 0.0, 0.0]
        self.camera_position_offset = [0.0, 0.0]
        
        # 设置透视投影
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, (width/height), 0.1, 50.0)
        
        # 设置相机位置
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # 开启和配置OpenGL功能，提高性能和质量
        # 开启深度测试
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)  # 使深度测试更准确
        
        # 开启光照效果
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # 设置光源
        light_position = [5.0, 5.0, 10.0, 1.0]  # 光源位置
        light_ambient = [0.2, 0.2, 0.2, 1.0]    # 环境光
        light_diffuse = [0.8, 0.8, 0.8, 1.0]    # 漫反射光
        light_specular = [1.0, 1.0, 1.0, 1.0]   # 镜面反射光
        
        glLightfv(GL_LIGHT0, GL_POSITION, light_position)
        glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse)
        glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular)
        
        # 反锯齿和平滑处理
        glEnable(GL_POINT_SMOOTH)
        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_POLYGON_SMOOTH)
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)
        glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)  # 添加透视校正提示
        
        # 启用混合功能，改善透明度和平滑度
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # 存储所有实体
        self.entities = []
        
        # 视角控制参数
        self.auto_rotate = False           # 默认关闭自动旋转
        self.rotation_speed = 0.5          # 旋转速度
        self.camera_distance = 8.0         # 摄像机距离
        self.camera_elevation = 30.0       # 摄像机仰角
        self.camera_azimuth = 0.0          # 摄像机方位角
        
        # 平滑过渡参数 - 添加目标值和插值因子
        self.target_elevation = 30.0
        self.target_azimuth = 0.0
        self.target_distance = 8.0
        self.smooth_factor = 0.2           # 插值因子，值越小过渡越平滑
        
        # 鼠标交互参数
        self.mouse_down = False
        self.right_mouse_down = False
        self.prev_mouse_pos = None
        self.mouse_sensitivity = 0.3       # 降低鼠标灵敏度，使旋转更平滑
        
        # 增加RGB数组标志，用于向后兼容
        self.need_return_rgb_array = False
        
        # 性能参数
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.fps = 60
        self.show_fps = False
        
        # 初始化视角
        self._update_camera()

    def add_entity(self, position, size=0.1, color=[0.5, 0.5, 0.8], entity_type='agent', name=None):
        """添加一个实体到场景中"""
        # 创建Entity对象
        entity = Entity3D(
            position=position,
            size=size,
            color=color,
            entity_type=entity_type,
            name=name
        )
        self.entities.append(entity)
        # 返回实体索引，用于后续更新
        return len(self.entities) - 1
    
    def update_entity(self, index, position):
        """更新实体位置"""
        if 0 <= index < len(self.entities):
            self.entities[index].position = position
        
    def update(self):
        """更新场景（响应事件、旋转场景等）"""
        # 计算帧率
        current_time = time.time()
        self.frame_count += 1
        if current_time - self.last_frame_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_frame_time = current_time
            if self.show_fps:
                print(f"FPS: {self.fps}")
        
        # 处理所有待处理事件
        for event in pygame.event.get():
            self.handle_event(event)
                
        # 处理连续按键输入，使方向键控制更加流畅
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.target_azimuth -= 1.0  # 减小单次变化量，但由于重复率高，总体效果更流畅
        if keys[pygame.K_RIGHT]:
            self.target_azimuth += 1.0
        if keys[pygame.K_UP]:
            self.target_elevation += 1.0
            if self.target_elevation > 85.0:
                self.target_elevation = 85.0
        if keys[pygame.K_DOWN]:
            self.target_elevation -= 1.0
            if self.target_elevation < 5.0:
                self.target_elevation = 5.0
        if keys[pygame.K_PLUS] or keys[pygame.K_EQUALS]:
            self.target_distance -= 0.1
            if self.target_distance < 1.0:
                self.target_distance = 1.0
        if keys[pygame.K_MINUS]:
            self.target_distance += 0.1
                
        # 仅在自动旋转开启时更新旋转角度
        if self.auto_rotate:
            self.target_azimuth += self.rotation_speed
            if self.target_azimuth >= 360:
                self.target_azimuth -= 360
        
        # 平滑过渡到目标视角参数
        self._smooth_camera_transition()
        
        # 渲染整个场景
        self._render_scene()
        
        # 更新显示
        pygame.display.flip()
        
        # 使用垂直同步控制帧率
        # pygame.time.wait(1)  # 轻微延迟，让其他进程有机会执行
        
        return True
    
    def _smooth_camera_transition(self):
        """实现相机参数的平滑过渡"""
        # 计算当前值与目标值之间的差值，并应用平滑因子
        self.camera_elevation += (self.target_elevation - self.camera_elevation) * self.smooth_factor
        
        # 方位角需要特殊处理，以避免在0-360度之间转换时跳变
        azimuth_diff = (self.target_azimuth - self.camera_azimuth)
        if azimuth_diff > 180:
            azimuth_diff -= 360
        elif azimuth_diff < -180:
            azimuth_diff += 360
        self.camera_azimuth += azimuth_diff * self.smooth_factor
        # 规范化方位角到0-360度
        if self.camera_azimuth >= 360:
            self.camera_azimuth -= 360
        elif self.camera_azimuth < 0:
            self.camera_azimuth += 360
            
        # 摄像机距离平滑过渡
        self.camera_distance += (self.target_distance - self.camera_distance) * self.smooth_factor
        
        # 更新相机位置
        self._update_camera()
        
    def _update_camera(self):
        """更新相机位置和视角 - 优化版MATLAB风格"""
        # 重置模型视图矩阵
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # 将仰角和方位角转换为弧度
        elev_rad = math.radians(self.camera_elevation)
        azim_rad = math.radians(self.camera_azimuth)
        
        # 计算相机位置（球坐标系转笛卡尔坐标系）
        x = self.camera_distance * math.cos(elev_rad) * math.sin(azim_rad)
        y = self.camera_distance * math.cos(elev_rad) * math.cos(azim_rad)
        z = self.camera_distance * math.sin(elev_rad)
        
        # 设置视角 - 固定目标中心点
        gluLookAt(
            x + self.center[0] + self.camera_position_offset[0],  # 相机位置
            y + self.center[1] + self.camera_position_offset[1], 
            z + self.center[2],
            self.center[0], self.center[1], self.center[2],      # 观察中心点
            0, 0, 1                                              # 上方向
        )
        
    def _render_scene(self):
        """渲染整个场景"""
        # 清除缓冲区 - 使用预计算的常量减少函数调用开销
        glClear(COLOR_BIT_DEPTH_BIT)
        
        # 绘制网格
        self._draw_grid()
        
        # 渲染所有实体 - 批量处理以提高性能
        for entity in self.entities:
            entity.render()
        
        # 如果启用FPS显示，在屏幕上绘制当前FPS
        if self.show_fps:
            self._draw_fps()

    def _draw_fps(self):
        """在屏幕上绘制FPS信息"""
        # 临时禁用光照和深度测试
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        
        # 切换到正交投影进行2D绘制
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, 0, self.height, -1, 1)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        # 绘制FPS文本
        glColor3f(1.0, 1.0, 1.0)
        # OpenGL没有内置文本绘制功能，这里简单绘制一个指示块
        glBegin(GL_QUADS)
        glVertex2f(10, self.height - 20)
        glVertex2f(50, self.height - 20)
        glVertex2f(50, self.height - 10)
        glVertex2f(10, self.height - 10)
        glEnd()
        
        # 恢复投影矩阵
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        
        # 恢复GL状态
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

    def _draw_grid(self):
        """绘制网格作为参考"""
        # 临时禁用光照以绘制网格
        glDisable(GL_LIGHTING)
        
        # 设置网格颜色
        glColor3f(0.3, 0.3, 0.3)
        
        # 绘制网格线
        glBegin(GL_LINES)
        
        grid_size = 5
        for i in range(-grid_size, grid_size + 1):
            # 平行于x轴的线
            glVertex3f(-grid_size, i, 0)
            glVertex3f(grid_size, i, 0)
            
            # 平行于y轴的线
            glVertex3f(i, -grid_size, 0)
            glVertex3f(i, grid_size, 0)
        
        glEnd()
        
        # 绘制坐标轴
        glBegin(GL_LINES)
        # X轴（红色）
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(1.0, 0.0, 0.0)
        
        # Y轴（绿色）
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 1.0, 0.0)
        
        # Z轴（蓝色）
        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, 1.0)
        glEnd()
        
        # 重新启用光照
        glEnable(GL_LIGHTING)
    
    def render(self, return_rgb_array=False):
        """渲染场景"""
        # 使用统一的渲染函数
        self._update_camera()
        self._render_scene()
        
        # 交换缓冲区，显示渲染结果
        pygame.display.flip()
        
        # 返回像素数据 (用于rgb_array模式)
        if return_rgb_array:
            try:
                buffer = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
                buffer = np.frombuffer(buffer, dtype=np.uint8).reshape(self.height, self.width, 3)
                buffer = np.flipud(buffer)  # OpenGL 和 Numpy 图像坐标系不同
                return buffer
            except Exception as e:
                print(f"读取像素缓冲区失败: {e}")
                # 如果读取失败，返回空白图像
                return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        return True
    
    def _create_backup_image(self):
        """创建备用的2D图像，显示更多信息"""
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img[:,:] = [30, 30, 80]  # 深蓝色背景
        
        # 添加网格线表示
        for x in range(0, self.width, 50):
            cv2.line(img, (x, 0), (x, self.height), [50, 50, 100], 1)
        for y in range(0, self.height, 50):
            cv2.line(img, (0, y), (self.width, y), [50, 50, 100], 1)
            
        # 添加智能体信息文本
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, "3D渲染失败，使用备用2D视图", (30, 30), font, 0.7, [255, 255, 255], 2)
        
        # 绘制所有实体
        center_x, center_y = self.width//2, self.height//2
        for i, entity in enumerate(self.entities):
            # 将3D坐标映射到2D图像 (改进版本)
            x = int(center_x + (entity.position[0] * 100))
            # Y和Z坐标共同确定屏幕Y位置，实现透视效果
            y = int(center_y - (entity.position[1] * 80) - (entity.position[2] * 40))
            
            radius = int(max(5, entity.size * 30))
            color = [int(c * 255) for c in entity.color]
            # 绘制实体
            cv2.circle(img, (x, y), radius, color, -1)
            
            # 添加标签
            label = f"{entity.entity_type}_{i}" if entity.name is None else entity.name
            cv2.putText(img, label, (x+radius+5, y+5), font, 0.5, [255, 255, 255], 1)
            
            # 添加位置信息
            pos_text = f"({entity.position[0]:.1f}, {entity.position[1]:.1f}, {entity.position[2]:.1f})"
            cv2.putText(img, pos_text, (x+radius+5, y+25), font, 0.4, [200, 200, 200], 1)
        
        return img
    
    def close(self):
        """关闭查看器"""
        pygame.quit()

    def handle_event(self, event):
        """处理单个事件"""
        if event.type == pygame.QUIT:
            pygame.quit()
            return False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                return False
            # 按R键开关自动旋转
            elif event.key == pygame.K_r:
                self.auto_rotate = not self.auto_rotate
                print(f"自动旋转: {'开启' if self.auto_rotate else '关闭'}")
            # 按F键开关FPS显示
            elif event.key == pygame.K_f:
                self.show_fps = not self.show_fps
                print(f"FPS显示: {'开启' if self.show_fps else '关闭'}")
            # 空格键重置视角
            elif event.key == pygame.K_SPACE:
                self.reset_view()
                
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # 左键
                self.mouse_down = True
                self.prev_mouse_pos = pygame.mouse.get_pos()
            elif event.button == 3:  # 右键
                self.right_mouse_down = True
                self.prev_mouse_pos = pygame.mouse.get_pos()
            elif event.button == 4:  # 滚轮上滚
                self.target_distance -= 0.5
                if self.target_distance < 1.0:
                    self.target_distance = 1.0
            elif event.button == 5:  # 滚轮下滚
                self.target_distance += 0.5
                
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:  # 左键
                self.mouse_down = False
            elif event.button == 3:  # 右键
                self.right_mouse_down = False
                
        elif event.type == pygame.MOUSEMOTION:
            current_pos = pygame.mouse.get_pos()
            if self.prev_mouse_pos is None:
                self.prev_mouse_pos = current_pos
                return True
                
            dx = current_pos[0] - self.prev_mouse_pos[0]
            dy = current_pos[1] - self.prev_mouse_pos[1]
            
            if self.mouse_down:  # 左键拖动 - 旋转视角
                # MATLAB风格旋转 - 使用更小的变化量和目标值
                self.target_azimuth += dx * self.mouse_sensitivity
                self.target_elevation -= dy * self.mouse_sensitivity
                
                # 限制仰角范围
                if self.target_elevation > 85.0:
                    self.target_elevation = 85.0
                if self.target_elevation < 5.0:
                    self.target_elevation = 5.0
                    
            elif self.right_mouse_down:  # 右键拖动 - 平移视角
                # 计算相机坐标系中的偏移量
                azim_rad = math.radians(self.camera_azimuth)
                
                # 计算相机坐标系中的移动向量 - 降低敏感度
                dx_cam = dx * 0.005
                dy_cam = -dy * 0.005
                
                # 将相机坐标系的移动转换为世界坐标系
                dx_world = dx_cam * math.cos(azim_rad) - dy_cam * math.sin(azim_rad)
                dy_world = dx_cam * math.sin(azim_rad) + dy_cam * math.cos(azim_rad)
                
                # 更新偏移量
                self.camera_position_offset[0] += dx_world
                self.camera_position_offset[1] += dy_world
                
                # 立即更新相机位置，平移无需平滑
                self._update_camera()
                
            self.prev_mouse_pos = current_pos
            
        return True

    def reset_view(self):
        """重置视角到默认值"""
        # 设置目标值而不是直接设置当前值，允许平滑过渡
        self.target_azimuth = 0.0
        self.target_elevation = 30.0
        self.target_distance = 8.0
        
        # 一些参数需要立即重置
        self.camera_position_offset = [0.0, 0.0]
        self.center = [0.0, 0.0, 0.0]
        self.auto_rotate = False
        print("视角已重置")

    def add_mountain(self, mountain_obj=None, mountain_data=None):
        """添加山脉地形到场景"""
        mountain = MountainEntity(mountain_data)
        mountain.build_mountain(mountain_obj)
        # 将山脉作为第一个实体，确保它在所有其他实体之下
        self.entities.insert(0, mountain)
        return mountain

# 创建3D世界的便捷函数
def create_3d_world(width=800, height=600):
    """创建并返回3D查看器"""
    viewer = Viewer3D(width, height)
    return viewer

class MountainEntity(Entity3D):
    """表示山脉地形的3D实体"""
    def __init__(self, mountain_data=None, color=[0.6, 0.4, 0.2]):
        super().__init__(position=[0, 0, 0], size=1.0, color=color, entity_type='mountain')
        self.mountain_data = mountain_data
        self.display_list = None  # 用于存储编译后的显示列表
        self.built = False
        
    def build_mountain(self, mountain_obj=None):
        """从山脉对象构建地形网格"""
        if mountain_obj:
            self.mountain_obj = mountain_obj
            # 需要从山脉对象中提取必要的数据
            self.peak_positions = getattr(mountain_obj, 'peak_positions', [])
            self.peak_heights = getattr(mountain_obj, 'peak_heights', [])
            self.mountain_range = getattr(mountain_obj, 'mountain_range', 4.0) / 2
            self.get_height = getattr(mountain_obj, 'get_height', lambda x, y: 0)
            
            # 生成地形网格
            resolution = 40  # 网格分辨率
            self.grid_x = np.linspace(-self.mountain_range, self.mountain_range, resolution)
            self.grid_y = np.linspace(-self.mountain_range, self.mountain_range, resolution)
            self.grid_z = np.zeros((resolution, resolution))
            
            # 计算高度场
            for i in range(resolution):
                for j in range(resolution):
                    self.grid_z[i, j] = self.get_height(self.grid_x[i], self.grid_y[j])
            
            self.built = True
        elif self.mountain_data:
            # 从数据字典构建
            grid_data = self.mountain_data.get('mountain_grid', {})
            self.grid_x = grid_data.get('X', np.linspace(-2, 2, 20))
            self.grid_y = grid_data.get('Y', np.linspace(-2, 2, 20))
            self.grid_z = grid_data.get('Z', np.zeros((20, 20)))
            
            mountain_info = self.mountain_data.get('mountain_info', {})
            self.peak_positions = mountain_info.get('peak_positions', [])
            self.peak_heights = mountain_info.get('peak_heights', [])
            self.mountain_range = mountain_info.get('mountain_range', 2.0)
            
            self.built = True
    
    def render(self):
        """渲染山脉地形"""
        if not self.built:
            # 如果地形还未构建，跳过渲染
            return
        
        # 临时禁用光照以使用自定义着色
        glPushAttrib(GL_LIGHTING_BIT)
        
        # 设置材质属性
        ambient = [self.color[0] * 0.3, self.color[1] * 0.3, self.color[2] * 0.3, 1.0]
        diffuse = [self.color[0], self.color[1], self.color[2], 1.0]
        specular = [0.2, 0.2, 0.2, 1.0]
        
        glMaterialfv(GL_FRONT, GL_AMBIENT, ambient)
        glMaterialfv(GL_FRONT, GL_DIFFUSE, diffuse)
        glMaterialfv(GL_FRONT, GL_SPECULAR, specular)
        glMaterialf(GL_FRONT, GL_SHININESS, 10.0)
        
        # 使用显示列表提高性能
        if self.display_list is None:
            self.display_list = glGenLists(1)
            glNewList(self.display_list, GL_COMPILE)
            
            # 绘制地形网格
            glBegin(GL_TRIANGLES)
            max_height = max(max(row) for row in self.grid_z) if len(self.grid_z) > 0 else 1.0
            for i in range(len(self.grid_x) - 1):
                for j in range(len(self.grid_y) - 1):
                    # 获取四个顶点的坐标和高度
                    x1, y1 = self.grid_x[i], self.grid_y[j]
                    x2, y2 = self.grid_x[i+1], self.grid_y[j+1]
                    z11 = self.grid_z[i][j]
                    z12 = self.grid_z[i][j+1]
                    z21 = self.grid_z[i+1][j]
                    z22 = self.grid_z[i+1][j+1]
                    
                    # 计算颜色（基于高度）
                    c11 = [0.3 + 0.4*(z11/max_height), 0.2 + 0.3*(z11/max_height), 0.1, 1.0]
                    c12 = [0.3 + 0.4*(z12/max_height), 0.2 + 0.3*(z12/max_height), 0.1, 1.0]
                    c21 = [0.3 + 0.4*(z21/max_height), 0.2 + 0.3*(z21/max_height), 0.1, 1.0]
                    c22 = [0.3 + 0.4*(z22/max_height), 0.2 + 0.3*(z22/max_height), 0.1, 1.0]
                    
                    # 只渲染有高度的区域（节省资源）
                    if z11 > 0.01 or z12 > 0.01 or z21 > 0.01:
                        # 计算法线向量（手动计算提高精度）
                        v1 = [x2-x1, 0, z21-z11]
                        v2 = [0, y2-y1, z12-z11]
                        normal = np.cross(v1, v2)
                        normal = normal / np.linalg.norm(normal)
                        
                        # 第一个三角形: (x1,y1,z11), (x2,y1,z21), (x1,y2,z12)
                        glColor3f(c11[0], c11[1], c11[2])
                        glNormal3f(normal[0], normal[1], normal[2])
                        glVertex3f(x1, y1, z11)
                        
                        glColor3f(c21[0], c21[1], c21[2])
                        glVertex3f(x2, y1, z21)
                        
                        glColor3f(c12[0], c12[1], c12[2])
                        glVertex3f(x1, y2, z12)
                    
                    if z22 > 0.01 or z12 > 0.01 or z21 > 0.01:
                        # 计算第二个三角形的法线
                        v1 = [x1-x2, 0, z12-z22]
                        v2 = [0, y1-y2, z21-z22]
                        normal = np.cross(v1, v2)
                        normal = normal / np.linalg.norm(normal)
                        
                        # 第二个三角形: (x2,y2,z22), (x1,y2,z12), (x2,y1,z21)
                        glColor3f(c22[0], c22[1], c22[2])
                        glNormal3f(normal[0], normal[1], normal[2])
                        glVertex3f(x2, y2, z22)
                        
                        glColor3f(c12[0], c12[1], c12[2])
                        glVertex3f(x1, y2, z12)
                        
                        glColor3f(c21[0], c21[1], c21[2])
                        glVertex3f(x2, y1, z21)
            glEnd()
            
            # 高亮显示山峰
            if self.peak_positions and self.peak_heights:
                glPointSize(5.0)
                glBegin(GL_POINTS)
                for pos, height in zip(self.peak_positions, self.peak_heights):
                    # 只标记较高的山峰
                    if height > 0.5:
                        glColor3f(0.8, 0.3, 0.2)
                        glVertex3f(pos[0], pos[1], height)
                glEnd()
            
            glEndList()
        
        # 调用显示列表
        glCallList(self.display_list)
        
        # 恢复之前的属性
        glPopAttrib()