# MADDPG/multiagent/event_handler.py
import pygame
from pygame.locals import *

class EventHandler:
    def __init__(self):
        self.quit_requested = False
        
    def process_events(self, viewer=None):
        """处理所有待处理的Pygame事件"""
        for event in pygame.event.get():
            # 检查退出事件
            if event.type == pygame.QUIT:
                self.quit_requested = True
                return False
                
            # 检查键盘事件
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.quit_requested = True
                    return False
                    
                # 摄像机控制
                if viewer:
                    # 自动旋转开关
                    if event.key == pygame.K_r:
                        if hasattr(viewer, 'auto_rotate'):
                            viewer.auto_rotate = not viewer.auto_rotate
                            print(f"自动旋转: {'开启' if viewer.auto_rotate else '关闭'}")
                    
                    # 重置视图
                    elif event.key == pygame.K_SPACE:
                        if hasattr(viewer, 'reset_view'):
                            viewer.reset_view()
                            print("视图已重置")
                    
                    # 摄像机操作 - 位置控制
                    elif event.key in [pygame.K_w, pygame.K_a, pygame.K_s, pygame.K_d, pygame.K_q, pygame.K_e]:
                        self._handle_camera_movement(event.key, viewer)
                        
                    # 摄像机操作 - 角度控制
                    elif event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]:
                        self._handle_camera_rotation(event.key, viewer)
            
            # 鼠标事件处理
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if viewer and hasattr(viewer, 'mouse_down'):
                    if event.button == 1:  # 左键
                        viewer.mouse_down = True
                        viewer.prev_mouse_pos = pygame.mouse.get_pos()
                        
            elif event.type == pygame.MOUSEBUTTONUP:
                if viewer and hasattr(viewer, 'mouse_down'):
                    if event.button == 1:  # 左键
                        viewer.mouse_down = False
                        
            elif event.type == pygame.MOUSEMOTION:
                if viewer and hasattr(viewer, 'mouse_down') and viewer.mouse_down:
                    if hasattr(viewer, 'prev_mouse_pos'):
                        current_pos = pygame.mouse.get_pos()
                        dx = current_pos[0] - viewer.prev_mouse_pos[0]
                        dy = current_pos[1] - viewer.prev_mouse_pos[1]
                        
                        if hasattr(viewer, 'camera_angle'):
                            viewer.camera_angle += dx * 0.5
                        if hasattr(viewer, 'camera_height'):
                            viewer.camera_height += dy * 0.05
                            
                        viewer.prev_mouse_pos = current_pos
                        
        return True
        
    def _handle_camera_movement(self, key, viewer):
        """处理摄像机移动"""
        if not hasattr(viewer, 'camera_position_offset'):
            return
            
        step = 0.5  # 移动步长
        
        if key == pygame.K_w:  # 前
            viewer.camera_position_offset[1] += step
        elif key == pygame.K_s:  # 后
            viewer.camera_position_offset[1] -= step
        elif key == pygame.K_a:  # 左
            viewer.camera_position_offset[0] -= step
        elif key == pygame.K_d:  # 右
            viewer.camera_position_offset[0] += step
        elif key == pygame.K_q:  # 上
            if hasattr(viewer, 'camera_height'):
                viewer.camera_height += step
        elif key == pygame.K_e:  # 下
            if hasattr(viewer, 'camera_height'):
                viewer.camera_height -= step
                
    def _handle_camera_rotation(self, key, viewer):
        """处理摄像机旋转"""
        if not hasattr(viewer, 'camera_angle'):
            return
            
        angle_step = 5.0  # 角度步长
        height_step = 0.5  # 高度步长
        
        if key == pygame.K_LEFT:
            viewer.camera_angle -= angle_step
        elif key == pygame.K_RIGHT:
            viewer.camera_angle += angle_step
        elif key == pygame.K_UP and hasattr(viewer, 'camera_height'):
            viewer.camera_height += height_step
        elif key == pygame.K_DOWN and hasattr(viewer, 'camera_height'):
            viewer.camera_height -= height_step