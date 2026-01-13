#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理logs目录：删除空文件夹和只有心跳log的文件夹，保留最新的文件夹
"""
import os
import sys
from pathlib import Path
from datetime import datetime

def is_empty_or_heartbeat_only(folder_path):
    """检查文件夹是否为空或只有心跳log"""
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        return False
    
    # 获取所有文件（递归查找）
    files = list(folder.rglob('*'))
    
    # 过滤掉目录，只保留文件
    files = [f for f in files if f.is_file()]
    
    # 如果没有任何文件，认为是空的
    if len(files) == 0:
        return True
    
    # 检查是否只有心跳log文件
    # 心跳log文件通常命名为 heartbeat.log 或包含 heartbeat 关键字
    all_heartbeat = True
    for file in files:
        file_name = file.name.lower()
        # 检查是否是心跳log文件
        is_heartbeat = 'heartbeat' in file_name
        # 如果文件大小很小（<1KB），也可能是心跳log
        if not is_heartbeat and file.stat().st_size < 1024:
            # 检查文件内容是否只包含时间戳或简单信息
            try:
                with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(200)  # 只读前200字符
                    # 如果内容很短或只包含时间戳，可能是心跳log
                    if len(content.strip()) < 50:
                        is_heartbeat = True
            except:
                pass
        
        if not is_heartbeat:
            all_heartbeat = False
            break
    
    return all_heartbeat

def get_latest_folder(logs_dir):
    """获取最新的文件夹（按修改时间）"""
    logs_path = Path(logs_dir)
    if not logs_path.exists():
        return None
    
    # 获取所有文件夹
    folders = [f for f in logs_path.iterdir() if f.is_dir()]
    
    if not folders:
        return None
    
    # 按修改时间排序，返回最新的
    latest = max(folders, key=lambda f: f.stat().st_mtime)
    return latest

def clean_logs(logs_dir='/home/tang/Desktop/logs', dry_run=True):
    """清理logs目录"""
    logs_path = Path(logs_dir)
    if not logs_path.exists():
        print(f"❌ 目录不存在: {logs_dir}")
        return
    
    # 获取最新文件夹
    latest_folder = get_latest_folder(logs_dir)
    if latest_folder:
        print(f"✅ 保留最新文件夹: {latest_folder.name}")
    else:
        print("⚠️  未找到最新文件夹")
    
    # 获取所有文件夹
    all_folders = [f for f in logs_path.iterdir() if f.is_dir()]
    
    deleted_count = 0
    kept_count = 0
    
    for folder in all_folders:
        # 跳过最新文件夹
        if latest_folder and folder == latest_folder:
            kept_count += 1
            continue
        
        # 检查是否为空或只有心跳log
        if is_empty_or_heartbeat_only(folder):
            if dry_run:
                print(f"🔍 [DRY RUN] 将删除: {folder.name}")
            else:
                try:
                    import shutil
                    shutil.rmtree(folder)
                    print(f"🗑️  已删除: {folder.name}")
                    deleted_count += 1
                except Exception as e:
                    print(f"❌ 删除失败 {folder.name}: {e}")
        else:
            kept_count += 1
            if not dry_run:
                print(f"✅ 保留: {folder.name} (包含其他文件)")
    
    print(f"\n📊 统计:")
    print(f"  - 保留: {kept_count} 个文件夹")
    if dry_run:
        print(f"  - [DRY RUN] 将删除: {len(all_folders) - kept_count} 个文件夹")
    else:
        print(f"  - 已删除: {deleted_count} 个文件夹")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='清理logs目录')
    parser.add_argument('--logs-dir', default='/home/tang/Desktop/logs', help='logs目录路径')
    parser.add_argument('--execute', action='store_true', help='执行删除（默认是dry run）')
    args = parser.parse_args()
    
    dry_run = not args.execute
    if dry_run:
        print("🔍 DRY RUN 模式（不会实际删除）")
        print("   使用 --execute 参数来实际执行删除\n")
    else:
        print("⚠️  执行模式（将实际删除文件）\n")
    
    clean_logs(args.logs_dir, dry_run=dry_run)

