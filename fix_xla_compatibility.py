#!/usr/bin/env python3
"""
XLA兼容性自动修复脚本

修复优先级：
P0 - 严重影响XLA编译：
  1. 移除 @tf.function 内的 .numpy() 调用
  2. 将 tf.cond 替换为 tf.where
  3. 缓存环境变量和配置标志

用法：
  python3 fix_xla_compatibility.py --check  # 仅检查问题
  python3 fix_xla_compatibility.py --fix    # 执行修复
"""

import re
import sys
from pathlib import Path

class XLACompatibilityFixer:
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.content = self.file_path.read_text()
        self.issues = []
        self.fixes_applied = []
        
    def check_issues(self):
        """检查所有XLA不兼容问题"""
        print("=" * 80)
        print("XLA兼容性检查")
        print("=" * 80)
        
        self._check_numpy_calls()
        self._check_tf_cond()
        self._check_python_variables()
        self._check_dynamic_shapes()
        
        print(f"\n总计发现 {len(self.issues)} 个问题")
        return self.issues
    
    def _check_numpy_calls(self):
        """检查.numpy()调用"""
        pattern = r'\.numpy\(\)'
        matches = list(re.finditer(pattern, self.content))
        
        print(f"\n[P0] 发现 {len(matches)} 处 .numpy() 调用")
        for i, match in enumerate(matches[:10], 1):  # 只显示前10个
            line_num = self.content[:match.start()].count('\n') + 1
            line = self.content.split('\n')[line_num-1].strip()
            print(f"  {line_num}: {line[:80]}")
            self.issues.append(('numpy_call', line_num, match.span()))
        
        if len(matches) > 10:
            print(f"  ... 还有 {len(matches)-10} 处")
    
    def _check_tf_cond(self):
        """检查tf.cond调用"""
        pattern = r'tf\.cond\s*\('
        matches = list(re.finditer(pattern, self.content))
        
        print(f"\n[P0] 发现 {len(matches)} 处 tf.cond 调用")
        for i, match in enumerate(matches, 1):
            line_num = self.content[:match.start()].count('\n') + 1
            line = self.content.split('\n')[line_num-1].strip()
            print(f"  {line_num}: {line[:80]}")
            self.issues.append(('tf_cond', line_num, match.span()))
    
    def _check_python_variables(self):
        """检查@tf.function内的Python变量访问"""
        # 查找所有@tf.function定义的函数
        tf_func_pattern = r'@tf\.function.*?\n\s*def\s+(\w+)'
        matches = list(re.finditer(tf_func_pattern, self.content, re.DOTALL))
        
        print(f"\n[P0] 发现 {len(matches)} 个 @tf.function")
        
        # 简单启发式：查找函数内的self.xxx访问（可能是Python变量）
        for match in matches[:5]:  # 只显示前5个
            func_name = match.group(1)
            line_num = self.content[:match.start()].count('\n') + 1
            print(f"  {line_num}: {func_name}() - 需要手动检查Python变量访问")
            self.issues.append(('python_var', line_num, func_name))
    
    def _check_dynamic_shapes(self):
        """检查动态shape操作"""
        pattern = r'tf\.shape\([^)]+\)\[.*?\]'
        matches = list(re.finditer(pattern, self.content))
        
        print(f"\n[P1] 发现 {len(matches)} 处动态shape操作")
        for i, match in enumerate(matches[:5], 1):
            line_num = self.content[:match.start()].count('\n') + 1
            line = self.content.split('\n')[line_num-1].strip()
            print(f"  {line_num}: {line[:80]}")
            self.issues.append(('dynamic_shape', line_num, match.span()))
        
        if len(matches) > 5:
            print(f"  ... 还有 {len(matches)-5} 处")
    
    def apply_fixes(self):
        """应用自动修复"""
        print("\n" + "=" * 80)
        print("开始应用修复")
        print("=" * 80)
        
        # 修复1：添加配置缓存
        self._fix_add_config_cache()
        
        # 修复2：替换tf.cond为tf.where（简单案例）
        self._fix_simple_tf_cond()
        
        print(f"\n总计应用 {len(self.fixes_applied)} 个修复")
        return self.fixes_applied
    
    def _fix_add_config_cache(self):
        """在__init__中添加配置缓存"""
        init_pattern = r'(def __init__\(self[^)]*\):.*?(?=\n    def |\n\nclass |\Z))'
        match = re.search(init_pattern, self.content, re.DOTALL)
        
        if not match:
            print("[SKIP] 未找到__init__方法")
            return
        
        init_content = match.group(1)
        
        # 检查是否已经添加了缓存
        if 'self.jit_compile_cached' in init_content:
            print("[SKIP] 配置缓存已存在")
            return
        
        cache_code = """
        # === XLA优化：缓存所有环境变量和配置标志，避免在@tf.function中访问 ===
        self.jit_compile_cached = bool(os.getenv('JIT_COMPILE', '0').lower() in ('1', 'true', 'yes', 'on'))
        self.debug_actor_graph = bool(os.getenv('DEBUG_ACTOR_GRAPH', '0').lower() in ('1', 'true', 'yes', 'on'))
        self.use_fr_feature_flag = bool(os.getenv('USE_FR_FEATURE', '1').lower() in ('1', 'true', 'yes', 'on'))
        self.debug_pf_forces_cached = bool(os.getenv('DEBUG_PF_FORCES', '0').lower() in ('1', 'true', 'yes', 'on'))
        self.q_clip_value_cached = float(os.getenv('Q_CLIP_VALUE', '5000.0'))
        self.huber_delta_cached = float(os.getenv('HUBER_DELTA', '1.8'))
        self.c_grad_clip_norm = float(os.getenv('GRAD_CLIP_NORM', '12.0'))
        self.c_gamma = float(os.getenv('GAMMA', '0.95'))
        self.c_map_size = float(os.getenv('MAP_SIZE', '200'))
        self.c_reward_clip = float(os.getenv('REWARD_CLIP_VALUE', '-250.0'))
        """
        
        # 在__init__末尾添加缓存代码
        insert_pos = match.end()
        self.content = self.content[:insert_pos] + cache_code + self.content[insert_pos:]
        
        self.fixes_applied.append(('add_config_cache', 'Added configuration caching'))
        print("[FIXED] 添加配置缓存到__init__")
    
    def _fix_simple_tf_cond(self):
        """替换简单的tf.cond为tf.where"""
        # 模式：tf.cond(condition, lambda: true_val, lambda: false_val)
        pattern = r'tf\.cond\(\s*([^,]+),\s*lambda:\s*([^,]+),\s*lambda:\s*([^)]+)\)'
        
        def replace_fn(match):
            condition = match.group(1)
            true_val = match.group(2).strip()
            false_val = match.group(3).strip()
            
            # 只替换简单的值返回（不包含复杂操作）
            if ('(' not in true_val and '(' not in false_val) or \
               (len(true_val) < 50 and len(false_val) < 50):
                self.fixes_applied.append(('replace_tf_cond', f'{condition} -> tf.where'))
                return f'tf.where({condition}, {true_val}, {false_val})'
            return match.group(0)  # 保持原样
        
        new_content, count = re.subn(pattern, replace_fn, self.content)
        
        if count > 0:
            self.content = new_content
            print(f"[FIXED] 替换了 {count} 处简单的 tf.cond")
    
    def save_fixed_file(self, backup=True):
        """保存修复后的文件"""
        if backup:
            backup_path = self.file_path.with_suffix('.py.xla_backup')
            backup_path.write_text(self.file_path.read_text())
            print(f"\n[BACKUP] 原文件备份到: {backup_path}")
        
        self.file_path.write_text(self.content)
        print(f"[SAVED] 修复后的文件: {self.file_path}")

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ['--check', '--fix']:
        print(__doc__)
        sys.exit(1)
    
    mode = sys.argv[1]
    file_path = '/home/tang/Desktop/paper3d_train_optimized.py'
    
    fixer = XLACompatibilityFixer(file_path)
    
    # 检查问题
    issues = fixer.check_issues()
    
    if mode == '--fix':
        # 应用修复
        fixes = fixer.apply_fixes()
        
        if fixes:
            fixer.save_fixed_file(backup=True)
            print("\n✅ 修复完成！")
            print("\n建议：")
            print("1. 手动检查并替换剩余的 tf.cond（复杂案例）")
            print("2. 将 @tf.function 内的 .numpy() 调用移到函数外")
            print("3. 使用 tf.ensure_shape 固定输入shape")
            print("4. 测试训练是否正常运行")
        else:
            print("\n⚠️  没有应用任何自动修复（可能需要手动处理）")
    
    print(f"\n{'='*80}")
    print(f"检查完成：发现 {len(issues)} 个问题")
    if mode == '--fix':
        print(f"应用修复：完成 {len(fixer.fixes_applied)} 个修复")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()

