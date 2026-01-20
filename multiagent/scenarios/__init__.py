import importlib.util
import os.path

def load(name):
    """自定义场景加载函数，使用importlib替代imp"""
    # 🔧 修复：如果name不包含.py扩展名，自动添加
    if not name.endswith('.py'):
        pathname = os.path.join(os.path.dirname(__file__), f"{name}.py")
    else:
        pathname = os.path.join(os.path.dirname(__file__), name)
    
    # 🔧 修复：检查文件是否存在
    if not os.path.exists(pathname):
        raise ImportError(f"场景文件不存在: {pathname}")
    
    spec = importlib.util.spec_from_file_location("scenario", pathname)
    if spec is None:
        raise ImportError(f"无法从 {pathname} 加载场景")
    scenario_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scenario_module)
    return scenario_module