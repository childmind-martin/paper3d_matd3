import os
import sys
import warnings

try:
    # 首先尝试导入 gymnasium
    from gymnasium.envs.registration import register
except ImportError:
    try:
        # 如果 gymnasium 不可用，尝试导入 gym
        from gym.envs.registration import register
    except ImportError:
        warnings.warn("既无法导入 gymnasium 也无法导入 gym。请安装其中一个。")
        # 创建一个空的 register 函数以避免错误
        def register(**kwargs):
            warnings.warn("无法注册环境，缺少必要的依赖。")

# Multiagent envs
# ----------------------------------------

register(
    id='MultiagentSimple-v0',
    entry_point='multiagent.envs:SimpleEnv',
    # FIXME(cathywu) currently has to be exactly max_path_length parameters in
    # rllab run script
    max_episode_steps=100,
)

register(
    id='MultiagentSimpleSpeakerListener-v0',
    entry_point='multiagent.envs:SimpleSpeakerListenerEnv',
    max_episode_steps=100,
)

warnings.warn("This code base is no longer maintained, and is not expected to be maintained again in the future. \n"
              "For the past handful of years, these environments been maintained inside of PettingZoo (see "
              "https://pettingzoo.farama.org/environments/mpe/). \nThis maintained version includes documentation, "
              "support for the PettingZoo API, support for current versions of Python, numerous bug fixes, \n"
              "support for installation via pip, and numerous other large quality of life improvements. \nWe "
              "encourage researchers to switch to this maintained version for all purposes other than comparing "
              "to results run on this version of the environments. \n")

if os.getenv('SUPPRESS_MA_PROMPT') != '1':
    prompt = (
        "Please read the raised warning, then press Enter to continue... "
        "(to suppress this prompt, please set the environment variable "
        "`SUPPRESS_MA_PROMPT=1`)\n"
    )
    try:
        if sys.stdin is not None and sys.stdin.isatty():
            input(prompt)
        else:
            warnings.warn(
                "SUPPRESS_MA_PROMPT is not set, but stdin is non-interactive; "
                "skipping multiagent confirmation prompt."
            )
    except EOFError:
        warnings.warn(
            "Encountered EOF while waiting for multiagent confirmation prompt; "
            "continuing in non-interactive mode."
        )
