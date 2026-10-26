"""Exercise actual TensorFlow replay, critic and delayed actor updates on CPU."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='-1', TF_CPP_MIN_LOG_LEVEL='3', QUIET_OUTPUT='1',
                  SUPPRESS_TERRAIN_OUTPUT='1', SUPPRESS_REWARD_CONFIG_OUTPUT='1',
                  SUPPRESS_MA_PROMPT='1', TF_NUM_INTRAOP_THREADS='1', TF_NUM_INTEROP_THREADS='1')
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import argparse
output_parser = argparse.ArgumentParser(description=__doc__)
output_parser.add_argument('--output', help='Optional JSON result path')
smoke_options = output_parser.parse_args()
import inspect
from dataclasses import replace
import numpy as np
import tensorflow as tf
from paper3d_train_optimized import parse_args, OptimizedMATD3, LiteReplayBuffer
from action_semantics import select_replay_action_channels
from multiagent.scenarios.paper3d_terrain_energy import Scenario
from multiagent.environment import MultiAgentEnv

sys.argv = ['smoke', '--algo', 'matd3', '--matd3-action-semantics-mode', 'raw_only',
            '--matd3-use-dual-q', 'false', '--matd3-use-separated-gradient', 'false',
            '--matd3-reconstruct-corrected-target', 'false', '--random-action-prob', '0',
            '--actor-hidden', '32,32', '--critic-hidden', '32,32', '--num-envs', '1',
            '--batch-size', '8', '--buffer-size', '64', '--use-fr-feature', '1',
            '--use-pf-feature', '1', '--action-force-ratio', '0.5']
args = parse_args()
args.jit_compile = False
args.amp_mode = 'off'
args.cross_agent_reference_enabled = False
args.matd3_use_hybrid_actor_objective = False
tf.random.set_seed(71)
scenario = Scenario(seed=67, terrain_complexity_level=1, use_dynamic_obstacles=False)
world = scenario.make_world()
world.episode_length = 40
env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,
                    done_callback=scenario.is_done, shared_viewer=False)
def reset_observation():
    result = env.reset()
    return result[0] if isinstance(result, tuple) else result
obs = np.asarray(reset_observation(), dtype=np.float32)
algo = OptimizedMATD3(3, [81]*3, [7]*3, args)
buffer = LiteReplayBuffer(64, 3, [81]*3, [7]*3, use_per=False, seed=71)
maximum_intervention = 0.0
for step in range(16):
    raw, executed, pf, clean, features, random_mask = [x.numpy() for x in
        algo.batch_select_actions_vectorized(obs[None, ...], add_noise=True)]
    next_obs, rewards, dones, info = env.step(executed[0])
    next_obs = np.asarray(next_obs, dtype=np.float32)
    first, second = select_replay_action_channels('raw_only', raw[0], executed[0])
    maximum_intervention = max(maximum_intervention, float(np.linalg.norm(raw[0,:,:3]-executed[0,:,:3])))
    buffer.add_single_aligned(obs, next_obs, first, np.asarray(rewards), np.asarray(dones),
                              fr_value=0.5, act_corrected=second, pf_features=features[0])
    obs = next_obs
    if all(dones):
        obs = np.asarray(reset_observation(), dtype=np.float32)

before_actor = [v.numpy().copy() for v in algo.agents[0]['actor'].trainable_variables]
before_critic = [v.numpy().copy() for v in algo.agents[0]['critic1'].trainable_variables]
original_step = algo._multi_agent_update_step
signature = inspect.signature(original_step.python_function)
checked = []
def observe_step(*values, **kwargs):
    bound = signature.bind(*values, **kwargs)
    for left, right in [('global_a_next_smooth', 'global_a_next_corr'),
                        ('global_actions_raw_all', 'global_actions_corrected_all')]:
        np.testing.assert_allclose(bound.arguments[left].numpy(), bound.arguments[right].numpy(), atol=1e-6)
    checked.append(True)
    return original_step(*values, **kwargs)
algo._multi_agent_update_step = observe_step
# Raw Q must never query the corrected actor or target action, although the
# interaction above must actually have applied APF.
def forbidden_correction(*args, **kwargs):
    raise AssertionError('raw-only update unexpectedly transformed a critic query through APF')
algo._apply_potential_field_correction_from_context_tf = forbidden_correction
algo._apply_potential_field_correction_from_geometry_context_tf = forbidden_correction
for _ in range(4):
    algo.update(buffer, batch_size=8)
original_sample = buffer.sample
def legacy_sample(size):
    return replace(original_sample(size), actions_corr=None)
buffer.sample = legacy_sample
for _ in range(2):
    algo.update(buffer, batch_size=8)
buffer.sample = original_sample

def distance(before, variables):
    return sum(float(np.linalg.norm(a-v.numpy())) for a,v in zip(before, variables))
actor_change = distance(before_actor, algo.agents[0]['actor'].trainable_variables)
critic_change = distance(before_critic, algo.agents[0]['critic1'].trainable_variables)
assert checked and maximum_intervention > 1e-5
assert actor_change > 0 and critic_change > 0
assert all(np.isfinite(v.numpy()).all() for agent in algo.agents for name in ('actor','critic1','critic2')
           for v in agent[name].trainable_variables)
result = dict(status='passed', runtime=sys.executable, device='CPU', environment_steps=16,
              update_calls=6, missing_alias_update_calls=2, checked_batches=len(checked), maximum_apf_intervention_norm=maximum_intervention,
              actor_weight_change_l2_sum=actor_change, critic_weight_change_l2_sum=critic_change,
              evidence_scope='runtime correctness only; not a performance or convergence experiment')
if smoke_options.output:
    output = Path(smoke_options.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
