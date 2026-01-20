#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import json
import argparse

def infer_per_mode(args_dict):
    per_enabled = bool(args_dict.get('per_enabled', False))
    if not per_enabled:
        return 'off'
    # weighted: per_reward_weight>0 或 per_uniform_mix>0 默认判为weighted
    prw = float(args_dict.get('per_reward_weight', 0.0))
    pum = float(args_dict.get('per_uniform_mix', 0.0))
    if prw > 1e-9 or pum > 1e-9:
        return 'weighted'
    return 'td'

def infer_pf_mode(args_dict):
    afr = float(args_dict.get('action_force_ratio', 0.0) or 0.0)
    use_tf_pf = bool(args_dict.get('use_tf_potential_field', True))
    return 'on' if (use_tf_pf and afr > 0.0) else 'off'

def main():
    parser = argparse.ArgumentParser(description='Plot ablation reward curves on one figure')
    parser.add_argument('--logs-glob', type=str, default='logs/ablation_*/*/results.json',
                        help='Glob pattern to results.json files')
    parser.add_argument('--out', type=str, default='ablation_rewards.png', help='Output image path')
    parser.add_argument('--csv', type=str, default='ablation_summary.csv', help='Output CSV summary path')
    parser.add_argument('--per-exp-outdir', type=str, default='', help='Export per-experiment reward curve PNGs to this directory; empty to disable')
    parser.add_argument('--lastn', type=int, default=10, help='Compute mean of last N episodes')
    args = parser.parse_args()

    files = sorted(glob.glob(args.logs_glob))
    if not files:
        print('No results.json found for pattern:', args.logs_glob)
        return

    # Non-interactive backend
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = []
    curves = []
    labels = []

    for f in files:
        try:
            data = json.load(open(f, 'r'))
        except Exception:
            continue
        rewards = data.get('rewards', [])
        meta = data.get('args', {})
        algo = meta.get('algo', 'maddpg')
        pf = infer_pf_mode(meta)
        per = infer_per_mode(meta)
        label = f"{algo} | pf-{pf} | per-{per}"
        labels.append(label)
        curves.append(rewards)

        best = data.get('best_reward', None)
        if rewards:
            lastn = rewards[-args.lastn:]
            mean_lastn = sum(lastn) / max(1, len(lastn))
        else:
            mean_lastn = None
        exp = os.path.basename(os.path.dirname(f))
        rows.append((exp, algo, pf, per, mean_lastn, best))

        # per-exp 单独奖励曲线
        if args.per_exp_outdir:
            try:
                os.makedirs(args.per_exp_outdir, exist_ok=True)
                import matplotlib.pyplot as _plt
                _plt.figure(figsize=(8, 4.5))
                if rewards:
                    _plt.plot(range(1, len(rewards)+1), rewards, label=f"{algo} | pf-{pf} | per-{per}")
                _plt.xlabel('Episode')
                _plt.ylabel('Reward')
                _plt.title(exp)
                _plt.grid(True, alpha=0.3)
                _plt.legend(loc='best', fontsize=8)
                _plt.tight_layout()
                out_path = os.path.join(args.per_exp_outdir, f"{exp}.png")
                _plt.savefig(out_path, dpi=140)
                _plt.close()
            except Exception as _e:
                print('Per-exp plot failed for', exp, _e)

    # 绘图
    plt.figure(figsize=(12, 7))
    for rewards, label in zip(curves, labels):
        if not rewards:
            continue
        plt.plot(range(1, len(rewards)+1), rewards, label=label, linewidth=1.5)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Ablation: Reward Curves')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=8)
    plt.tight_layout()
    plt.savefig(args.out, dpi=160)
    print('Saved figure to', args.out)

    # 汇总CSV
    try:
        import csv
        with open(args.csv, 'w', newline='') as fp:
            w = csv.writer(fp)
            w.writerow(['exp', 'algo', 'pf', 'per', f'mean_last{args.lastn}', 'best'])
            for r in rows:
                w.writerow(list(r))
        print('Saved summary to', args.csv)
    except Exception as e:
        print('Save CSV failed:', e)

if __name__ == '__main__':
    main()


