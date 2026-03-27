#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from ablation_action_pf_comparison import (
    load_metrics,
    plot_comparison_rewards,
    plot_comparison_success_collision_clearance,
    plot_comparison_losses,
    setup_english_fonts
)

def main():
    setup_english_fonts()
    
    batch_dir = Path("/home/tang/Desktop/ablation_experiments/latest")
    
    EXPERIMENT_CONFIGS = [
        {
            "label": "baseline_no_shuffle",
            "name": "No Role Shuffle",
            "name_en": "No Role Shuffle",
        },
        {
            "label": "role_shuffle_enabled",
            "name": "With Role Shuffle",
            "name_en": "With Role Shuffle",
        }
    ]
    
    results = []
    # Try to find the log directories mapping to the labels
    logs_root = Path("/home/tang/Desktop/logs")
    
    for cfg in EXPERIMENT_CONFIGS:
        label = cfg["label"]
        # Find latest log dir matching the label prefix
        candidates = []
        for root, dirs, files in os.walk(logs_root):
            if "results.json" in files:
                rel_path = Path(root).relative_to(logs_root)
                if rel_path.parts[0].startswith(f"{label}_"):
                    candidates.append(Path(root))

        
        if not candidates:
            print(f"Skipping {label}, no logs found.")
            continue
            
        candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        latest_log_dir = candidates[0]
        
        print(f"Loading metrics for {label} from {latest_log_dir}...")
        metrics = load_metrics(str(latest_log_dir))
        if metrics:
            results.append({
                "label": label,
                "name": cfg["name"],
                "name_en": cfg["name_en"],
                "metrics": metrics,
            })
            
    if not results:
        print("No results to plot.")
        return

    output_dir = batch_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    title = "Role Randomization Ablation Comparison"
    
    try:
        plot_comparison_rewards(results, title, output_dir / "rewards_comparison.png")
        plot_comparison_success_collision_clearance(results, title, output_dir / "metrics_comparison.png")
        plot_comparison_losses(results, title, output_dir / "losses_comparison.png")
        print(f"\n✅ All plots saved to: {output_dir}/")
        
        # Also copy to ablation_role_shuffle_outputs for visibility
        compat_dir = Path("/home/tang/Desktop/ablation_role_shuffle_outputs")
        compat_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        for f in output_dir.glob("*.png"):
            shutil.copy(f, compat_dir / f.name)
            
    except Exception as e:
        print(f"\n❌ Failed to generate plots: {e}")

if __name__ == "__main__":
    main()
