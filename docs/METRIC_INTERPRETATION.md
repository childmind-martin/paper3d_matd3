# Metric Interpretation

This document explains how the main paper metrics should be read when inspecting Table 5, Table 6, and the related result artifacts.

## Team Success Rate

Team success rate is the strict all-agent completion metric. An episode is counted as successful only when all three UAVs satisfy the required goal-arrival and safety conditions under the reported protocol.

This metric is the most direct indicator of coordinated task completion. It is also intentionally demanding: a run with one or two arriving UAVs is not counted as team success if the third UAV fails to arrive or violates the safety condition.

## Any-Agent Arrival

Any-agent arrival reports whether at least one UAV reaches its goal condition. It is included because strict team success can be low in the coupled terrain-obstacle-cooperation setting.

This metric helps distinguish policies that make some task progress from policies that fail to bring any vehicle to a goal region. It is not a replacement for team success.

## Two-Agent Arrival

Two-agent arrival reports whether at least two UAVs reach their goal conditions. It is a stronger partial-completion diagnostic than any-agent arrival, but it still does not imply full coordinated completion.

Together, any-agent arrival and two-agent arrival show whether a method is failing entirely, making partial progress, or approaching the strict all-agent requirement.

## Collision-Free Rate

Collision-free rate measures the fraction of episodes without collision events. It should not be interpreted alone.

A conservative or non-arriving policy can sometimes have fewer collisions while still failing the navigation task. Conversely, a policy that makes more progress may expose itself to more terrain, obstacle, or coordination risk. Collision-free rate should therefore be read together with team success, partial-arrival metrics, final goal distance, and total collision burden.

## Final Goal Distance

Final goal distance measures remaining distance to the assigned goals at the end of an episode. Lower values indicate stronger progress toward the target positions, even when strict team completion is not achieved.

This metric is useful when team success is sparse because it reveals whether a method reduces the navigation gap or merely avoids collisions without reaching useful goal configurations.

## Total Collision Burden

Total collision burden summarizes the amount of collision exposure accumulated during evaluation. It complements collision-free rate by reflecting the severity or frequency of collision events across episodes.

This metric should be interpreted as a safety burden diagnostic, not as a standalone ranking criterion.

## Dense Reward

Dense reward aggregates shaped task feedback from the evaluation environment. It can summarize progress, safety, and task-related shaping terms, but it should not be used as the only ranking signal.

Safety, progress, partial arrival, and strict team completion can trade off. A higher dense reward does not automatically imply a stronger strict team-completion profile, and a method with better completion may not dominate every reward-shaped diagnostic.

## Reading Table 6

Table 6 should be read as a multi-metric deployment diagnostic under the reported Level-2 checkpoint-FR protocol, not as a universal MARL ranking.

For example, MATD3 Sep-Grad is completion-oriented in Table 6 because it provides the strongest strict all-agent completion profile under that protocol. MATD3 Dual-Q is progress/safety-oriented in Table 6 because its dense reward, final-distance, and collision-related diagnostics emphasize complementary progress and safety behavior.

The intended interpretation is bounded: the metrics show different aspects of behavior under a matched guidance-layer benchmark, and no single metric establishes universal algorithmic superiority.
