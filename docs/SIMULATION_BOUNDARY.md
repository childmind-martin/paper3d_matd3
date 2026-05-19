# Simulation Boundary

This document states the simulation boundary for the reviewer artifact. It mirrors the claim boundary used in the paper's guidance-layer simulation section.

## Benchmark Scope

The main benchmark is **guidance-layer cooperative UAV navigation**. It studies how multi-agent learning behaves when UAV policies produce commands that are corrected by an artificial potential field (APF) before environment execution.

The main reported setting uses APF-corrected translational guidance commands. These commands drive the navigation update used for terrain-aware, obstacle-aware, and formation-level cooperative behavior.

## Finite-Response Surrogate

The implementation includes a first-order attitude-response lag as a compact finite-response surrogate. This lag represents the fact that a desired acceleration or thrust-direction command cannot be realized instantaneously by the lower-level attitude/thrust-direction loop.

This makes the execution interface more restrictive than an instantaneous point-mass update, while keeping the benchmark focused on guidance-layer navigation and learning-interface effects.

## Not a Full Flight-Control Benchmark

The main benchmark is not a full six-degree-of-freedom quadrotor flight-control benchmark. It does not evaluate rotor-level thrust allocation, full attitude dynamics, aerodynamic effects, sensing latency, onboard state estimation, or hardware-in-the-loop flight control.

The reported results should therefore be read as guidance-layer cooperative navigation evidence, not as high-fidelity flight-control validation.

## Gravity Setting

Gravity is disabled in the reported guidance-layer benchmark. This setting is intentional: the paper studies execution-corrected cooperative navigation and raw/corrected action semantics, not low-level thrust balancing, attitude stabilization, or gravity compensation.

The simulation still applies the same guidance-layer dynamics mode and parameter set across compared methods.

Parser defaults in individual training scripts are not the authoritative paper
configuration. The reported batches should be interpreted through the saved
run configuration, manifest entries, environment overrides, and paper protocol;
for the reported guidance-layer benchmark, gravity is disabled.

## Fairness Within the Benchmark

All compared methods share the same simulation mode, terrain/obstacle protocol, dynamics parameters, and evaluation controls within each reported batch. Therefore, comparisons are fair within the reported guidance-layer benchmark.

The conclusions are restricted to learning-interface effects under matched guidance-layer simulation.
