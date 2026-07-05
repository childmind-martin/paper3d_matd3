#!/usr/bin/env python3
"""Consistency and live metrics for the Gazebo APF backend."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from gazebo_apf import GazeboAPFResult
from gazebo_apf_state_provider import GazeboAPFSceneState, GazeboAPFStateProvider


def _to_float_list(value: Any) -> List[float]:
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        return [float(v) for v in arr.tolist()]
    except Exception:
        return []


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        value = float(value)
        if np.isfinite(value):
            return value
    except Exception:
        pass
    return default


def _vec3_or_none(value: Any) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.size >= 3 and np.all(np.isfinite(arr[:3])):
            return arr[:3].astype(np.float64, copy=True)
    except Exception:
        pass
    return None


def _direction_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size < 3 or b.size < 3:
        return 0.0
    a = a[:3]
    b = b[:3]
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-9:
        return 1.0 if np.linalg.norm(a - b) <= 1e-9 else 0.0
    return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))


class GazeboAPFValidator:
    def __init__(
        self,
        output_dir: Any = "results/gazebo_apf_validation",
        mismatch_threshold: float = 0.05,
        direction_threshold: float = 0.995,
        max_cases: int = 50,
    ) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mismatch_threshold = float(mismatch_threshold)
        self.direction_threshold = float(direction_threshold)
        self.max_cases = max(1, int(max_cases))
        self.metrics_path = self.output_dir / "apf_consistency_metrics.json"
        self.cases_path = self.output_dir / "apf_mismatch_cases.json"
        self.live_csv_path = self.output_dir / "gazebo_apf_live_metrics.csv"
        self.adapter_metrics_path = self.output_dir / "gazebo_apf_adapter_metrics.json"
        self.case_plot_dir = self.output_dir / "mismatch_visualizations"
        self.case_plot_dir.mkdir(parents=True, exist_ok=True)

        self.count = 0
        self.sum_corrected_error = 0.0
        self.max_corrected_error = 0.0
        self.sum_pf_error = 0.0
        self.max_pf_error = 0.0
        self.sum_direction_cosine = 0.0
        self.min_direction_cosine = 1.0
        self.sum_force_norm_error = 0.0
        self.max_force_norm_error = 0.0
        self.total_mismatch_count = 0
        self.mismatch_cases: List[Dict[str, Any]] = []
        self._live_header_written = self.live_csv_path.exists() and self.live_csv_path.stat().st_size > 0

    def compare(
        self,
        original_corrected_actions: Any,
        original_pf_forces: Any,
        gazebo_result: GazeboAPFResult,
        state: GazeboAPFSceneState,
        state_provider: GazeboAPFStateProvider,
        raw_actions: Any,
        original_observations: Optional[Any] = None,
        gazebo_observations: Optional[Any] = None,
        episode: Optional[int] = None,
        step: Optional[int] = None,
        seed: Optional[int] = None,
        force_ratio: Optional[float] = None,
    ) -> Dict[str, Any]:
        original_corr = np.asarray(original_corrected_actions, dtype=np.float64)
        if original_corr.ndim == 1:
            original_corr = original_corr.reshape(1, -1)
        original_pf = np.asarray(original_pf_forces, dtype=np.float64)
        if original_pf.ndim == 1:
            original_pf = original_pf.reshape(1, -1)
        gazebo_corr = np.asarray(gazebo_result.corrected_actions, dtype=np.float64)
        gazebo_pf = np.asarray(gazebo_result.pf_forces, dtype=np.float64)
        raw = np.asarray(raw_actions, dtype=np.float64)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        n = min(original_corr.shape[0], gazebo_corr.shape[0], original_pf.shape[0], gazebo_pf.shape[0], len(state.agents))
        step_records = []
        for idx in range(n):
            orig_head = original_corr[idx, :3]
            gz_head = gazebo_corr[idx, :3]
            orig_pf = original_pf[idx, :3]
            gz_pf = gazebo_pf[idx, :3]
            corr_err = float(np.linalg.norm(orig_head - gz_head))
            pf_err = float(np.linalg.norm(orig_pf - gz_pf))
            cos = _direction_cosine(orig_pf, gz_pf)
            force_norm_err = abs(float(np.linalg.norm(orig_pf)) - float(np.linalg.norm(gz_pf)))
            mismatch_score = self._mismatch_score(corr_err, pf_err, cos, force_norm_err)
            self.count += 1
            self.sum_corrected_error += corr_err
            self.max_corrected_error = max(self.max_corrected_error, corr_err)
            self.sum_pf_error += pf_err
            self.max_pf_error = max(self.max_pf_error, pf_err)
            self.sum_direction_cosine += cos
            self.min_direction_cosine = min(self.min_direction_cosine, cos)
            self.sum_force_norm_error += force_norm_err
            self.max_force_norm_error = max(self.max_force_norm_error, force_norm_err)

            reason = self._classify_reason(corr_err, pf_err, cos, force_norm_err)
            is_mismatch = (
                corr_err > self.mismatch_threshold
                or pf_err > self.mismatch_threshold
                or cos < self.direction_threshold
            )
            record = {
                "episode": int(episode) if episode is not None else None,
                "step": int(step) if step is not None else None,
                "seed": int(seed) if seed is not None else state.terrain_seed,
                "agent_id": int(idx),
                "force_ratio": _safe_float(force_ratio),
                "raw_action": _to_float_list(raw[idx] if idx < raw.shape[0] else []),
                "original_corrected_action": _to_float_list(original_corr[idx]),
                "original_corrected_action_head": _to_float_list(orig_head),
                "original_pf_force": _to_float_list(orig_pf),
                "original_pf_norm": float(np.linalg.norm(orig_pf)),
                "gazebo_corrected_action": _to_float_list(gazebo_corr[idx]),
                "gazebo_corrected_action_head": _to_float_list(gz_head),
                "gazebo_pf_force": _to_float_list(gz_pf),
                "gazebo_pf_norm": float(np.linalg.norm(gz_pf)),
                "corrected_action_error": corr_err,
                "pf_force_error": pf_err,
                "direction_cosine_similarity": cos,
                "force_norm_error": force_norm_err,
                "mismatch_score": mismatch_score,
                "mismatch": bool(is_mismatch),
                "difference_reason": reason,
            }
            step_records.append(record)
            if is_mismatch:
                self.total_mismatch_count += 1
                case = self._build_case(
                    base_record=record,
                    state=state,
                    state_provider=state_provider,
                    raw_action=raw[idx] if idx < raw.shape[0] else None,
                    original_corrected=orig_head,
                    original_pf=orig_pf,
                    gazebo_debug=gazebo_result.debug[idx] if idx < len(gazebo_result.debug) else {},
                    original_observation=original_observations[idx] if original_observations is not None and idx < len(original_observations) else None,
                    gazebo_observation=gazebo_observations[idx] if gazebo_observations is not None and idx < len(gazebo_observations) else None,
                )
                inserted = False
                if len(self.mismatch_cases) < self.max_cases:
                    self.mismatch_cases.append(case)
                    inserted = True
                else:
                    min_idx = min(
                        range(len(self.mismatch_cases)),
                        key=lambda i: float(self.mismatch_cases[i].get("mismatch_score", 0.0) or 0.0),
                    )
                    if mismatch_score > float(self.mismatch_cases[min_idx].get("mismatch_score", 0.0) or 0.0):
                        self.mismatch_cases[min_idx] = case
                        inserted = True
                if inserted:
                    self.mismatch_cases.sort(
                        key=lambda item: float(item.get("mismatch_score", 0.0) or 0.0),
                        reverse=True,
                    )
                    self._write_case_plot(case)
        self.flush()
        return {
            "records": step_records,
            "metrics": self.current_metrics(),
        }

    def record_live_step(
        self,
        gazebo_result: GazeboAPFResult,
        state: GazeboAPFSceneState,
        episode: Optional[int],
        step: Optional[int],
        comparison_records: Optional[Sequence[Dict[str, Any]]] = None,
        nominal_cmd_vel: Optional[Any] = None,
        sent_cmd_vel: Optional[Any] = None,
        feedback_velocities: Optional[Any] = None,
        safety_filter_records: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        comp_by_agent = {}
        for item in comparison_records or []:
            try:
                comp_by_agent[int(item.get("agent_id"))] = item
            except Exception:
                continue
        safety_by_agent = {}
        for item in safety_filter_records or []:
            try:
                safety_by_agent[int(item.get("agent_id"))] = item
            except Exception:
                continue
        nominal_cmd = None
        if nominal_cmd_vel is not None:
            try:
                nominal_cmd = np.asarray(nominal_cmd_vel, dtype=np.float64)
            except Exception:
                nominal_cmd = None
        sent_cmd = None
        if sent_cmd_vel is not None:
            try:
                sent_cmd = np.asarray(sent_cmd_vel, dtype=np.float64)
            except Exception:
                sent_cmd = None
        feedback_vel = None
        if feedback_velocities is not None:
            try:
                feedback_vel = np.asarray(feedback_velocities, dtype=np.float64)
            except Exception:
                feedback_vel = None
        def _component(values: Any, index: int) -> Any:
            try:
                arr = np.asarray(values, dtype=np.float64).reshape(-1)
                if index < arr.size and np.isfinite(arr[index]):
                    return float(arr[index])
            except Exception:
                pass
            return ""
        rows = []
        for idx, debug in enumerate(gazebo_result.debug):
            comp = comp_by_agent.get(idx, {})
            safety = safety_by_agent.get(idx, {})
            nearest = debug.get("nearest_obstacle") or {}
            predicted_cmd = debug.get("cmd_vel", [None, None, None])
            nominal_actual_cmd = (
                nominal_cmd[idx].tolist()
                if nominal_cmd is not None and idx < nominal_cmd.shape[0]
                else predicted_cmd
            )
            actual_cmd = sent_cmd[idx].tolist() if sent_cmd is not None and idx < sent_cmd.shape[0] else predicted_cmd
            feedback_cmd = (
                feedback_vel[idx].tolist()
                if feedback_vel is not None and idx < feedback_vel.shape[0]
                else [None, None, None]
            )
            predicted_arr = _vec3_or_none(predicted_cmd)
            actual_arr = _vec3_or_none(actual_cmd)
            feedback_arr = _vec3_or_none(feedback_cmd)
            prediction_error = (
                float(np.linalg.norm(predicted_arr - actual_arr))
                if predicted_arr is not None and actual_arr is not None
                else ""
            )
            feedback_error = (
                float(np.linalg.norm(actual_arr - feedback_arr))
                if actual_arr is not None and feedback_arr is not None
                else ""
            )
            feedback_cosine = (
                _direction_cosine(actual_arr, feedback_arr)
                if actual_arr is not None and feedback_arr is not None
                else ""
            )
            feedback_speed_ratio = ""
            if actual_arr is not None and feedback_arr is not None:
                actual_norm = float(np.linalg.norm(actual_arr))
                feedback_norm = float(np.linalg.norm(feedback_arr))
                feedback_speed_ratio = float(feedback_norm / max(actual_norm, 1e-9))
            original_corr = comp.get("original_corrected_action") or comp.get("original_corrected_action_head") or []
            original_pf = comp.get("original_pf_force") or []
            gazebo_corr = comp.get("gazebo_corrected_action") or debug.get("corrected_action", [])
            gazebo_pf = comp.get("gazebo_pf_force") or debug.get("pf_force_action", [])
            rows.append(
                {
                    "episode": int(episode) if episode is not None else "",
                    "step": int(step) if step is not None else "",
                    "agent_id": idx,
                    "frame": state.frame if state.frame is not None else "",
                    "source": state.source,
                    "contact": int(bool(debug.get("contact", False))),
                    "pose_x": debug.get("pose", [None, None, None])[0],
                    "pose_y": debug.get("pose", [None, None, None])[1],
                    "pose_z": debug.get("pose", [None, None, None])[2],
                    "vel_x": debug.get("velocity", [None, None, None])[0],
                    "vel_y": debug.get("velocity", [None, None, None])[1],
                    "vel_z": debug.get("velocity", [None, None, None])[2],
                    "raw_ax": gazebo_result.raw_actions[idx, 0] if idx < gazebo_result.raw_actions.shape[0] else "",
                    "raw_ay": gazebo_result.raw_actions[idx, 1] if idx < gazebo_result.raw_actions.shape[0] else "",
                    "raw_az": gazebo_result.raw_actions[idx, 2] if idx < gazebo_result.raw_actions.shape[0] else "",
                    "python_corr_ax": _component(original_corr, 0),
                    "python_corr_ay": _component(original_corr, 1),
                    "python_corr_az": _component(original_corr, 2),
                    "python_pf_x": _component(original_pf, 0),
                    "python_pf_y": _component(original_pf, 1),
                    "python_pf_z": _component(original_pf, 2),
                    "python_pf_norm": _safe_float(comp.get("original_pf_norm"), ""),
                    "gazebo_corr_ax": _component(gazebo_corr, 0),
                    "gazebo_corr_ay": _component(gazebo_corr, 1),
                    "gazebo_corr_az": _component(gazebo_corr, 2),
                    "gazebo_pf_x": _component(gazebo_pf, 0),
                    "gazebo_pf_y": _component(gazebo_pf, 1),
                    "gazebo_pf_z": _component(gazebo_pf, 2),
                    "gazebo_pf_norm": _safe_float(comp.get("gazebo_pf_norm"), ""),
                    "corr_ax": debug.get("corrected_action", [None, None, None])[0],
                    "corr_ay": debug.get("corrected_action", [None, None, None])[1],
                    "corr_az": debug.get("corrected_action", [None, None, None])[2],
                    "nominal_cmd_vx": nominal_actual_cmd[0],
                    "nominal_cmd_vy": nominal_actual_cmd[1],
                    "nominal_cmd_vz": nominal_actual_cmd[2],
                    "cmd_vx": actual_cmd[0],
                    "cmd_vy": actual_cmd[1],
                    "cmd_vz": actual_cmd[2],
                    "filter_mode": safety.get("mode", ""),
                    "filter_enabled": int(bool(safety.get("enabled", False))) if safety else "",
                    "filter_active": int(bool(safety.get("filter_active", False))) if safety else "",
                    "filter_trigger_reason": safety.get("filter_trigger_reason", ""),
                    "filter_nearest_obstacle": safety.get("nearest_obstacle_id", ""),
                    "filter_surface_distance": safety.get("surface_distance", ""),
                    "filter_clearance": safety.get("clearance", ""),
                    "filter_agent_radius": safety.get("agent_radius", ""),
                    "filter_stopping_distance": safety.get("stopping_distance", ""),
                    "filter_safety_margin": safety.get("safety_margin", ""),
                    "filter_stopping_margin": safety.get("stopping_margin", ""),
                    "filter_inward_velocity_before": safety.get("inward_velocity_before_filter", ""),
                    "filter_inward_velocity_after": safety.get("inward_velocity_after_filter", ""),
                    "filter_relative_inward_velocity_before": safety.get("relative_inward_velocity_before_filter", ""),
                    "filter_relative_inward_velocity_after": safety.get("relative_inward_velocity_after_filter", ""),
                    "filter_current_relative_inward_velocity": safety.get("current_relative_inward_velocity", ""),
                    "filter_closing_inward_velocity_for_stopping": safety.get("closing_inward_velocity_for_stopping", ""),
                    "filter_tangential_speed_before": safety.get("tangential_speed_before_filter", ""),
                    "filter_tangential_speed_after": safety.get("tangential_speed_after_filter", ""),
                    "filter_goal_distance": safety.get("goal_distance", ""),
                    "filter_goal_projection_before": safety.get("goal_projection_before_filter", ""),
                    "filter_goal_projection_after": safety.get("goal_projection_after_filter", ""),
                    "filter_tangential_velocity_kept_ratio": safety.get("tangential_velocity_kept_ratio", ""),
                    "filter_outward_velocity_added": safety.get("outward_velocity_added", ""),
                    "filter_invasiveness": safety.get("filter_invasiveness", ""),
                    "filter_boundary_dwell_steps": safety.get("boundary_dwell_steps", ""),
                    "filter_line_to_goal_blocked": int(bool(safety.get("line_to_goal_blocked", False))) if safety else "",
                    "filter_nearest_agent_distance": safety.get("nearest_agent_distance", ""),
                    "filter_formation_error": safety.get("formation_error", ""),
                    "filter_avoidance_state": safety.get("avoidance_state", ""),
                    "filter_line_to_goal_clearance": safety.get("line_to_goal_clearance", ""),
                    "filter_halfspace_lower_bound": safety.get("halfspace_lower_bound", ""),
                    "filter_normal_velocity_before_projection": safety.get("normal_velocity_before_projection", ""),
                    "filter_normal_velocity_after_projection": safety.get("normal_velocity_after_projection", ""),
                    "filter_halfspace_projection_delta_norm": safety.get("halfspace_projection_delta_norm", ""),
                    "filter_tangent_recovery_applied": int(bool(safety.get("tangent_recovery_applied", False))) if safety else "",
                    "filter_goal_projection_recovery_applied": int(bool(safety.get("goal_projection_recovery_applied", False))) if safety else "",
                    "filter_outward_speed_applied": safety.get("outward_speed_applied", ""),
                    "filter_allowed_inward_velocity": safety.get("allowed_inward_velocity", ""),
                    "filter_cmd_delta_norm": safety.get("cmd_delta_norm", ""),
                    "predicted_cmd_vx": predicted_cmd[0],
                    "predicted_cmd_vy": predicted_cmd[1],
                    "predicted_cmd_vz": predicted_cmd[2],
                    "feedback_vx": feedback_cmd[0],
                    "feedback_vy": feedback_cmd[1],
                    "feedback_vz": feedback_cmd[2],
                    "predicted_cmd_error": prediction_error,
                    "cmd_feedback_error": feedback_error,
                    "cmd_feedback_cosine": feedback_cosine,
                    "cmd_feedback_speed_ratio": feedback_speed_ratio,
                    "pf_norm": float(np.linalg.norm(gazebo_result.pf_forces[idx, :3])),
                    "force_norm": debug.get("force_norm"),
                    "terrain_clearance": debug.get("terrain_clearance"),
                    "nearest_obstacle": nearest.get("name"),
                    "nearest_obstacle_surface_distance": nearest.get("surface_distance"),
                    "nearest_obstacle_clearance": nearest.get("clearance"),
                    "corrected_action_error": comp.get("corrected_action_error", ""),
                    "pf_force_error": comp.get("pf_force_error", ""),
                    "direction_cosine_similarity": comp.get("direction_cosine_similarity", ""),
                    "difference_reason": comp.get("difference_reason", ""),
                }
            )
        self._append_live_rows(rows)

    def current_metrics(self) -> Dict[str, Any]:
        count = max(self.count, 1)
        return {
            "sample_count": int(self.count),
            "mean_error": float(self.sum_corrected_error / count),
            "max_error": float(self.max_corrected_error),
            "mean_pf_force_error": float(self.sum_pf_error / count),
            "max_pf_force_error": float(self.max_pf_error),
            "direction_cosine_similarity": float(self.sum_direction_cosine / count),
            "min_direction_cosine_similarity": float(self.min_direction_cosine if self.count else 1.0),
            "mean_force_norm_error": float(self.sum_force_norm_error / count),
            "max_force_norm_error": float(self.max_force_norm_error),
            "mismatch_count": int(self.total_mismatch_count),
            "stored_mismatch_case_count": int(len(self.mismatch_cases)),
            "mismatch_threshold": float(self.mismatch_threshold),
            "direction_threshold": float(self.direction_threshold),
            "output_dir": str(self.output_dir),
        }

    def flush(self) -> None:
        self.metrics_path.write_text(json.dumps(self.current_metrics(), ensure_ascii=False, indent=2), encoding="utf-8")
        self.cases_path.write_text(json.dumps(self.mismatch_cases, ensure_ascii=False, indent=2), encoding="utf-8")

    def finalize(self, first_contact_step: Optional[int] = None) -> Dict[str, Any]:
        adapter_metrics = self._write_adapter_metrics(first_contact_step=first_contact_step)
        metrics = self.current_metrics()
        metrics["adapter_metrics_path"] = str(self.adapter_metrics_path)
        metrics["adapter_metrics"] = adapter_metrics
        self.metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        self.cases_path.write_text(json.dumps(self.mismatch_cases, ensure_ascii=False, indent=2), encoding="utf-8")
        return metrics

    def _write_adapter_metrics(self, first_contact_step: Optional[int] = None) -> Dict[str, Any]:
        rows = self._read_live_rows()
        metrics = self._adapter_metrics_from_rows(rows, first_contact_step=first_contact_step)
        self.adapter_metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        return metrics

    def _read_live_rows(self) -> List[Dict[str, Any]]:
        if not self.live_csv_path.exists() or self.live_csv_path.stat().st_size <= 0:
            return []
        try:
            with self.live_csv_path.open("r", newline="", encoding="utf-8") as f:
                return [dict(row) for row in csv.DictReader(f)]
        except Exception:
            return []

    def _adapter_metrics_from_rows(
        self,
        rows: Sequence[Dict[str, Any]],
        first_contact_step: Optional[int] = None,
    ) -> Dict[str, Any]:
        thresholds = self._adapter_speed_thresholds()
        try:
            main_threshold = float(os.getenv("GAZEBO_APF_ADAPTER_MAIN_SPEED_THRESHOLD", "0.1"))
        except Exception:
            main_threshold = 0.1
        if main_threshold not in thresholds:
            thresholds.append(main_threshold)
            thresholds = sorted(set(float(v) for v in thresholds))

        pre_rows: List[Dict[str, Any]] = []
        post_rows: List[Dict[str, Any]] = []
        for row in rows:
            step = self._row_float(row, "step")
            if first_contact_step is None or step is None or step < int(first_contact_step):
                pre_rows.append(row)
            else:
                post_rows.append(row)

        pre = self._adapter_split_metrics(pre_rows, thresholds)
        post = self._adapter_split_metrics(post_rows, thresholds)
        main_key = self._threshold_key(main_threshold)
        main_report = pre.get(main_key, {})
        return {
            "first_contact_step": int(first_contact_step) if first_contact_step is not None else None,
            "pre_contact_definition": "step < first_contact_step",
            "post_contact_definition": "step >= first_contact_step",
            "main_speed_threshold": float(main_threshold),
            "main_speed_threshold_key": main_key,
            "main_report_uses_pre_contact_only": True,
            "main_report": main_report,
            "pre_contact": pre,
            "post_contact": post,
            "row_count": int(len(rows)),
            "pre_contact_row_count": int(len(pre_rows)),
            "post_contact_row_count": int(len(post_rows)),
        }

    def _adapter_speed_thresholds(self) -> List[float]:
        raw = os.getenv("GAZEBO_APF_ADAPTER_SPEED_THRESHOLDS", "0,0.05,0.1,0.2,0.5,1.0")
        values: List[float] = []
        for part in str(raw).split(","):
            try:
                value = float(part.strip())
            except Exception:
                continue
            if value >= 0.0 and np.isfinite(value):
                values.append(float(value))
        if not values:
            values = [0.0, 0.1, 0.5]
        return sorted(set(values))

    @staticmethod
    def _threshold_key(threshold: float) -> str:
        return f"cmd_norm_gt_{float(threshold):g}"

    @staticmethod
    def _row_float(row: Dict[str, Any], key: str) -> Optional[float]:
        try:
            value = float(row.get(key, ""))
            if np.isfinite(value):
                return value
        except Exception:
            pass
        return None

    def _adapter_split_metrics(
        self,
        rows: Sequence[Dict[str, Any]],
        thresholds: Sequence[float],
    ) -> Dict[str, Any]:
        return {
            self._threshold_key(threshold): self._adapter_threshold_metrics(rows, float(threshold))
            for threshold in thresholds
        }

    def _adapter_threshold_metrics(self, rows: Sequence[Dict[str, Any]], threshold: float) -> Dict[str, Any]:
        errors: List[float] = []
        cosines: List[float] = []
        speed_ratios: List[float] = []
        cmd_norms: List[float] = []
        feedback_norms: List[float] = []
        axis_errors = {"x": [], "y": [], "z": []}
        axis_sign_matches = {"x": [], "y": [], "z": []}
        for row in rows:
            cmd = np.asarray(
                [
                    self._row_float(row, "cmd_vx"),
                    self._row_float(row, "cmd_vy"),
                    self._row_float(row, "cmd_vz"),
                ],
                dtype=object,
            )
            feedback = np.asarray(
                [
                    self._row_float(row, "feedback_vx"),
                    self._row_float(row, "feedback_vy"),
                    self._row_float(row, "feedback_vz"),
                ],
                dtype=object,
            )
            if any(v is None for v in cmd) or any(v is None for v in feedback):
                continue
            cmd_arr = np.asarray(cmd, dtype=np.float64)
            feedback_arr = np.asarray(feedback, dtype=np.float64)
            cmd_norm = float(np.linalg.norm(cmd_arr))
            if not np.isfinite(cmd_norm) or cmd_norm <= float(threshold):
                continue
            feedback_norm = float(np.linalg.norm(feedback_arr))
            error = self._row_float(row, "cmd_feedback_error")
            cosine = self._row_float(row, "cmd_feedback_cosine")
            speed_ratio = self._row_float(row, "cmd_feedback_speed_ratio")
            if error is None:
                error = float(np.linalg.norm(cmd_arr - feedback_arr))
            errors.append(float(error))
            cmd_norms.append(cmd_norm)
            feedback_norms.append(feedback_norm)
            if cosine is not None:
                cosines.append(float(cosine))
            if speed_ratio is not None:
                speed_ratios.append(float(speed_ratio))
            for axis, axis_idx in (("x", 0), ("y", 1), ("z", 2)):
                c = float(cmd_arr[axis_idx])
                f = float(feedback_arr[axis_idx])
                axis_errors[axis].append(abs(c - f))
                if abs(c) <= 1e-9 and abs(f) <= 1e-9:
                    axis_sign_matches[axis].append(1.0)
                elif abs(c) <= 1e-9 or abs(f) <= 1e-9:
                    axis_sign_matches[axis].append(0.0)
                else:
                    axis_sign_matches[axis].append(1.0 if np.sign(c) == np.sign(f) else 0.0)

        def _mean(values: Sequence[float]) -> Optional[float]:
            return float(np.mean(values)) if values else None

        def _max(values: Sequence[float]) -> Optional[float]:
            return float(np.max(values)) if values else None

        def _min(values: Sequence[float]) -> Optional[float]:
            return float(np.min(values)) if values else None

        def _p95(values: Sequence[float]) -> Optional[float]:
            return float(np.percentile(values, 95)) if values else None

        return {
            "speed_threshold": float(threshold),
            "sample_count": int(len(errors)),
            "mean_error": _mean(errors),
            "p95_error": _p95(errors),
            "max_error": _max(errors),
            "mean_cosine": _mean(cosines),
            "min_cosine": _min(cosines),
            "mean_speed_ratio": _mean(speed_ratios),
            "mean_cmd_norm": _mean(cmd_norms),
            "mean_feedback_norm": _mean(feedback_norms),
            "axis_mae": {axis: _mean(values) for axis, values in axis_errors.items()},
            "axis_sign_match_rate": {
                axis: _mean(values) for axis, values in axis_sign_matches.items()
            },
        }

    def _classify_reason(self, corr_err: float, pf_err: float, cos: float, force_norm_err: float) -> str:
        reasons = []
        if corr_err > self.mismatch_threshold:
            reasons.append("corrected_action_delta")
        if pf_err > self.mismatch_threshold:
            reasons.append("pf_vector_delta")
        if cos < self.direction_threshold:
            reasons.append("force_direction_delta")
        if force_norm_err > self.mismatch_threshold:
            reasons.append("force_norm_delta")
        return ",".join(reasons) if reasons else "within_threshold"

    def _mismatch_score(self, corr_err: float, pf_err: float, cos: float, force_norm_err: float) -> float:
        direction_gap = max(0.0, self.direction_threshold - float(cos))
        return float(max(corr_err, pf_err, force_norm_err, direction_gap))

    def _build_case(
        self,
        base_record: Dict[str, Any],
        state: GazeboAPFSceneState,
        state_provider: GazeboAPFStateProvider,
        raw_action: Any,
        original_corrected: Any,
        original_pf: Any,
        gazebo_debug: Dict[str, Any],
        original_observation: Any,
        gazebo_observation: Any,
    ) -> Dict[str, Any]:
        idx = int(base_record["agent_id"])
        agent = state.agents[idx]
        nearest = state_provider.nearest_obstacle(state, idx)
        return {
            **base_record,
            "pose": agent.position.astype(float).tolist(),
            "velocity": agent.velocity.astype(float).tolist(),
            "goal": state.goals[idx].astype(float).tolist(),
            "nearest_obstacle": nearest,
            "terrain_clearance": state_provider.terrain_clearance(state, idx),
            "raw_action": _to_float_list(raw_action),
            "original_apf_output": {
                "corrected_action_head": _to_float_list(original_corrected),
                "pf_force": _to_float_list(original_pf),
                "force_norm": float(np.linalg.norm(np.asarray(original_pf, dtype=np.float64).reshape(-1)[:3])),
            },
            "gazebo_apf_output": {
                "corrected_action": gazebo_debug.get("corrected_action"),
                "corrected_acceleration": gazebo_debug.get("corrected_acceleration"),
                "cmd_vel": gazebo_debug.get("cmd_vel"),
                "pf_force": gazebo_debug.get("pf_force_action"),
                "force_norm": gazebo_debug.get("force_norm"),
                "force_direction": gazebo_debug.get("force_direction"),
                "goal_force": gazebo_debug.get("goal_force"),
                "terrain_force": gazebo_debug.get("terrain_force"),
                "obstacle_force": gazebo_debug.get("obstacle_force"),
                "agent_agent_repulsion": gazebo_debug.get("agent_agent_repulsion"),
            },
            "original_observation_head": _to_float_list(original_observation)[:81],
            "gazebo_observation_head": _to_float_list(gazebo_observation)[:81],
        }

    def _write_case_plot(self, case: Dict[str, Any]) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.patches import Circle
        except Exception:
            return
        try:
            idx = len(self.mismatch_cases)
            path = self.case_plot_dir / f"mismatch_case_{idx:03d}_ep{case.get('episode')}_step{case.get('step')}_agent{case.get('agent_id')}.png"
            pose = np.asarray(case.get("pose", [0.0, 0.0, 0.0]), dtype=np.float64)
            goal = np.asarray(case.get("goal", [0.0, 0.0, 0.0]), dtype=np.float64)
            orig_pf = np.asarray(case.get("original_apf_output", {}).get("pf_force", [0.0, 0.0, 0.0]), dtype=np.float64)
            gz_pf = np.asarray(case.get("gazebo_apf_output", {}).get("pf_force", [0.0, 0.0, 0.0]), dtype=np.float64)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter([pose[0]], [pose[1]], c="black", label="agent")
            ax.scatter([goal[0]], [goal[1]], c="green", marker="*", s=120, label="goal")
            nearest = case.get("nearest_obstacle") or {}
            if nearest.get("center") is not None:
                center = np.asarray(nearest.get("center"), dtype=np.float64)
                radius = float(nearest.get("radius", 0.0) or 0.0)
                ax.add_patch(Circle((center[0], center[1]), radius, fill=False, color="red", linewidth=1.5, label="nearest obstacle"))
                ax.scatter([center[0]], [center[1]], c="red", s=20)
            scale = 10.0
            ax.arrow(pose[0], pose[1], orig_pf[0] * scale, orig_pf[1] * scale, color="blue", width=0.25, length_includes_head=True, label="original pf")
            ax.arrow(pose[0], pose[1], gz_pf[0] * scale, gz_pf[1] * scale, color="orange", width=0.25, length_includes_head=True, label="gazebo pf")
            ax.set_title(f"APF mismatch ep={case.get('episode')} step={case.get('step')} agent={case.get('agent_id')}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.axis("equal")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(path, dpi=150)
            plt.close(fig)
            case["visualization"] = str(path)
        except Exception:
            return

    def _append_live_rows(self, rows: Sequence[Dict[str, Any]]) -> None:
        if not rows:
            return
        fieldnames = list(rows[0].keys())
        with self.live_csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not self._live_header_written:
                writer.writeheader()
                self._live_header_written = True
            for row in rows:
                writer.writerow(row)
