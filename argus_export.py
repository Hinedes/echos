"""Export Echos Genesis state in ARGUS's strict trajectory format.

The raycaster origin is a simulation query point.  It is deliberately not
used as the physical emitter position in this exporter.
"""

from dataclasses import dataclass
import json

import numpy as np


PHYSICAL_EMITTER_OFFSET_BODY_M = np.array([0.0825, 0.0, 0.0])
MIC_OFFSETS_BODY_M = np.array([
    [0.0785, 0.0, 0.0215],
    [-0.0785, 0.0, -0.0215],
    [0.0, 0.0785, 0.0],
    [0.0, -0.0785, 0.0],
])
_BODY_FORWARD = np.array([1.0, 0.0, 0.0])


def _unit_quaternion(q):
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)
    if q.shape != (4,) or not np.isfinite(norm) or norm == 0:
        raise ValueError("quaternion must be a finite, non-zero wxyz vector")
    return q / norm


def _quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _quat_to_rotmat(q):
    w, x, y, z = _unit_quaternion(q)
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * x * x - 2 * z * z, 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * x * x - 2 * y * y],
    ])


@dataclass(frozen=True)
class GimbalState:
    """The authoritative sensor orientation for one recorded frame.

    Echos represents positive sensor pitch as +Z in body coordinates.  The
    corresponding active body-frame quaternion is therefore a negative Y
    rotation under the wxyz convention used by ARGUS and Echos.
    """

    pitch_rad: float
    quat_body_wxyz: np.ndarray

    @classmethod
    def from_pitch(cls, pitch_rad: float):
        if not np.isfinite(pitch_rad):
            raise ValueError("gimbal pitch must be finite")
        half = -float(pitch_rad) / 2.0
        return cls(
            pitch_rad=float(pitch_rad),
            quat_body_wxyz=np.array([np.cos(half), 0.0, np.sin(half), 0.0]),
        )

    def __post_init__(self):
        if not np.isfinite(self.pitch_rad):
            raise ValueError("gimbal pitch must be finite")
        quat = _unit_quaternion(self.quat_body_wxyz)
        half = -float(self.pitch_rad) / 2.0
        expected = np.array([np.cos(half), 0.0, np.sin(half), 0.0])
        if abs(np.dot(quat, expected)) < 1.0 - 1e-10:
            raise ValueError("gimbal quaternion does not match gimbal pitch")
        object.__setattr__(self, "quat_body_wxyz", quat)

    def derive(self, body_quat_wxyz):
        """Derive every exported orientation/geometry field from this state."""
        body_q = _unit_quaternion(body_quat_wxyz)
        gimbal_world_q = _unit_quaternion(_quat_multiply(body_q, self.quat_body_wxyz))
        beam_world = _quat_to_rotmat(gimbal_world_q) @ _BODY_FORWARD
        return body_q, gimbal_world_q, beam_world


class TrajectoryRecorder:
    """Collect synchronized Echos state and write an ARGUS NPZ contract."""

    def __init__(self, physics_dt_s=0.01, control_dt_s=0.01):
        if not np.isfinite(physics_dt_s) or physics_dt_s <= 0:
            raise ValueError("physics_dt_s must be finite and positive")
        if not np.isfinite(control_dt_s) or control_dt_s <= 0:
            raise ValueError("control_dt_s must be finite and positive")
        self.physics_dt_s = float(physics_dt_s)
        self.control_dt_s = float(control_dt_s)
        self._last_t = None
        self._frames = {key: [] for key in (
            "t_s", "body_pos_world_m", "body_quat_world_wxyz",
            "body_vel_world_mps", "body_omega_body_rps",
            "body_omega_world_rps", "gimbal_pitch_rad",
            "gimbal_quat_body_wxyz", "gimbal_quat_world_wxyz",
            "beam_axis_world", "emitter_pos_world_m", "mic_pos_world_m",
        )}

    def record(self, t_s, body_pos_world_m, body_quat_world_wxyz,
               body_vel_world_mps, body_omega_world_rps, gimbal):
        t_s = float(t_s)
        if not np.isfinite(t_s) or (self._last_t is not None and t_s <= self._last_t):
            raise ValueError("trajectory timestamps must be finite and increasing")
        pos = np.asarray(body_pos_world_m, dtype=float)
        vel = np.asarray(body_vel_world_mps, dtype=float)
        omega_world = np.asarray(body_omega_world_rps, dtype=float)
        if any(value.shape != (3,) or np.any(~np.isfinite(value))
               for value in (pos, vel, omega_world)):
            raise ValueError("body position, velocity, and angular velocity must be finite 3-vectors")
        body_q, gimbal_world_q, beam_world = gimbal.derive(body_quat_world_wxyz)
        R_body = _quat_to_rotmat(body_q)
        emitter = pos + R_body @ PHYSICAL_EMITTER_OFFSET_BODY_M
        mics = pos + (R_body @ MIC_OFFSETS_BODY_M.T).T
        self._frames["t_s"].append(t_s)
        self._frames["body_pos_world_m"].append(pos)
        self._frames["body_quat_world_wxyz"].append(body_q)
        self._frames["body_vel_world_mps"].append(vel)
        self._frames["body_omega_body_rps"].append(R_body.T @ omega_world)
        self._frames["body_omega_world_rps"].append(omega_world)
        self._frames["gimbal_pitch_rad"].append(gimbal.pitch_rad)
        self._frames["gimbal_quat_body_wxyz"].append(gimbal.quat_body_wxyz)
        self._frames["gimbal_quat_world_wxyz"].append(gimbal_world_q)
        self._frames["beam_axis_world"].append(beam_world)
        self._frames["emitter_pos_world_m"].append(emitter)
        self._frames["mic_pos_world_m"].append(mics)
        self._last_t = t_s

    def arrays(self):
        if len(self._frames["t_s"]) < 2:
            raise ValueError("trajectory needs at least two frames")
        return {key: np.asarray(value, dtype=float) for key, value in self._frames.items()}

    def save(self, path):
        arrays = self.arrays()
        metadata = {
            "schema": "echos-argus-trajectory",
            "version": 1,
            "physics_dt_s": self.physics_dt_s,
            "control_dt_s": self.control_dt_s,
            "world_handedness": "right",
            "world_up_axis": "+Z",
            "body_forward_axis": "+X",
            "body_positive_y_semantics": "left",
            "quaternion_order": "wxyz",
            "quaternion_semantics": "active_body_to_world",
            "position_unit": "m",
            "time_unit": "s",
            "angular_unit": "rad",
            "physical_emitter_offset_body_m": PHYSICAL_EMITTER_OFFSET_BODY_M.tolist(),
        }
        np.savez_compressed(path, **arrays, metadata_json=json.dumps(metadata))
        return path
