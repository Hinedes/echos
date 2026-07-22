"""Replay an exported Echos trajectory through ARGUS's motion-aware chain.

Run with ARGUS importable, for example:

    PYTHONPATH=/root/ARGUS python replay_argus.py trajectory.npz
"""

import sys

import numpy as np

from argus.acoustic_scene import Reflector
from argus.detect import coherent_onset, median_tof
from argus.motion import ECHOS_MIC_OFFSETS, SampledTrajectory
from argus.schedule import build_schedule
from argus.solve import solve_point
from argus.synthesize import synthesize


def replay(path, speed=343.0, target=None):
    trajectory = SampledTrajectory(path, strict_contract=True)
    schedule = build_schedule([40_000, 42_000, 44_000, 46_000, 48_000], 0.004, 250_000)
    if target is None:
        origin0 = trajectory.emitter_pos(0.0)
        target = origin0 + trajectory.beam_axis(0.0) * 2.5
    target = np.asarray(target, dtype=float)
    waveforms = synthesize(
        schedule, [Reflector(target)], ECHOS_MIC_OFFSETS, speed,
        coherent=True, trajectory=trajectory,
    )
    arrivals = coherent_onset(waveforms, schedule)
    estimates = []
    for column, segment in enumerate(schedule.segments):
        t_emit = segment.start
        mic_positions = np.array([
            trajectory.mic_pos(t_emit + arrivals[i, column], i)
            for i in range(len(ECHOS_MIC_OFFSETS))
        ])
        estimates.append(solve_point(
            arrivals[:, column], mic_positions, speed,
            origin=trajectory.emitter_pos(t_emit),
            beam_axis=trajectory.beam_axis(t_emit),
        ))
    estimate = np.median(estimates, axis=0)
    error = float(np.linalg.norm(estimate - target))
    if not np.isfinite(error) or error >= 0.1:
        raise RuntimeError(f"ARGUS replay error {error:.4f} m exceeds 0.1 m")
    print(f"ARGUS_REPLAY_PASS error_m={error:.6f} target={target.tolist()}")
    return estimate, target


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python replay_argus.py trajectory.npz")
    replay(sys.argv[1])
