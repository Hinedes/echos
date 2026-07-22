import numpy as np

from argus_export import (
    GimbalState, MIC_OFFSETS_BODY_M, PHYSICAL_EMITTER_OFFSET_BODY_M,
    TrajectoryRecorder, _quat_to_rotmat,
)


def test_one_gimbal_state_drives_exported_transforms(tmp_path):
    recorder = TrajectoryRecorder()
    body_q = np.array([1.0, 0.0, 0.0, 0.0])
    state = GimbalState.from_pitch(np.radians(15.0))
    recorder.record(0.0, [1.0, 2.0, 3.0], body_q, [0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6], state)
    recorder.record(0.01, [1.0, 2.0, 3.0], body_q, [0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6], state)
    arrays = recorder.arrays()
    assert arrays["gimbal_pitch_rad"][0] == state.pitch_rad
    assert np.allclose(
        arrays["beam_axis_world"][0],
        _quat_to_rotmat(arrays["gimbal_quat_world_wxyz"][0]) @ [1.0, 0.0, 0.0],
    )
    assert np.allclose(
        arrays["emitter_pos_world_m"][0], [1.0, 2.0, 3.0] + PHYSICAL_EMITTER_OFFSET_BODY_M,
    )
    assert np.allclose(
        arrays["mic_pos_world_m"][0], [1.0, 2.0, 3.0] + MIC_OFFSETS_BODY_M,
    )
    path = recorder.save(tmp_path / "trajectory.npz")
    assert path.exists()


def test_timestamps_must_increase():
    recorder = TrajectoryRecorder()
    state = GimbalState.from_pitch(0.0)
    recorder.record(0.0, [0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], state)
    try:
        recorder.record(0.0, [0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], state)
    except ValueError:
        return
    raise AssertionError("non-increasing timestamp accepted")


def test_pitch_and_quaternion_cannot_diverge():
    try:
        GimbalState(np.radians(15.0), np.array([1.0, 0.0, 0.0, 0.0]))
    except ValueError:
        return
    raise AssertionError("independent pitch/quaternion state accepted")
