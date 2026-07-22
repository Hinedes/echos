import numpy as np

from echos_frontier import (
    BODY_HALF_EXTENTS,
    MissionContactFailure,
    body,
    physical_acceptance_ok,
    physical_body_clearance,
    physical_contacts,
    run_frontier_exploration,
    rs,
    scene,
)


IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def test_forced_wall_contact_is_reported():
    body.set_pos((2.75, -1.5, 1.0), zero_velocity=True)
    body.set_quat(IDENTITY, zero_velocity=True, relative=False)
    scene.step()
    contacts = physical_contacts()
    assert contacts
    assert any(contact["other_entity"] == "south_wall" for contact in contacts)
    assert any(contact["penetration_depth_m"] > 0.0 for contact in contacts)
    assert all("contact_point" in contact and "normal" in contact for contact in contacts)


def test_side_wall_clearance_is_not_forward_sensor_range():
    clearance, obstacle = physical_body_clearance((-0.40, 0.0, 1.0), IDENTITY)
    assert obstacle == "west_wall"
    assert 0.02 < clearance < 0.05


def test_body_overlap_detected_when_center_is_outside_wall():
    clearance, obstacle = physical_body_clearance((2.75, -1.44, 1.0), IDENTITY)
    assert obstacle == "south_wall"
    assert clearance < 0.0
    assert BODY_HALF_EXTENTS[1] > 0.05


def test_safe_parallel_flight_clearance_matches_box_geometry():
    clearance, obstacle = physical_body_clearance((2.75, -1.0, 1.0), IDENTITY)
    assert obstacle == "south_wall"
    assert abs(clearance - 0.4125) < 1e-6


def test_dashboard_gate_rejects_contact_or_clearance_failure():
    assert not physical_acceptance_ok(1, 0.9)
    assert not physical_acceptance_ok(0, 0.2)
    assert physical_acceptance_ok(0, 0.200001)


def test_full_frontier_mission_has_physical_clearance():
    result = run_frontier_exploration()
    assert result["collision_count"] == 0
    assert result["physical_clearance_m"] > 0.20
    assert result["passed"]
