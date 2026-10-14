import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


MODULE_PATH = Path(__file__).resolve().parents[1] / 'n3_contact_optimizer.py'
SPEC = importlib.util.spec_from_file_location('n3_contact_optimizer', MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def make_controller(**config_values):
    grid = np.array([[x, y, 0.0] for x in (-0.01, 0.0, 0.01)
                     for y in (-0.01, 0.0, 0.01)])
    config_values.setdefault('recovery_frames', 1)
    config_values.setdefault('optimization_budget_ms', 1000.0)
    config = MODULE.N3ContactConfig(**config_values)
    return MODULE.N3ContactOptimizer(
        np.zeros(3), [0.0, 0.0, 0.0, 1.0], [grid, grid], config,
        normal=[0.0, 0.0, 1.0]), [grid, grid]


def step(controller, clouds, now, force=(0.0, 0.0, 2.25), target=(0.001, 0.0, 0.0),
         times=None, moment=(0.0, 0.0, 0.1), target_quat=(0.0, 0.0, 0.0, 1.0),
         final_point=False, force_control=False):
    return controller.step(
        controller.command_position.copy(), controller.command_rotation.as_quat(),
        np.asarray(force, dtype=float), np.asarray(moment, dtype=float), np.asarray(target, dtype=float),
        target_quat, controller.command_position.copy(), clouds,
        [now] * 7 if times is None else times, True, now=now,
        final_point=final_point, force_control=force_control)


def test_initial_normal_is_used_without_probe_data():
    controller, _ = make_controller()
    np.testing.assert_allclose(controller.normal, [0.0, 0.0, 1.0])
    np.testing.assert_allclose(controller.stiffness, np.eye(3) * 100.0)


@pytest.mark.parametrize('normal', [None, [0.0, 0.0, 0.0], [np.nan, 0.0, 1.0], [0.0, 1.0]])
def test_invalid_normal_is_rejected(normal):
    grid = np.zeros((3, 3))
    with pytest.raises(ValueError, match='estimated_contact_normal'):
        MODULE.N3ContactOptimizer(np.zeros(3), [0, 0, 0, 1], [grid, grid], normal=normal)


def test_position_target_does_not_move_robot_or_block_advance():
    controller, clouds = make_controller()
    for frame in range(2):
        command, _, advance = step(controller, clouds, 1.0 + frame * 0.01)
    assert controller.status['mode'] == 'tracking'
    np.testing.assert_array_equal(command, np.zeros(3))
    assert advance


def test_soft_normal_feedback_does_not_block_orientation_advance():
    controller, clouds = make_controller()
    _, _, advance = step(controller, clouds, 1.0, force=(0.0, 0.0, 1.5))
    assert advance
    assert controller.status['mode'] == 'tracking'
    assert controller.status['normal_in_target'] is False
    np.testing.assert_array_equal(controller.previous_velocity[:2], np.zeros(2))
    assert controller.previous_velocity[2] < 0.0


def test_friction_excess_pauses_advance_and_optimizes_recovery():
    controller, clouds = make_controller()
    _, _, advance = step(controller, clouds, 1.0, force=(1.0, 0.0, 2.25))
    assert not advance
    assert controller.status['mode'] == 'recovery'
    np.testing.assert_allclose(controller.status['command_velocity'][3:], 0.0)
    assert controller.status['linear_speed'] <= controller.config.linear_speed
    assert controller.status['objective_after'] <= controller.status['objective_before']


def test_stable_friction_excess_waits_ten_seconds_before_increasing_safe_mu():
    controller, clouds = make_controller()
    initial_mu = controller.mu_safe
    for frame in range(200):
        step(controller, clouds, 1.0 + frame * 0.05, force=(0.6, 0.0, 2.25))
        assert controller.mu_safe == initial_mu
        assert controller.fault is None
        assert controller.status['reason'] in ('friction_recovery', 'optimization_failed')
        assert not controller.status.get('advance', False)
    step(controller, clouds, 11.0, force=(0.6, 0.0, 2.25))
    assert controller.mu_safe > initial_mu
    assert controller.status['friction_adapting'] is True
    assert controller.fault is None
    for frame in range(1, 6):
        step(controller, clouds, 11.0 + frame * 0.05, force=(0.6, 0.0, 2.25))
        if not controller.status['friction_exceeded']:
            break
    assert controller.status['friction_adapting'] is False
    assert controller.status['friction_exceeded'] is False


def test_force_bias_is_rotated_from_initial_reference():
    controller, clouds = make_controller()
    controller.force_bias = np.array([0.0, 0.0, -4.0])
    raw = np.array([0.0, 0.0, 2.25]) + controller.force_bias
    step(controller, clouds, 1.0, force=raw - controller.force_bias)
    assert controller.status['normal_force'] == pytest.approx(2.25)


def test_duplicate_tactile_frames_do_not_update_friction_window():
    controller, clouds = make_controller()
    for frame in range(5):
        step(controller, clouds, 1.0 + frame * 0.01, force=(0.1, 0.0, 2.25), times=[1.0] * 7)
    assert len(controller.force_ratios) == 1


def test_object_jump_pauses_without_latching_controller():
    controller, clouds = make_controller()
    controller.previous_velocity[0] = 0.001
    step(controller, clouds, 1.0)
    jumped = clouds[0].copy() + [0.01, 0.0, 0.0]
    command, _, advance = step(controller, [jumped, clouds[1]], 1.01)
    assert controller.fault is None
    assert controller.status['reason'] == 'object_pose_jump'
    assert not advance
    np.testing.assert_allclose(controller.previous_velocity, np.zeros(6))


def test_orientation_tracking_converges_without_following_position():
    controller, clouds = make_controller()
    target_quat = Rotation.from_euler('y', 0.15, degrees=True).as_quat()
    arrived = False
    for frame in range(300):
        force = controller.command_rotation.inv().apply([0.0, 0.0, 2.25])
        command, _, arrived = step(
            controller, clouds, 1.0 + frame * 0.01, force=force,
            target=(0.02, -0.03, 0.01), target_quat=target_quat, final_point=True)
        assert controller.fault is None
        np.testing.assert_allclose(command, np.zeros(3), atol=1e-12)
        assert controller.status['linear_speed'] == pytest.approx(np.linalg.norm(controller.previous_velocity[:3]))
        assert controller.status['angular_speed'] == pytest.approx(np.linalg.norm(controller.previous_velocity[3:]))
        if arrived:
            break
    assert arrived
    assert controller.status['angle_error'] <= controller.config.angle_tolerance
    assert controller.status['angle_error_deg'] <= 0.02


def test_target_orientation_is_published_without_internal_rotation_ramp():
    controller, clouds = make_controller()
    target_quat = Rotation.from_euler('y', 0.2, degrees=True).as_quat()
    command, command_quat, _ = step(
        controller, clouds, 1.0, target_quat=target_quat)
    np.testing.assert_allclose(command_quat, target_quat)
    assert controller.fault is None
    np.testing.assert_allclose(controller.command_rotation.as_quat(), target_quat)


def test_mu_stays_constant_during_normal_tracking():
    controller, clouds = make_controller()
    controller.mu_max_estimate = 0.5
    controller.mu_safe = 0.4
    for frame in range(40):
        _, _, advance = step(controller, clouds, 1.0 + frame * 0.01,
                             force=(0.2 + frame * 0.003, 0.0, 2.25))
        assert advance
        assert controller.mu_safe == 0.4
    assert len(controller.force_ratios) == 5


def test_mu_stops_increasing_once_friction_allows_tracking():
    controller, clouds = make_controller()
    for frame in range(220):
        _, _, advance = step(controller, clouds, 1.0 + frame * 0.05,
                             force=(0.6, 0.0, 2.25))
        if advance:
            break
    assert advance
    adapted_mu = controller.mu_safe
    assert adapted_mu > 0.21
    for following_frame in range(1, 30):
        _, _, advance = step(controller, clouds, 1.0 + (frame + following_frame) * 0.05,
                             force=(0.6, 0.0, 2.25))
        assert advance
        assert controller.mu_safe == adapted_mu


def test_mu_freezes_during_unstable_episode_and_recovers():
    controller, clouds = make_controller(recovery_frames=3)
    step(controller, clouds, 1.0)
    before = controller.mu_safe
    for frame in range(1, 5):
        shifted = [cloud + [frame * 0.00007, 0.0, 0.0] for cloud in clouds]
        _, _, advance = step(controller, shifted, 1.0 + frame * 0.01)
        assert not advance
        assert controller.mu_safe == pytest.approx(before)
        assert controller.fault is None
    for frame in range(5, 8):
        _, _, advance = step(controller, shifted, 1.0 + frame * 0.01)
    assert advance
    assert not controller.object_unstable
    shifted = [cloud + [0.01, 0.0, 0.0] for cloud in shifted]
    step(controller, shifted, 1.08)
    assert controller.mu_safe == pytest.approx(before)


def test_tracking_to_object_unstable_reduces_mu_once_per_transition():
    controller, clouds = make_controller(recovery_frames=3)
    for frame in range(3):
        step(controller, clouds, 1.0 + frame * 0.01)
    assert controller.status['mode'] == 'tracking'
    initial_mu = controller.mu_safe
    shifted = [cloud + [0.00007, 0.0, 0.0] for cloud in clouds]
    _, _, advance = step(controller, shifted, 1.03)
    assert not advance
    assert controller.status['mode'] == 'recovery'
    assert controller.status['reason'] == 'object_pose_unstable'
    assert controller.mu_safe == pytest.approx(0.95 * initial_mu)
    assert controller.status['mu_safe'] == controller.mu_safe
    assert controller.status['friction_limit'] == pytest.approx(controller.mu_safe * 2.25)
    for frame in range(1, 4):
        step(controller, shifted, 1.03 + frame * 0.01)
        assert controller.mu_safe == pytest.approx(0.95 * initial_mu)
    assert controller.status['mode'] == 'tracking'
    shifted = [cloud + [0.00007, 0.0, 0.0] for cloud in shifted]
    step(controller, shifted, 1.07)
    assert controller.status['reason'] == 'object_pose_unstable'
    assert controller.mu_safe == pytest.approx(0.95 ** 2 * initial_mu)


def test_force_control_prevents_mu_increase_and_advance():
    controller, clouds = make_controller()
    before = controller.mu_safe
    for frame in range(10):
        _, _, advance = step(controller, clouds, 1.0 + frame * 0.01,
                             force=(0.675, 0.0, 2.25), force_control=True)
        assert not advance
        assert controller.mu_safe == before


def test_invalid_cloud_and_stale_sensor_hold():
    controller, clouds = make_controller()
    bad = clouds[0].copy()
    bad[0, 2] += 0.01
    step(controller, [bad, clouds[1]], 1.0)
    assert controller.fault == 'object_fit_residual'
    controller, clouds = make_controller()
    step(controller, clouds, 1.0, times=[0.0] * 7)
    assert controller.fault == 'stale_sensor'


@pytest.mark.parametrize('force', [[0.9, 0.3, 2.1], [0.42, 0.0, 2.25], [0.2, 0.1, 1.8]])
def test_analytic_objective_gradient_matches_finite_difference(force):
    controller, _ = make_controller()
    controller.stiffness = np.array([[900.0, 100.0, 20.0],
                                     [100.0, 600.0, -30.0],
                                     [20.0, -30.0, 1500.0]])
    delta = np.array([2.0e-6, -3.0e-6, 1.0e-6])
    arguments = (np.array(force), 0.01, 0.15,
                 np.array([-0.04, 0.02, 0.05]), 0.24)
    _, gradient = controller._objective_and_gradient(delta, *arguments)
    numerical = np.zeros(3)
    for axis in range(3):
        offset = np.eye(3)[axis] * 1.0e-9
        numerical[axis] = (controller._objective(delta + offset, *arguments)
                           - controller._objective(delta - offset, *arguments)) / 2.0e-9
    np.testing.assert_allclose(gradient, numerical, rtol=1.0e-6, atol=1.0e-5)


@pytest.mark.parametrize('force_values', [[0.2, 0.1, 2.6], [0.55, 0.0, 2.6]])
def test_solver_reduces_cost_and_returns_feasible_next_position(force_values):
    controller, _ = make_controller()
    controller.stiffness = np.diag([1000.0, 1200.0, 1800.0])
    position = np.array([0.5, -0.03, 0.05])
    force = np.array(force_values)
    dt = 0.01
    next_position, delta, before, after = controller._optimize_position(
        position, force, dt, 0.0, np.zeros(3), 0.24)
    assert controller.optimization_status['solver_success']
    assert after < before
    assert after == pytest.approx(
        controller._objective(delta, force, dt, 0.0, np.zeros(3), 0.24), abs=1.0e-10)
    np.testing.assert_allclose(next_position, position + delta)
    assert delta[0] > 0.0
    assert delta[2] > 0.0
    assert np.linalg.norm(delta / dt) <= controller.config.linear_speed + 1.0e-10
    assert np.all(np.abs(delta / dt) <= controller.config.linear_acceleration * dt + 1.0e-10)
    predicted_force = force - controller.stiffness @ delta
    assert np.linalg.norm(predicted_force[:2]) + controller.config.force_reserve <= 0.24 * predicted_force[2]


def test_step_publishes_exact_solver_position():
    controller, clouds = make_controller()
    command, _, _ = step(controller, clouds, 1.0, force=(0.2, 0.0, 2.4))
    assert controller.status['solver_success']
    np.testing.assert_array_equal(command, controller.status['optimized_next_position'])
    np.testing.assert_allclose(command, controller.status['optimized_delta_position'])
    assert controller.status['solver_ms'] > 0.0
    assert controller.status['controller_ms'] >= controller.status['solver_ms']


def test_active_predicted_cone_is_enforced_by_solver():
    controller, _ = make_controller(
        objective_normal_weight=1000.0, objective_tangent_weight=0.0,
        objective_object_weight=0.0, objective_wrench_weight=0.0,
        objective_friction_weight=0.0)
    controller.stiffness = np.diag([1000.0, 1000.0, 2000.0])
    force = np.array([0.7, 0.0, 3.0])
    _, delta, _, _ = controller._optimize_position(
        np.zeros(3), force, 0.01, 0.0, np.zeros(3), 0.24)
    assert controller.optimization_status['solver_success']
    predicted = force - controller.stiffness @ delta
    assert np.linalg.norm(predicted[:2]) + 0.02 <= 0.24 * predicted[2] + 1.0e-8
    assert delta[0] > 0.0
    assert delta[2] > 0.0


def test_solver_failure_never_publishes_candidate(monkeypatch):
    from types import SimpleNamespace

    controller, clouds = make_controller()
    def fail_solver(*arguments, **keywords):
        return SimpleNamespace(success=False, x=np.ones(3) * 100.0,
                               fun=1.0, nit=30, message='Iteration limit')
    monkeypatch.setattr(MODULE, 'minimize', fail_solver)
    command, _, advance = step(controller, clouds, 1.0)
    np.testing.assert_array_equal(command, np.zeros(3))
    assert not advance
    assert controller.status['reason'] == 'optimization_failed'
    assert not controller.status['solver_success']


def test_infeasible_solution_rejected_even_when_solver_reports_success(monkeypatch):
    from types import SimpleNamespace

    controller, clouds = make_controller()
    def infeasible_solver(*arguments, **keywords):
        return SimpleNamespace(success=True, x=np.ones(3), fun=0.0, nit=1, message='Success')
    monkeypatch.setattr(MODULE, 'minimize', infeasible_solver)
    command, _, advance = step(controller, clouds, 1.0)
    np.testing.assert_array_equal(command, np.zeros(3))
    assert not advance
    assert not controller.status['solver_success']


def test_solver_timeout_holds_without_latching():
    controller, clouds = make_controller(optimization_budget_ms=0.0)
    command, _, advance = step(controller, clouds, 1.0, force=(0.1, 0.0, 2.25))
    np.testing.assert_array_equal(command, np.zeros(3))
    assert not advance
    assert controller.fault is None
    assert controller.status['solver_budget_exceeded']
    assert controller.status['reason'] == 'optimization_failed'


def test_force_change_preserves_vector_and_does_not_repeat_on_duplicate_frames():
    controller, clouds = make_controller()
    step(controller, clouds, 1.0, force=(0.1, 0.2, 2.25))
    step(controller, clouds, 1.01, force=(0.12, 0.15, 2.3))
    np.testing.assert_allclose(controller.force_change, [0.02, -0.05, 0.05])
    step(controller, clouds, 1.02, force=(0.12, 0.15, 2.3), times=[1.01] * 7)
    np.testing.assert_allclose(controller.force_change, [0.02, -0.05, 0.05])


def test_friction_penalty_increases_inside_measured_cone():
    controller, _ = make_controller(
        objective_normal_weight=0.0, objective_tangent_weight=0.0,
        objective_smooth_weight=0.0, objective_object_weight=0.0,
        objective_wrench_weight=0.0)
    costs = [controller._objective(np.zeros(3), np.array([tangent, 0.0, 2.25]),
                                   0.01, 0.0, np.zeros(3), 0.24)
             for tangent in (0.3, 0.45, 0.5)]
    assert costs[0] == 0.0
    assert 0.0 < costs[1] < costs[2]
    assert 0.5 + controller.config.force_reserve < 0.24 * 2.25


def test_friction_recovery_holds_measured_orientation_and_restores_without_mu_growth():
    controller, clouds = make_controller()
    target_quat = Rotation.from_euler('y', 3.0, degrees=True).as_quat()
    force = np.array([0.8, 0.0, 2.25])
    initial_mu = controller.mu_safe
    for frame in range(500):
        position = controller.command_position.copy()
        command, command_quat, _ = step(
            controller, clouds, 1.0 + frame * 0.01, force=force,
            target_quat=target_quat)
        assert controller.fault is None
        assert controller.mu_safe == initial_mu
        if controller.status['tracking_allowed']:
            np.testing.assert_allclose(command_quat, target_quat)
            assert controller.friction_recovery_started_at is None
            break
        np.testing.assert_allclose(command_quat, [0.0, 0.0, 0.0, 1.0])
        force -= controller.stiffness @ (command - position)
    else:
        pytest.fail('Position optimization did not restore the friction constraint.')


def test_friction_hold_does_not_follow_target_or_measured_orientation_drift():
    controller, clouds = make_controller()
    step(controller, clouds, 1.0, force=(0.8, 0.0, 2.25),
         target_quat=Rotation.from_euler('y', 3.0, degrees=True).as_quat())
    controller.command_rotation = Rotation.from_euler('x', 0.05, degrees=True)
    _, command_quat, advance = step(
        controller, clouds, 1.01, force=(0.8, 0.0, 2.25),
        target_quat=Rotation.from_euler('y', -3.0, degrees=True).as_quat())
    np.testing.assert_allclose(command_quat, [0.0, 0.0, 0.0, 1.0])
    assert not advance


def test_mu_cap_and_no_progress_protection_after_recovery_window():
    controller, clouds = make_controller()
    for frame in range(401):
        step(controller, clouds, 1.0 + frame * 0.05, force=(0.8, 0.0, 2.25))
        assert controller.mu_safe <= controller.mu_max_estimate
        assert controller.fault is None
    assert controller.mu_safe == pytest.approx(controller.mu_max_estimate)
    assert controller.status['friction_exceeded']
    step(controller, clouds, 21.05, force=(0.8, 0.0, 2.25))
    assert controller.fault == 'no_progress_timeout'


def test_force_control_interrupts_friction_recovery_wait():
    controller, clouds = make_controller()
    initial_mu = controller.mu_safe
    for frame in range(100):
        step(controller, clouds, 1.0 + frame * 0.05, force=(0.6, 0.0, 2.25))
    step(controller, clouds, 6.0, force=(0.6, 0.0, 2.25), force_control=True)
    assert controller.friction_recovery_started_at is None
    step(controller, clouds, 6.05, force=(0.6, 0.0, 2.25))
    assert controller.friction_recovery_started_at == 6.05
    assert controller.mu_safe == pytest.approx(initial_mu)


def test_new_segment_resets_friction_recovery_state():
    controller, clouds = make_controller()
    step(controller, clouds, 1.0, force=(0.6, 0.0, 2.25))
    assert controller.friction_hold_rotation is not None
    controller.start_segment()
    assert controller.friction_hold_rotation is None
    assert controller.friction_recovery_started_at is None


def test_ordinary_tracking_still_has_no_progress_timeout():
    controller, clouds = make_controller(progress_timeout=0.03)
    target_quat = Rotation.from_euler('y', 3.0, degrees=True).as_quat()
    for frame in range(5):
        controller.step(
            np.zeros(3), [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 2.25],
            [0.0, 0.0, 0.1], np.zeros(3), target_quat, np.zeros(3), clouds,
            [1.0 + frame * 0.01] * 7, True, now=1.0 + frame * 0.01)
    assert controller.fault == 'no_progress_timeout'


def test_control_gap_waits_for_all_inputs_then_resumes():
    controller, clouds = make_controller(sensor_timeout=0.1)
    step(controller, clouds, 1.0)
    saved_position = controller.command_position.copy()
    saved_quat = controller.command_rotation.as_quat()
    target_quat = Rotation.from_euler('y', 0.2, degrees=True).as_quat()
    command, quat, advance = step(controller, clouds, 1.2, target_quat=target_quat)
    assert controller.fault is None
    assert controller.status['mode'] == 'waiting_data'
    assert controller.status['reason'] == 'control_gap'
    assert not advance
    np.testing.assert_array_equal(command, saved_position)
    np.testing.assert_array_equal(quat, saved_quat)
    np.testing.assert_array_equal(controller.previous_velocity, np.zeros(6))
    for missing in range(7):
        now = 1.21 + missing * 0.01
        arrivals = [now] * 7
        arrivals[missing] = 1.2
        step(controller, clouds, now, times=arrivals, target_quat=target_quat)
        assert controller.status['mode'] == 'waiting_data'
        assert controller.fault is None
    _, quat, _ = step(controller, clouds, 1.28, target_quat=target_quat)
    assert controller.control_gap_wait_started_at is None
    assert controller.status['mode'] == 'tracking'
    np.testing.assert_allclose(quat, target_quat)


@pytest.mark.parametrize('times', [[], [1.21] * 6, [np.nan] * 7,
                                  [1.0] * 7, [1.3] * 7])
def test_control_gap_invalid_or_old_inputs_keep_waiting(times):
    controller, clouds = make_controller(sensor_timeout=0.1)
    step(controller, clouds, 1.0)
    step(controller, clouds, 1.2)
    _, _, advance = step(controller, clouds, 1.21, times=times)
    assert controller.status['mode'] == 'waiting_data'
    assert controller.fault is None
    assert not advance


@pytest.mark.parametrize('timeout', [0.1, 0.2])
def test_control_gap_requires_call_interval_to_recover(timeout):
    controller, clouds = make_controller(sensor_timeout=timeout)
    step(controller, clouds, 1.0)
    for frame in range(1, 4):
        now = 1.0 + frame * (timeout + 0.05)
        step(controller, clouds, now)
        assert controller.status['mode'] == 'waiting_data'
        assert controller.fault is None
    _, _, advance = step(controller, clouds, now + 0.01)
    assert advance


def test_control_gap_wait_does_not_solve_or_consume_recovery_timeout(monkeypatch):
    controller, clouds = make_controller(sensor_timeout=0.1)
    step(controller, clouds, 1.0, force=(0.8, 0.0, 2.25))
    initial_mu = controller.mu_safe
    with monkeypatch.context() as patch:
        def unexpected_solver(*arguments, **keywords):
            pytest.fail('Optimizer ran while waiting for fresh input.')
        patch.setattr(MODULE, 'minimize', unexpected_solver)
        for frame in range(401):
            step(controller, clouds, 1.2 + frame * 0.05,
                 times=[1.0] * 7, force=(0.8, 0.0, 2.25))
            assert controller.status['mode'] == 'waiting_data'
            assert controller.fault is None
            assert controller.mu_safe == initial_mu
    step(controller, clouds, 21.21, force=(0.8, 0.0, 2.25))
    assert controller.fault is None
    assert controller.status['reason'] == 'friction_recovery'
    assert controller.status['friction_recovery_seconds'] == 0.0
    assert controller.status['no_progress_seconds'] == pytest.approx(0.0)
    assert controller.mu_safe == initial_mu


def test_control_gap_rechecks_stability_before_releasing_orientation():
    controller, clouds = make_controller(recovery_frames=3, sensor_timeout=0.1)
    for frame in range(3):
        step(controller, clouds, 1.0 + frame * 0.01)
    step(controller, clouds, 1.2)
    target_quat = Rotation.from_euler('y', 0.2, degrees=True).as_quat()
    for frame in range(2):
        _, quat, advance = step(controller, clouds, 1.21 + frame * 0.01,
                                target_quat=target_quat)
        assert controller.status['reason'] == 'object_pose_unstable'
        assert not advance
        np.testing.assert_allclose(quat, [0.0, 0.0, 0.0, 1.0])
    _, quat, _ = step(controller, clouds, 1.23, target_quat=target_quat)
    assert controller.status['mode'] == 'tracking'
    np.testing.assert_allclose(quat, target_quat)


def test_control_gap_rechecks_force_control_and_object_jump():
    controller, clouds = make_controller(sensor_timeout=0.1)
    step(controller, clouds, 1.0)
    step(controller, clouds, 1.2)
    target_quat = Rotation.from_euler('y', 0.2, degrees=True).as_quat()
    _, quat, advance = step(controller, clouds, 1.21, force_control=True,
                            target_quat=target_quat)
    assert controller.status['reason'] == 'force_control'
    assert not advance
    np.testing.assert_allclose(quat, [0.0, 0.0, 0.0, 1.0])
    jumped = [cloud + [0.01, 0.0, 0.0] for cloud in clouds]
    _, quat, advance = step(controller, jumped, 1.22, target_quat=target_quat)
    assert controller.status['reason'] == 'object_pose_jump'
    assert not advance
    np.testing.assert_allclose(quat, [0.0, 0.0, 0.0, 1.0])


def test_control_gap_does_not_disable_or_clear_other_latched_faults():
    controller, clouds = make_controller(sensor_timeout=0.1)
    step(controller, clouds, 1.0)
    step(controller, clouds, 1.2)
    step(controller, clouds, 1.21, force=(0.0, 0.0, np.nan))
    assert controller.fault == 'nonfinite_input'
    step(controller, clouds, 1.5)
    assert controller.status['mode'] == 'hold'
    assert controller.fault == 'nonfinite_input'


def test_control_gap_preserves_existing_no_progress_time():
    controller, clouds = make_controller(sensor_timeout=0.1, progress_timeout=0.05)
    target_quat = Rotation.from_euler('y', 3.0, degrees=True).as_quat()
    for now in (1.0, 1.02, 1.2, 1.21, 1.23, 1.25):
        controller.step(
            np.zeros(3), [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 2.25],
            [0.0, 0.0, 0.1], np.zeros(3), target_quat, np.zeros(3), clouds,
            [now] * 7, True, now=now)
        if now == 1.21:
            assert controller.fault is None
            assert controller.status['no_progress_seconds'] == pytest.approx(0.02)
    assert controller.fault == 'no_progress_timeout'


@pytest.mark.parametrize('normal_force, expected_penalty',
                         [(1.5, 50.0), (1.9, 10.0), (2.0, 0.0), (2.25, 0.0)])
def test_underforce_penalty_is_linear_and_only_below_two_newtons(normal_force, expected_penalty):
    controller, _ = make_controller()
    arguments = (np.zeros(3), np.array([0.0, 0.0, normal_force]),
                 0.01, 0.0, np.zeros(3), 0.24)
    cost, gradient = controller._objective_and_gradient(*arguments)
    controller.config.objective_normal_underforce_weight = 0.0
    baseline_cost, baseline_gradient = controller._objective_and_gradient(*arguments)
    assert cost - baseline_cost == pytest.approx(expected_penalty)
    expected_gradient = np.array([0.0, 0.0, 10000.0 if normal_force < 2.0 else 0.0])
    np.testing.assert_allclose(gradient - baseline_gradient, expected_gradient)


def test_solver_uses_underforce_penalty_to_increase_normal_force():
    controller, _ = make_controller(objective_normal_weight=0.0)
    force = np.array([0.0, 0.0, 1.8])
    position, delta, before, after = controller._optimize_position(
        np.zeros(3), force, 0.01, 0.0, np.zeros(3), 0.24)
    assert controller.optimization_status['solver_success']
    assert position is not None
    assert after < before
    assert after == pytest.approx(
        controller._objective(delta, force, 0.01, 0.0, np.zeros(3), 0.24))
    assert (force - controller.stiffness @ delta)[2] > force[2]


@pytest.mark.parametrize('force,moment', [([0.0, 0.0, 0.05], [0.0, 0.0, 1.0]),
                                        ([0.0, 0.0, 12.0], [0.0, 0.0, 1.0]),
                                        ([0.0, 0.0, 2.3], [0.0, 0.0, 51.0])])
def test_wrench_limit_requests_pid_recovery_and_freezes_orientation(force, moment):
    controller, clouds = make_controller()
    target = Rotation.from_euler('y', 3.0, degrees=True).as_quat()
    _, quat, advance = step(controller, clouds, 1.0, force=force, moment=moment,
                            target_quat=target)
    assert controller.fault is None
    assert controller.status['mode'] == 'recovery'
    assert controller.status['pid_recovery']
    assert not advance
    np.testing.assert_allclose(quat, [0.0, 0.0, 0.0, 1.0])


def test_wrench_pid_recovery_continues_until_adjustment_target_then_resumes():
    controller, clouds = make_controller()
    step(controller, clouds, 1.0, force=(0.0, 0.0, 0.05))
    step(controller, clouds, 1.01, force=(0.0, 0.0, 1.0), moment=(0.0, 0.0, 1.0))
    assert controller.status['pid_recovery']
    _, _, advance = step(controller, clouds, 1.02, force=(0.0, 0.0, 2.3),
                         moment=(0.0, 0.0, 1.0), force_control=True)
    assert controller.fault is None
    assert controller.wrench_recovery_started_at is None
    assert advance


def test_wrench_pid_recovery_times_out_at_ten_seconds_even_if_waiting_for_data():
    controller, clouds = make_controller()
    step(controller, clouds, 1.0, force=(0.0, 0.0, 0.05))
    for frame in range(1, 200):
        step(controller, clouds, 1.0 + frame * 0.05, force=(0.0, 0.0, 0.05))
        assert controller.fault is None
        assert controller.status['mode'] == 'recovery'
    step(controller, clouds, 11.0, force=(0.0, 0.0, 0.05))
    assert controller.status['mode'] == 'hold'
    assert not controller.status['pid_recovery']
    assert controller.fault == 'wrench_recovery_timeout'
    controller, clouds = make_controller()
    step(controller, clouds, 1.0, force=(0.0, 0.0, 0.05))
    step(controller, clouds, 11.1)
    assert controller.fault == 'wrench_recovery_timeout'


@pytest.mark.parametrize('normal,force,moment,expected_sign',
                         [(0.05, 0.05, 1.0, -1), (12.0, 12.0, 1.0, 1),
                          (2.3, 2.3, 51.0, 1)])
def test_node_pid_applies_existing_z_correction(normal, force, moment, expected_sign):
    import ast
    from types import SimpleNamespace

    source = MODULE_PATH.with_name('pose_planning_node_realtime_contact_line_v2.py').read_text()
    method = next(node for node in ast.walk(ast.parse(source))
                  if isinstance(node, ast.FunctionDef) and node.name == '_apply_n3_wrench_recovery_pid')
    namespace = {'np': np}
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<pid-test>', 'exec'), namespace)
    node = SimpleNamespace(freq=100.0, z_delta=0.0,
                           get_logger=lambda: SimpleNamespace(info=lambda message: None))
    controller, _ = make_controller()
    controller.status = dict(normal_force=normal, force_norm=force, moment_norm=moment)
    position = np.array([0.5, -0.01, 0.35])
    command = namespace['_apply_n3_wrench_recovery_pid'](node, controller, position)
    assert node.force_control_flag
    assert np.sign(command[2] - position[2]) == expected_sign
    assert abs(command[2] - position[2]) <= 0.0035 / node.freq + 1e-12
    np.testing.assert_array_equal(command[:2], position[:2])
    np.testing.assert_array_equal(controller.command_position, command)
    previous_error = node.force_pid_previous_error
    namespace['_apply_n3_wrench_recovery_pid'](node, controller, position)
    assert node.force_pid_previous_error == previous_error
    assert controller.status['pid_correction'] == pytest.approx(
        np.clip(0.0035 * previous_error, -0.0035, 0.0035))
