import argparse
import importlib.util
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import scipy
from scipy.spatial.transform import Rotation


MODULE_PATH = Path(__file__).resolve().parents[1] / 'n3_contact_optimizer.py'
SPEC = importlib.util.spec_from_file_location('benchmark_n3_module', MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def distribution(samples, budget_ms):
    values = np.asarray(samples)
    return dict(count=len(samples), mean_ms=float(np.mean(values)),
                p50_ms=float(np.percentile(values, 50)),
                p95_ms=float(np.percentile(values, 95)),
                p99_ms=float(np.percentile(values, 99)),
                max_ms=float(np.max(values)),
                over_budget=int(np.sum(values > budget_ms)),
                budget_ms=budget_ms)


def run_case(name, stiffness, force, frames, warmup, budget_ms):
    grid = np.array([[horizontal, vertical, 0.0]
                     for horizontal in np.linspace(-0.01, 0.01, 7)
                     for vertical in np.linspace(-0.01, 0.01, 8)])[:50]
    config = MODULE.N3ContactConfig(
        recovery_frames=1, optimization_budget_ms=budget_ms, progress_timeout=100000.0)
    controller = MODULE.N3ContactOptimizer(
        np.zeros(3), [0.0, 0.0, 0.0, 1.0], [grid, grid], config,
        normal=[0.0, 0.0, 1.0], stiffness=stiffness)
    durations = []
    solver_durations = []
    iterations = []
    modes = Counter()
    failures = Counter()
    cold_start = {}
    for frame in range(frames + warmup):
        now = 1.0 + frame * config.control_period
        actual_rotation = controller.command_rotation
        world_force = np.asarray(force) + [0.02 * np.sin(frame * 0.07), 0.0,
                                          0.05 * np.sin(frame * 0.11)]
        measured_force = actual_rotation.inv().apply(world_force)
        target = Rotation.from_euler('y', 0.2 * np.sin(frame * 0.01), degrees=True).as_quat()
        started_at = time.perf_counter()
        controller.step(
            controller.command_position.copy(), actual_rotation.as_quat(),
            measured_force, [0.0, 0.0, 0.1], np.zeros(3), target,
            np.zeros(3), [grid, grid], [now] * 7, True, now=now)
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        status = controller.status
        if controller.fault or 'solver_ms' not in status or status.get('solver_skipped'):
            raise RuntimeError(f'Benchmark bypassed solver: {controller.fault}, {status}')
        if frame == 0:
            cold_start = dict(step_ms=elapsed_ms, solver_ms=status['solver_ms'],
                              success=status['solver_success'])
        if frame < warmup:
            continue
        durations.append(elapsed_ms)
        solver_durations.append(status['solver_ms'])
        iterations.append(status['solver_iterations'])
        modes[status['mode']] += 1
        if not status['solver_success']:
            failures[status['solver_message']] += 1
    return dict(
        scenario=name, configured_solver_budget_ms=budget_ms, cold_start=cold_start,
        solver=distribution(solver_durations, 6.0),
        controller_step=distribution(durations, 10.0),
        iterations_mean=float(np.mean(iterations)), iterations_max=int(max(iterations)),
        modes=dict(modes), solver_failures=dict(failures))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=int, default=1000)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    cases = [
        ('diagonal', np.diag([1000.0, 1200.0, 1800.0]), [0.2, 0.1, 2.4]),
        ('coupled_near_cone', np.array([[1800.0, 200.0, 100.0],
                                       [200.0, 1200.0, -100.0],
                                       [100.0, -100.0, 2400.0]]), [0.59, 0.1, 2.25]),
        ('weak_probe', np.diag([51.3, 310.8, 0.103]), [0.2, 0.0, 1.8]),
    ]
    report = dict(
        environment=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                         platform=platform.platform(), processor=platform.processor()),
        frames_per_scenario=arguments.frames, warmup_per_scenario=arguments.warmup,
        tactile_points_per_side=50, period_ms=10.0,
        scope='Sequential offline synthetic replay; no ROS, DDS, robot or real-time scheduler.',
        profiles={})
    for profile, budget in [('unrestricted_timing', 1000000.0), ('production_budget', 6.0)]:
        report['profiles'][profile] = [
            run_case(name, stiffness, force, arguments.frames, arguments.warmup, budget)
            for name, stiffness, force in cases]
    arguments.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    for profile, results in report['profiles'].items():
        for result in results:
            timings = result['controller_step']
            print(f'{profile}/{result["scenario"]}: step mean={timings["mean_ms"]:.3f}ms, '
                  f'p99={timings["p99_ms"]:.3f}ms, max={timings["max_ms"]:.3f}ms, '
                  f'over10ms={timings["over_budget"]}/{timings["count"]}, '
                  f'solver_failures={sum(result["solver_failures"].values())}')


if __name__ == '__main__':
    main()
