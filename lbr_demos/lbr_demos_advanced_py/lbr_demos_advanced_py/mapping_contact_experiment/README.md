# Contact Mapping Experiment

This folder contains an offline, reproducible implementation for the mapping
task described in `../mapping task.md`.

## What It Does

- Uses `motion_planning_5.py` style 4-direction discrete actions.
- Uses the source movable object geometry: one square object with vertices
  `[-25,25], [-25,-25], [25,-25], [25,25]` in the object reference frame.
- Treats `obstacle_ref` only as a prior contour; the algorithm does not use or
  assume a global scale relation between `obstacle_ref` and `obstacle_true`.
- Represents every corresponding obstacle edge with an independent support-line
  constant bounded by the known exploration rectangle.
- The known exploration rectangle is `obstacle_true`'s bounding box plus
  `50 mm` on each side in this simulation.
- Generates candidate contact pairs after each contact.
- Enforces zero-area overlap between every saved/contact `movable_objects`
  pose and `obstacle_true`; boundary contact is allowed.
- Executes motion as 4-direction axis-aligned polylines only (`+X`, `-X`,
  `+Y`, `-Y`). If a planned movement would overlap `obstacle_true`, the
  simulation stops at the first boundary contact and records that as the actual
  contact observation.
- Updates the obstacle support-line constants and reconstructs vertices from
  adjacent support-line intersections.
- Visualizes each vertex coordinate range as a SLAM-style uncertainty ellipse.
- Runs several exploration strategies and selects the one with the fewest
  discrete action steps among successful runs.
- Saves every exploration step as a PNG plus trace data as CSV/JSON/NPZ.

## Run

```bash
MPLCONFIGDIR=/tmp/mplconfig XDG_CACHE_HOME=/tmp/xdg-cache \
python3 mapping_contact_experiment.py --out final_results_independent_edges
```

From the repository working directory, the command used for the checked result
was:

```bash
MPLCONFIGDIR=/tmp/mplconfig XDG_CACHE_HOME=/tmp/xdg-cache \
python3 mapping_contact_experiment/mapping_contact_experiment.py \
  --out mapping_contact_experiment/final_results_independent_edges
```

The `MPLCONFIGDIR` and `XDG_CACHE_HOME` variables avoid Matplotlib cache
permission warnings in this ROS workspace.

## Checked Result

The verified output is in `final_results_independent_edges/`.

- Best strategy: `nearest_next`
- Success: `True`
- Probe count: `12`
- Discrete action steps: `916`
- Max vertex error: `0.000000 mm`
- Mean vertex error: `0.000000 mm`
- Max contact overlap area: `0.000000000000 mm^2`

Open `final_results_independent_edges/REPORT.md` for the concise experiment report.
