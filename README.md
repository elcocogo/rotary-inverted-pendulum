# Rotary Inverted Pendulum

A MuJoCo simulation of a rotary inverted pendulum (Furuta pendulum): a motorized arm rotates in the horizontal plane and carries a free-swinging pendulum at its tip. The goal is to swing the pendulum up from a hanging position and balance it upright.

The MJCF model is generated from a [Jinja2](https://jinja.palletsprojects.com/) template (`model.xml`), parameterized in Python, so geometry and physical constants live in one place.

## Controllers

Three control laws are implemented and selectable at runtime:

| Name           | Strategy                                                                 |
|----------------|---------------------------------------------------------------------------|
| `swingup`      | Energy-based swing-up: injects/removes energy until the pendulum reaches the energy of the upright position (default) |
| `proportional` | Simple proportional feedback around the upright equilibrium — only works if already close to it |
| `hybrid`       | Energy-based swing-up until the pendulum is near the top, then switches to proportional feedback |

## Installation

Dependencies are managed with [uv](https://docs.astral.sh/uv/). Install uv, then:

```bash
uv sync
```

This creates a `.venv` and installs the exact dependency versions pinned in `uv.lock`.

## Usage

```bash
uv run python sim.py --controller swingup
uv run python sim.py --controller hybrid -v   # -v prints angle/velocity/torque at each step
uv run python sim.py --help
```

A MuJoCo viewer window opens and the simulation runs in real time. Close the window to stop.

## Project structure

```
.
├── sim.py       # simulation entry point: model build, controllers, real-time loop
├── model.xml    # Jinja2-templated MJCF model (parameterized geometry)
├── scene.xml    # static MJCF scene (ground, lighting, etc.), included by model.xml
├── docs/        # background reading on rotary inverted pendulum dynamics and control
├── pyproject.toml
└── uv.lock      # pinned dependency versions, keep in sync with pyproject.toml
```

`model_rendered.xml` is generated at each run (the fully-substituted MJCF, for inspection) and is not meant to be edited or committed.

## Background

The `docs/` folder contains reference material on the dynamics and control of rotary inverted pendulums (energy-based swing-up, LQR balancing) used while developing the controllers in this repo.
