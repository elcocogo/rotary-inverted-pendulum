"""Simulation MuJoCo d'un pendule inversé rotatif (Rotary Inverted Pendulum).

Le modèle MJCF est généré depuis un template Jinja2 (model.xml) paramétré par
`Params`. Le script lance ensuite une simulation temps réel avec le viewer
MuJoCo, pilotée par la loi de commande choisie via `--controller`.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import asdict, dataclass

import mujoco
import mujoco.viewer
import numpy as np
from jinja2 import Template

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GRAVITY = 9.81  # m/s^2

logger = logging.getLogger(__name__)


@dataclass
class Params:
    """Paramètres géométriques et temporels du modèle, injectés dans model.xml."""

    time_step: float = 0.001
    motor_pos_z: float = 1.5
    motor_radius: float = 0.2
    motor_halfheight: float = 0.2
    arm_length: float = 1.0
    arm_radius: float = 0.05
    pendulum_length: float = 1.0
    pendulum_radius: float = 0.05
    alpha0: float = np.pi+0.05  # angle initial du pendule (rad)

    @property
    def fps(self) -> int:
        return int(1 / self.time_step)

    def to_template_context(self) -> dict:
        """Dict prêt à être injecté dans le template Jinja2 model.xml."""
        return {**asdict(self), "fps": self.fps, "np": np, "base_dir": BASE_DIR}


@dataclass
class ControllerParams:
    """Gains et seuils des lois de commande."""

    torque_max: float = 10.0
    kp: float = 200.0  # gain proportionnel (= 20 * torque_max)
    mu: float = 1.0  # gain du correcteur swing-up par énergie
    angle_tolerance: float = np.deg2rad(5)  # seuil de bascule du mode hybride


def build_model_xml(params: Params, *, save_rendered_to: str | None = None) -> str:
    """Rend model.xml avec les paramètres fournis et retourne le MJCF résultant."""
    template_path = os.path.join(BASE_DIR, "model.xml")
    with open(template_path, encoding="utf-8") as f:
        template = Template(f.read())
    xml = template.render(params.to_template_context())

    if save_rendered_to:
        with open(save_rendered_to, "w", encoding="utf-8") as f:
            f.write(xml)

    return xml


def get_joint_ids(model: mujoco.MjModel) -> dict[str, int]:
    return {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in ("pendulum_joint", "arm_joint")
    }


def pivot_inertia(model: mujoco.MjModel, params: Params) -> float:
    """Moment d'inertie du pendule autour de son point de pivot (Huygens-Steiner)."""
    pendulum = model.body("pendulum")
    j_center = model.body_inertia[pendulum.id][0]
    return j_center + model.body_mass[pendulum.id] * (params.pendulum_length / 2) ** 2


def proportional_control(angle: float, ctrl: ControllerParams) -> float:
    """Stabilisation autour du point haut (angle = pi) par retour proportionnel."""
    error = np.arctan2(np.sin(np.pi - angle), np.cos(np.pi - angle))
    return np.clip(ctrl.kp * error, -ctrl.torque_max, ctrl.torque_max)


def energy_swingup_control(
    angle: float,
    angular_velocity: float,
    model: mujoco.MjModel,
    params: Params,
    pivot_j: float,
    ctrl: ControllerParams,
) -> float:
    """Swing-up par injection d'énergie jusqu'à atteindre l'énergie du point haut."""
    pendulum_mass = model.body_mass[model.body("pendulum").id]
    e_ref = pendulum_mass * GRAVITY * params.pendulum_length
    e_kin = 0.5 * pivot_j * angular_velocity**2
    e_pot = 0.5 * pendulum_mass * GRAVITY * params.pendulum_length * (1 - np.cos(angle))
    energy_error = (e_kin + e_pot) - e_ref
    sign = np.where(np.cos(angle) * angular_velocity >= 0, 1, -1)
    return np.clip(ctrl.mu * energy_error * sign, -ctrl.torque_max, ctrl.torque_max)


def hybrid_control(
    angle: float,
    angular_velocity: float,
    model: mujoco.MjModel,
    params: Params,
    pivot_j: float,
    ctrl: ControllerParams,
) -> float:
    """Swing-up par énergie, puis bascule sur un retour proportionnel près du sommet."""
    error = np.arctan2(np.sin(np.pi - angle), np.cos(np.pi - angle))
    if np.abs(error) <= ctrl.angle_tolerance:
        return proportional_control(angle, ctrl)
    return energy_swingup_control(angle, angular_velocity, model, params, pivot_j, ctrl)


CONTROLLERS = {
    "swingup": energy_swingup_control,
    "hybrid": hybrid_control,
    # signature plus courte que les autres : ne dépend pas de model/params/pivot_j
    "proportional": lambda angle, _avel, _model, _params, _pivot_j, ctrl: proportional_control(angle, ctrl),
}


def set_camera(viewer, params: Params) -> None:
    viewer.cam.lookat[:] = [0, 0, params.motor_pos_z]  # point visé, au centre de l'image
    viewer.cam.distance = 5.0  # zoom sur ce point
    viewer.cam.azimuth = 180  # angle autour de l'axe z passant par ce point
    viewer.cam.elevation = -45.0  # angle de surplomb (négatif = au dessus)


def run_simulation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    params: Params,
    ctrl: ControllerParams,
    controller_name: str,
) -> None:
    joint_ids = get_joint_ids(model)
    pivot_j = pivot_inertia(model, params)
    control_fn = CONTROLLERS[controller_name]

    with mujoco.viewer.launch_passive(model, data) as viewer:
        set_camera(viewer, params)

        while viewer.is_running():
            step_start = time.time()

            angle = data.qpos[model.jnt_qposadr[joint_ids["pendulum_joint"]]] + params.alpha0
            angular_velocity = data.qvel[model.jnt_dofadr[joint_ids["pendulum_joint"]]]

            data.ctrl[0] = control_fn(angle, angular_velocity, model, params, pivot_j, ctrl)
            logger.debug("angle=%.3f qvel=%.3f ctrl=%.3f", angle, angular_velocity, data.ctrl[0])

            mujoco.mj_step(model, data)
            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--controller",
        choices=sorted(CONTROLLERS),
        default="swingup",
        help="loi de commande utilisée pendant la simulation",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="affiche l'état du système à chaque pas de temps"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")

    params = Params()
    ctrl = ControllerParams()

    xml = build_model_xml(params, save_rendered_to=os.path.join(BASE_DIR, "model_rendered.xml"))
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    run_simulation(model, data, params, ctrl, args.controller)


if __name__ == "__main__":
    main()
