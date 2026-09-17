"""Simulation MuJoCo d'un pendule inversé rotatif (Rotary Inverted Pendulum).

Le modèle MJCF est généré depuis un template Jinja2 (model.xml) paramétré par
`Params`. Le script lance ensuite une simulation temps réel avec le viewer
MuJoCo, pilotée par la loi de commande choisie via `--controller`.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from queue import Empty

import mujoco
import mujoco.viewer
import numpy as np
import scipy.linalg
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
    alpha0: float = 0.0  # angle initial du pendule (rad) — voir INITIAL_ANGLES pour les presets nommés

    @property
    def fps(self) -> int:
        return int(1 / self.time_step)

    def to_template_context(self) -> dict:
        """Dict prêt à être injecté dans le template Jinja2 model.xml."""
        return {**asdict(self), "fps": self.fps, "np": np, "base_dir": BASE_DIR}


INITIAL_ANGLES = {
    "down": 0.0,  # pendule pendant vers le bas, à l'arrêt (cas classique du swing-up)
    "up": np.pi,  # pendule à la verticale, exactement à l'équilibre instable
    "near-up": np.pi - np.deg2rad(3),  # léger écart (3°) par rapport au sommet
    "side": np.pi / 2,  # pendule à l'horizontale
}


@dataclass
class ControllerParams:
    """Gains et seuils des lois de commande."""

    torque_max: float = 10.0
    kp: float = 200.0  # gain du correcteur "proportional" — volontairement naïf, voir README
    mu: float = 1.0  # gain du correcteur swing-up par énergie
    angle_tolerance: float = np.deg2rad(20)  # seuil de bascule du mode hybride vers le LQR

    # Pondérations du LQR (état = [angle bras, angle pendule, vitesse bras, vitesse pendule]).
    # L'angle du bras est pondéré faiblement mais pas à zéro : un poids nul en ferait un état
    # purement intégrateur (valeur propre 1) et rend la résolution de l'équation de Riccati
    # numériquement instable.
    lqr_q_arm_angle: float = 1.0
    lqr_q_pendulum_angle: float = 20.0
    lqr_q_arm_velocity: float = 1.0
    lqr_q_pendulum_velocity: float = 1.0
    lqr_r: float = 0.05
    lqr_gain: np.ndarray | None = None  # calculé une fois au démarrage, voir compute_lqr_gain()


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


def compute_lqr_gain(model: mujoco.MjModel, params: Params, ctrl: ControllerParams, joint_ids: dict[str, int]) -> np.ndarray:
    """Linéarise le modèle autour du point haut et calcule le gain du régulateur LQR discret.

    La linéarisation (A, B) est obtenue par différences finies via MuJoCo
    (`mjd_transitionFD`), plutôt que dérivée à la main — cette approche
    utilise directement le modèle réel (masses, inerties, couplages), qui
    s'est déjà montré plus fiable qu'une formule physique simplifiée
    (voir `energy_swingup_control`). u = -K @ (x - x_eq).
    """
    pendulum_qpos_idx = model.jnt_qposadr[joint_ids["pendulum_joint"]]
    arm_qpos_idx = model.jnt_qposadr[joint_ids["arm_joint"]]
    pendulum_dof_idx = model.jnt_dofadr[joint_ids["pendulum_joint"]]
    arm_dof_idx = model.jnt_dofadr[joint_ids["arm_joint"]]

    scratch = mujoco.MjData(model)
    scratch.qpos[pendulum_qpos_idx] = np.pi - params.alpha0
    scratch.qpos[arm_qpos_idx] = 0.0
    mujoco.mj_forward(model, scratch)

    n_states = model.nq + model.nv
    a = np.zeros((n_states, n_states))
    b = np.zeros((n_states, model.nu))
    mujoco.mjd_transitionFD(model, scratch, 1e-6, True, a, b, None, None)

    q_diag = np.zeros(n_states)
    q_diag[arm_qpos_idx] = ctrl.lqr_q_arm_angle
    q_diag[pendulum_qpos_idx] = ctrl.lqr_q_pendulum_angle
    q_diag[model.nq + arm_dof_idx] = ctrl.lqr_q_arm_velocity
    q_diag[model.nq + pendulum_dof_idx] = ctrl.lqr_q_pendulum_velocity
    q = np.diag(q_diag)
    r = np.array([[ctrl.lqr_r]])

    p = scipy.linalg.solve_discrete_are(a, b, q, r)
    return np.linalg.inv(r + b.T @ p @ b) @ (b.T @ p @ a)


def proportional_control(angle: float, ctrl: ControllerParams) -> float:
    """Stabilisation autour du point haut (angle = pi) par retour proportionnel.

    Volontairement naïf : sans terme de freinage sur la vitesse ni retour sur
    l'état du bras, ce correcteur seul n'arrive pas à se stabiliser sur ce
    système non amorti (il rebondit indéfiniment près du sommet). Conservé
    comme référence pédagogique — comparer `--controller proportional` et
    `--controller hybrid` (qui utilise le LQR) le montre bien.
    """
    error = np.arctan2(np.sin(np.pi - angle), np.cos(np.pi - angle))
    return np.clip(ctrl.kp * error, -ctrl.torque_max, ctrl.torque_max)


def lqr_control(
    _angle: float,
    _angular_velocity: float,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    params: Params,
    ctrl: ControllerParams,
    joint_ids: dict[str, int],
) -> float:
    """Stabilisation en position haute par retour d'état LQR (bras + pendule, angle + vitesse)."""
    x = np.concatenate([data.qpos, data.qvel])
    x_eq = np.zeros_like(x)
    x_eq[model.jnt_qposadr[joint_ids["pendulum_joint"]]] = np.pi - params.alpha0
    u = -ctrl.lqr_gain @ (x - x_eq)
    return float(np.clip(u[0], -ctrl.torque_max, ctrl.torque_max))


def energy_swingup_control(
    angle: float,
    angular_velocity: float,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    params: Params,
    ctrl: ControllerParams,
    _joint_ids: dict[str, int],
) -> float:
    """Swing-up par injection d'énergie jusqu'à atteindre l'énergie du point haut.

    L'énergie cinétique est celle du système complet (bras + pendule couplés),
    calculée par MuJoCo à partir de sa matrice de masse. Une formule à pivot
    fixe (0.5 * J * angular_velocity**2) sous-estime/surestime cette énergie
    selon la vitesse du bras, ce qui bloquait le swing-up sur un cycle limite
    bien avant d'atteindre le point haut.
    """
    pendulum_mass = model.body_mass[model.body("pendulum").id]
    e_ref = pendulum_mass * GRAVITY * params.pendulum_length
    mujoco.mj_energyVel(model, data)
    e_kin = data.energy[1]
    e_pot = 0.5 * pendulum_mass * GRAVITY * params.pendulum_length * (1 - np.cos(angle))
    energy_error = (e_kin + e_pot) - e_ref
    sign = np.where(np.cos(angle) * angular_velocity >= 0, 1, -1)
    return np.clip(ctrl.mu * energy_error * sign, -ctrl.torque_max, ctrl.torque_max)


def hybrid_control(
    angle: float,
    angular_velocity: float,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    params: Params,
    ctrl: ControllerParams,
    joint_ids: dict[str, int],
) -> float:
    """Swing-up par énergie, puis bascule sur un retour d'état LQR près du sommet.

    Le seuil de bascule (`angle_tolerance`) est sensible : trop large, le LQR
    (valide seulement près de l'équilibre) prend le relais alors que la
    linéarisation n'est plus une bonne approximation, ce qui redéstabilise le
    système au lieu de le stabiliser. 20° s'est montré fiable en pratique.
    """
    error = np.arctan2(np.sin(np.pi - angle), np.cos(np.pi - angle))
    if np.abs(error) <= ctrl.angle_tolerance:
        return lqr_control(angle, angular_velocity, model, data, params, ctrl, joint_ids)
    return energy_swingup_control(angle, angular_velocity, model, data, params, ctrl, joint_ids)


CONTROLLERS = {
    "swingup": energy_swingup_control,
    "hybrid": hybrid_control,
    "lqr": lqr_control,  # stabilisation seule, utile depuis --initial-angle near-up/up
    # signature plus courte que les autres : ne dépend pas de model/data/params/joint_ids
    "proportional": lambda angle, _avel, _model, _data, _params, ctrl, _joint_ids: proportional_control(angle, ctrl),
}


def _plot_worker(queue: mp.Queue, label: str, window_seconds: float, redraw_every: int) -> None:
    """Boucle d'affichage, exécutée dans son propre processus.

    Sur macOS, une fenêtre matplotlib doit être créée sur le thread principal
    de SON processus. `mjpython` réserve déjà le thread principal du processus
    de simulation à la fenêtre MuJoCo ; matplotlib ne peut donc pas s'afficher
    dans ce même processus — d'où ce processus dédié, qui a son propre thread
    principal libre.
    """
    import matplotlib.pyplot as plt

    times: deque[float] = deque()
    values: deque[float] = deque()

    plt.ion()
    fig, ax = plt.subplots()
    (line,) = ax.plot([], [])
    ax.set_xlabel("temps (s)")
    ax.set_ylabel(label)
    fig.show()

    received_since_redraw = 0
    while plt.fignum_exists(fig.number):
        try:
            t, value = queue.get(timeout=0.05)
        except Empty:
            fig.canvas.flush_events()
            continue

        times.append(t)
        values.append(value)
        while times and times[0] < t - window_seconds:
            times.popleft()
            values.popleft()

        received_since_redraw += 1
        if received_since_redraw < redraw_every:
            continue
        received_since_redraw = 0

        line.set_data(times, values)
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw_idle()
        fig.canvas.flush_events()


class RealTimePlot:
    """Trace en direct une grandeur scalaire au fil de la simulation (fenêtre glissante).

    Générique : n'importe quelle grandeur scalaire échantillonnée dans la
    boucle de simulation peut être tracée en lui donnant juste un nom et un
    appel `update(t, value)` à chaque pas — voir `METRICS` pour brancher une
    nouvelle grandeur. L'affichage tourne dans un processus séparé (voir
    `_plot_worker`) ; `update` se contente d'empiler les points dans une file.
    """

    def __init__(self, label: str, *, window_seconds: float = 10.0, redraw_every: int = 20) -> None:
        self._queue: mp.Queue = mp.Queue()
        self._process = mp.Process(
            target=_plot_worker, args=(self._queue, label, window_seconds, redraw_every), daemon=True
        )
        self._process.start()

    def update(self, t: float, value: float) -> None:
        self._queue.put((t, value))


def energy_metric(
    angle: float, _angular_velocity: float, _u: float, model: mujoco.MjModel, data: mujoco.MjData, params: Params
) -> float:
    pendulum_mass = model.body_mass[model.body("pendulum").id]
    mujoco.mj_energyVel(model, data)
    e_kin = data.energy[1]
    e_pot = 0.5 * pendulum_mass * GRAVITY * params.pendulum_length * (1 - np.cos(angle))
    return e_kin + e_pot


def angle_metric(
    angle: float, _angular_velocity: float, _u: float, _model: mujoco.MjModel, _data: mujoco.MjData, _params: Params
) -> float:
    return np.arctan2(np.sin(angle), np.cos(angle))


def torque_metric(
    _angle: float, _angular_velocity: float, u: float, _model: mujoco.MjModel, _data: mujoco.MjData, _params: Params
) -> float:
    return u


METRICS: dict[str, Callable[..., float]] = {
    "energy": energy_metric,
    "angle": angle_metric,
    "torque": torque_metric,
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
    plot_metric: str | None = None,
) -> None:
    joint_ids = get_joint_ids(model)
    control_fn = CONTROLLERS[controller_name]
    live_plot = RealTimePlot(plot_metric) if plot_metric else None

    with mujoco.viewer.launch_passive(model, data) as viewer:
        set_camera(viewer, params)

        while viewer.is_running():
            step_start = time.time()

            angle = data.qpos[model.jnt_qposadr[joint_ids["pendulum_joint"]]] + params.alpha0
            angular_velocity = data.qvel[model.jnt_dofadr[joint_ids["pendulum_joint"]]]

            u = control_fn(angle, angular_velocity, model, data, params, ctrl, joint_ids)
            data.ctrl[0] = u
            logger.debug("angle=%.3f qvel=%.3f ctrl=%.3f", angle, angular_velocity, u)

            if live_plot is not None:
                live_plot.update(data.time, METRICS[plot_metric](angle, angular_velocity, u, model, data, params))

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
        "--initial-angle",
        choices=sorted(INITIAL_ANGLES),
        default="down",
        help="position initiale du pendule",
    )
    parser.add_argument(
        "--plot",
        choices=sorted(METRICS),
        default=None,
        help="trace cette grandeur en direct dans une fenêtre matplotlib",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="affiche l'état du système à chaque pas de temps"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")

    params = Params(alpha0=INITIAL_ANGLES[args.initial_angle])
    ctrl = ControllerParams()

    xml = build_model_xml(params, save_rendered_to=os.path.join(BASE_DIR, "model_rendered.xml"))
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    if args.controller in ("lqr", "hybrid"):
        ctrl.lqr_gain = compute_lqr_gain(model, params, ctrl, get_joint_ids(model))

    run_simulation(model, data, params, ctrl, args.controller, plot_metric=args.plot)


if __name__ == "__main__":
    main()
