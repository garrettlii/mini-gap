"""Builds the MuJoCo model for the workcell. OWNER: done

The scene lives in assets/franka_emika_panda/workcell.xml (edit it freely).
A few elements attach to bodies defined inside Menagerie's panda.xml, which an
including file cannot modify, so they are added here with MjSpec.
"""

from __future__ import annotations

from pathlib import Path

import mujoco

REPO = Path(__file__).resolve().parents[2]
WORKCELL_XML = REPO / "assets" / "franka_emika_panda" / "workcell.xml"

# Names used across the codebase. Change them here, nowhere else.
ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
ARM_ACTUATORS = tuple(f"actuator{i}" for i in range(1, 8))
GRIPPER_ACTUATOR = "actuator8"      # ctrl 0 = closed, 255 = open (Menagerie remap)
GRIPPER_OPEN_CTRL = 255.0
GRIPPER_CLOSED_CTRL = 0.0
FINGER_JOINTS = ("finger_joint1", "finger_joint2")
TCP_SITE = "tcp"
CUBES = ("red_cube", "green_cube", "blue_cube")
ZONES = ("zone_a", "zone_b")
CAMERAS = ("overhead", "front", "wrist")
HOME_KEY = "home"
TABLE_Z = 0.0
CUBE_HALF = 0.02

GRIP_GAIN = 10.0                    # gripper stiffness multiplier over Menagerie's

# Franka's flange-to-fingertip-center distance.
TCP_OFFSET = (0.0, 0.0, 0.1034)


def build_spec(xml_path: Path = WORKCELL_XML) -> mujoco.MjSpec:
    if not xml_path.exists():
        raise FileNotFoundError(
            f"{xml_path} not found. Run scripts/setup_mac.sh to fetch the Menagerie Panda."
        )
    if not (xml_path.parent / "panda.xml").exists():
        raise FileNotFoundError(
            f"panda.xml missing next to {xml_path.name}. Run scripts/setup_mac.sh."
        )
    spec = mujoco.MjSpec.from_file(str(xml_path))
    hand = spec.body("hand")
    # group=3 keeps it hidden in renders by default; toggle group 3 in the viewer to see it.
    hand.add_site(name=TCP_SITE, pos=list(TCP_OFFSET), size=[0.006, 0, 0],
                  rgba=[1, 0, 1, 0.6], group=3)
    # A camera looks along its own -z; quat (0,1,0,0) flips it to look along the hand's +z.
    hand.add_camera(name="wrist", pos=[0.06, 0.0, 0.03], quat=[0, 1, 0, 0], fovy=75)
    # Gravity compensation, as Franka's own controller does: without it the position servos
    # sag ~7 mm under the arm's weight and every TCP target is missed by that much.
    for body in spec.bodies:
        if body.name.startswith("link") or body.name in ("hand", "left_finger", "right_finger"):
            body.gravcomp = 1.0
    # Menagerie's gripper squeezes a 4 cm cube with ~1 N per finger, less than the cube's
    # table friction, so a 0.5 mm off-center grasp stalls on one finger. 10x stiffer is ~10 N
    # (the real hand: up to 70 N); damping scales by sqrt(10) to stay about as damped.
    grip = spec.actuator(GRIPPER_ACTUATOR)
    grip.gainprm[0] *= GRIP_GAIN
    grip.biasprm[1] *= GRIP_GAIN
    grip.biasprm[2] *= GRIP_GAIN ** 0.5
    return spec


def build_model(xml_path: Path = WORKCELL_XML) -> mujoco.MjModel:
    return build_spec(xml_path).compile()
