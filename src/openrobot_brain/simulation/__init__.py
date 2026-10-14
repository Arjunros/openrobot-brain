from pathlib import Path

PANDA_DIR = Path(__file__).with_name("franka_emika_panda")
DEFAULT_SCENE = PANDA_DIR / "orb_tabletop.xml"
CUBE_BODIES = ("red_cube", "green_cube", "blue_cube")
ROBOT_BODIES = ("link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7",
                "hand", "left_finger", "right_finger")
