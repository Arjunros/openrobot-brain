# Third-party model: Franka Emika Panda

Source: Google DeepMind MuJoCo Menagerie, `franka_emika_panda/`
https://github.com/google-deepmind/mujoco_menagerie (commit d7b481bac63d3d600ab404ee74f7474c5efe8a67)
License: Apache-2.0 (see `LICENSE` in this folder).

Modifications by OpenRobot-Brain:
- `panda.xml`: added a `tcp` site on the `hand` body at the fingertip centre (z = 0.1034 m).

The `mjx_*` variants and the original `scene.xml` were not copied.
- `orb_tabletop.xml` (OpenRobot-Brain's own scene: work mat, cubes, cameras) lives here so mesh paths resolve; it is not part of Menagerie.
