import numpy as np
import pytest

from openrobot_brain.safety import SafetyError, ValidatedPlan, validate_round
from openrobot_brain.schema import parse_agent_json
from conftest import make_world, move_to, pick, place_at, place_on, respond


def check(robot, kin, actions, world=None, q=None, finger_open=True):
    r = parse_agent_json(respond("act", actions))
    return validate_round(list(r.actions), robot, kin, robot.rest_pose if q is None else q,
                          world or make_world(), finger_open)


def test_pick_and_stack_is_valid(robot, kin):
    vp = check(robot, kin, [pick("red_cube"), place_on("blue_cube")])
    assert isinstance(vp, ValidatedPlan)
    red, blue = vp.world_end.objects["red_cube"].pos, vp.world_end.objects["blue_cube"].pos
    assert np.allclose(red, blue + [0, 0, 0.04])
    assert vp.world_end.held is None
    assert vp.segments[-1].label == "return to camera pose"


def test_three_high_tower_is_valid(robot, kin):
    check(robot, kin, [pick("red_cube"), place_on("blue_cube"), pick("green_cube"), place_on("red_cube")])


def test_fourth_level_exceeds_stack_limit(robot, kin):
    w = make_world(blue_cube=(0.45, -0.15, 0.02), red_cube=(0.45, -0.15, 0.06), green_cube=(0.45, -0.15, 0.10),
                   yellow_cube=(0.6, 0.1, 0.02))
    with pytest.raises(SafetyError, match="stack would exceed 3"):
        check(robot, kin, [pick("yellow_cube"), place_on("green_cube")], w)


@pytest.mark.parametrize("actions,match", [
    ([place_on("blue_cube")], "not holding anything"),
    ([place_at(0.5, 0.0)], "not holding anything"),
    ([pick("red_cube"), pick("green_cube")], "already holding 'red_cube'"),
    ([pick("purple_cube")], "unknown object 'purple_cube'"),
    ([pick("red_cube"), place_on("red_cube")], "on itself"),
    ([pick("red_cube"), place_at(1.2, 0.0)], "outside the workspace"),
    ([pick("red_cube"), place_at(0.62, 0.02)], "too close to 'green_cube'"),
    ([move_to(0.5, 0.45, 0.2)], "outside the workspace"),
    ([move_to(0.5, 0.0, 0.005)], "outside"),
    ([move_to(0.5, 0.0, 0.9)], "outside"),
    ([{"action": "wait", "duration_s": 0.0}], "duration"),
    ([{"action": "wait", "duration_s": 10.0}], "duration"),
])
def test_preconditions_and_bounds(robot, kin, actions, match):
    with pytest.raises(SafetyError, match=match):
        check(robot, kin, actions)


def test_cannot_pick_object_with_something_on_top(robot, kin):
    w = make_world(blue_cube=(0.45, -0.15, 0.02), red_cube=(0.45, -0.15, 0.06), green_cube=(0.6, 0.0, 0.02))
    with pytest.raises(SafetyError, match="'blue_cube' has 'red_cube' on top"):
        check(robot, kin, [pick("blue_cube")], w)
    with pytest.raises(SafetyError, match="already has 'red_cube' on top"):
        check(robot, kin, [pick("green_cube"), place_on("blue_cube")], w)


def test_held_object_collision_is_caught(robot, kin):
    # carry the red cube and lower it into the green cube
    with pytest.raises(SafetyError, match="held object 'red_cube' would hit 'green_cube'"):
        check(robot, kin, [pick("red_cube"), move_to(0.6, 0.0, 0.05)])


def test_arm_collision_with_object_is_caught(robot, kin):
    """Neighbours 5.5 cm away on two sides: every one of the four grasps puts an open
    finger into a cube, so the validator must reject the pick."""
    w = make_world(red_cube=(0.45, 0.15, 0.02), blue_cube=(0.395, 0.15, 0.02), green_cube=(0.45, 0.095, 0.02))
    with pytest.raises(SafetyError, match="arm would hit"):
        check(robot, kin, [pick("red_cube")], w)


def test_grasp_choice_turns_to_avoid_a_neighbour(robot, kin):
    """One neighbour 5.5 cm away along x blocks the straight grasp (fingers close along x).
    The skill picks the grasp turned 90 degrees instead, and the validator accepts it."""
    from openrobot_brain.simulation.kinematics import tcp_yaw
    w = make_world(red_cube=(0.45, 0.15, 0.02), blue_cube=(0.395, 0.15, 0.02))
    vp = check(robot, kin, [pick("red_cube")], w)
    grasp = next(s for s in vp.segments if s.label == "close gripper")
    yaw = np.degrees(tcp_yaw(kin.fk(grasp.q[0])[1]))
    assert abs(abs(yaw) - 90) < 2, yaw


def test_collision_check_uses_object_rotation(robot, kin):
    """At the grasp pose, a straight neighbour 8.6 cm away clears the open fingers, but the
    same cube turned 45 degrees pokes a corner into them. The contact check must see the turn."""
    q = kin.ik(np.array([0.45, 0.15, 0.02]), robot.ready_pose).q
    straight = make_world(red_cube=(0.45, 0.15, 0.02), blue_cube=(0.364, 0.15, 0.02))
    turned = straight.copy()
    turned.objects["blue_cube"].yaw = np.pi / 4

    def hits_blue(w):
        return any("blue_cube" in pair for pair in kin.contacts(q, True, w))
    assert not hits_blue(straight)
    assert hits_blue(turned)


def test_world_copy_keeps_yaw():
    w = make_world()
    w.objects["red_cube"].yaw = 0.5
    assert w.copy().objects["red_cube"].yaw == 0.5      # the safety check works on copies


def test_rotated_cube_is_gripped_square_on(robot, kin):
    """The planned grasp turns the gripper to the cube's own angle (or 90 degrees from it),
    and a cube placed on another ends up lined up with it."""
    from openrobot_brain.simulation.kinematics import tcp_yaw
    from openrobot_brain.world import wrap_square
    w = make_world()
    w.objects["red_cube"].yaw = np.radians(30)
    w.objects["blue_cube"].yaw = np.radians(-20)
    vp = check(robot, kin, [pick("red_cube"), place_on("blue_cube")], w)
    grasp = next(s for s in vp.segments if s.label == "close gripper")
    assert abs(np.degrees(wrap_square(tcp_yaw(kin.fk(grasp.q[0])[1]) - np.radians(30)))) < 1
    end = vp.world_end.objects
    assert abs(np.degrees(wrap_square(end["red_cube"].yaw - end["blue_cube"].yaw))) < 1


def test_grasp_rotation():
    from openrobot_brain.simulation.kinematics import R_DOWN, grasp_rotation
    assert np.allclose(grasp_rotation(0.0), R_DOWN)
    R = grasp_rotation(np.pi / 6)
    assert np.allclose(R[:, 2], [0, 0, -1])                         # still pointing straight down
    assert np.isclose(np.degrees(np.arctan2(R[1, 1], R[0, 1])), 30)  # finger axis turned 30 degrees


def test_neighbour_along_y_is_fine(robot, kin):
    # the same gap along y does not obstruct the fingers
    check(robot, kin, [pick("red_cube")], make_world(red_cube=(0.45, 0.15, 0.02), blue_cube=(0.45, 0.095, 0.02)))


def test_unreachable_object(robot, kin):
    w = make_world(red_cube=(0.80, 0.0, 0.02))
    with pytest.raises(SafetyError, match="outside the reachable workspace"):
        check(robot, kin, [pick("red_cube")], w)


def test_action_count_limit(robot, kin):
    with pytest.raises(SafetyError, match="limit is 6"):
        check(robot, kin, [{"action": "wait", "duration_s": 0.1}] * 7)


def test_bad_start_configuration(robot, kin):
    q = robot.rest_pose.copy()
    q[3] = 0.5  # joint4 must be negative
    with pytest.raises(SafetyError, match="start configuration"):
        check(robot, kin, [{"action": "wait", "duration_s": 0.1}], q=q)


def test_all_violations_found_before_anything_moves(robot, kin):
    with pytest.raises(SafetyError) as exc:
        check(robot, kin, [pick("red_cube"), place_at(1.5, 0)] + [{"action": "wait", "duration_s": 0.1}] * 5)
    assert any("limit is 6" in v for v in exc.value.violations)


def test_validated_trajectory_obeys_all_limits(robot, kin):
    vp = check(robot, kin, [pick("red_cube"), place_on("blue_cube"), pick("green_cube"), place_at(0.4, 0.25)])
    lo, hi, ws, lim = robot.lower(), robot.upper(), robot.workspace, robot.safety
    for s in vp.segments:
        assert np.all(s.q >= lo - 1e-9) and np.all(s.q <= hi + 1e-9)
        if len(s.t) > 1:
            dq = np.abs(np.diff(s.q, axis=0))
            assert dq.max() <= lim.max_joint_step_rad
            assert (dq / np.diff(s.t)[:, None]).max() <= lim.max_joint_velocity_rad_s + 1e-6
        if s.kind == "cartesian":
            p0, p1 = s.tcp_targets[0], s.tcp_targets[-1]
            if np.linalg.norm(p1[:2] - p0[:2]) > 0.01:
                assert min(p0[2], p1[2]) >= ws.transit_z - 2e-3, s.label   # sideways only at transit height
            for q, target in zip(s.q, s.tcp_targets):
                p, R = kin.fk(q)
                assert np.linalg.norm(p - target) < lim.ik_pos_tol_m
                assert R[2, 2] < -0.999   # gripper pointing straight down


def test_validated_plan_cannot_be_forged(robot):
    with pytest.raises(TypeError):
        ValidatedPlan(object(), [], robot, robot.rest_pose, make_world(), [], make_world())


def test_corner_to_corner_moves_are_valid(robot, kin):
    ws = robot.workspace
    corners = [(ws.x[0], ws.y[0]), (ws.x[1], ws.y[1]), (ws.x[0], ws.y[1]), (ws.x[1], ws.y[0])]
    for (ax, ay), (bx, by) in zip(corners, corners[1:]):
        check(robot, kin, [pick("c"), place_at(bx, by)], make_world(c=(ax, ay, 0.02)))
