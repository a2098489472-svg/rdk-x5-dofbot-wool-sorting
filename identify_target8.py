#!/usr/bin/env python
# coding: utf-8

import rospy
import Arm_Lib
import cv2 as cv
import numpy as np
import threading
import math
from time import sleep
from identify_grap8 import identify_grap
from yolo_detector import YOLOFiberDetector
from dofbot_info.srv import kinemarics, kinemaricsRequest, kinemaricsResponse


# ====================== WORKSPACE / DETECTION =========================

WORKSPACE = {
    "img_w": 640,
    "img_h": 480,
    "x_min": -0.080,
    "x_max":  0.080,
    "y_near": 0.190,
    "y_far":  0.318,
}

COLOR_OBSTACLE_RANGES = {
    "red": [
        ((0, 150, 100), (10, 255, 255)),
        ((170, 150, 100), (180, 255, 255)),
    ],
    "blue": [
        ((100, 150, 100), (130, 255, 255))
    ],
    "green": [
        ((40, 100, 100), (80, 255, 255))
    ],
}

MIN_OBSTACLE_AREA_PX = 800
IGNORE_BOTTOM_RATIO = 0.15


# ====================== MOTION CONSTANTS ==============================

GRAB_Z = 0.195
TRAVEL_Z = 0.235

SAFETY_MARGIN = 0.07
DETOUR_BONUS = 0.015
OBS_RADIUS_INFLATE = 1.10
ARC_POINTS = 14
ARM_BASE_XY = (0.0, 0.0)

# 放置避障阶段专用平滑策略：
# ARC 路径的最后一个点通常是放置走廊目标点，例如 (0.160, 0.200)。
# 中间弧线点经过 shrink_toward_base() 后会明显靠近底座，
# 最后一个点不参与 shrink，因此容易出现 “倒数第2点 -> 最后点” 的大幅前冲。
# 对 place transit 阶段删除这个最后走廊点，由后续 place rotate 高位姿态完成放置过渡。
DROP_PLACE_ARC_FINAL_GOAL = True
DROP_PLACE_ARC_FINAL_THRESHOLD = 0.050

# 放置避障阶段专用平滑起步策略：
# 抓取后若立即进入 place transit 避障，ARC 路径的第 1 段常出现
# target_xy -> shrink 后入口点的大跨度后撤。这里自动在第 1 段插入中间点，
# 并按距离自适应增加每个小段的运行时间，让避障起步更丝滑。
SMOOTH_PLACE_ARC_ENTRY = True
SMOOTH_PLACE_ARC_ENTRY_THRESHOLD = 0.045
SMOOTH_PLACE_ARC_ENTRY_MAX_STEP = 0.030
ADAPTIVE_ARC_STEP_TIME = True
ADAPTIVE_ARC_STEP_MIN_MS = 650
ADAPTIVE_ARC_STEP_MAX_MS = 1300
ADAPTIVE_ARC_STEP_MS_PER_M = 10000


# ====================== GRIPPER MODEL =================================

GRIPPER_LENGTH = 0.12
GRIPPER_LATERAL_MARGIN = 0.012
GRIPPER_SAMPLES_PER_SEG = 6
GRIPPER_ARC_INFLATE = 0.05
WRIST_HARD_MARGIN = 0.005

WAYPOINT_SHRINK = 0.12


# ====================== DRAW COLORS ===================================

CLASS_PREVIEW_COLOR = {
    "plastic": (0, 200, 255),
    "box": (255, 150, 50),
    "leaf": (50, 255, 150),
    "default": (200, 200, 200),
}

GRIPPER_BGR = (255, 0, 200)
GRIPPER_END_BGR = (255, 80, 255)


# ----------------------------------------------------------------------
# Coordinate helpers
# ----------------------------------------------------------------------

def pixel_to_world(px, py):
    w = WORKSPACE

    wx = w["x_min"] + (px / w["img_w"]) * (w["x_max"] - w["x_min"])
    wy = w["y_far"] - (py / w["img_h"]) * (w["y_far"] - w["y_near"])

    return wx, wy


def world_to_pixel(wx, wy):
    w = WORKSPACE

    px = int((wx - w["x_min"]) / (w["x_max"] - w["x_min"]) * w["img_w"])
    py = int((w["y_far"] - wy) / (w["y_far"] - w["y_near"]) * w["img_h"])

    return px, py


def meters_per_pixel():
    w = WORKSPACE

    sx = (w["x_max"] - w["x_min"]) / w["img_w"]
    sy = (w["y_far"] - w["y_near"]) / w["img_h"]

    return 0.5 * (sx + sy)


def line_circle_dist(p1, p2, c):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    denom = dx * dx + dy * dy

    if denom < 1e-9:
        return math.hypot(p1[0] - c[0], p1[1] - c[1])

    u = ((c[0] - p1[0]) * dx + (c[1] - p1[1]) * dy) / denom
    u = max(0.0, min(1.0, u))

    cx = p1[0] + u * dx
    cy = p1[1] + u * dy

    return math.hypot(cx - c[0], cy - c[1])


def segment_clear(a, b, obstacles, margin):
    for ox, oy, r in obstacles:
        if line_circle_dist(a, b, (ox, oy)) < (r + margin):
            return False

    return True


def in_play_area(p):
    w = WORKSPACE

    return (
        w["x_min"] - 0.04 <= p[0] <= w["x_max"] + 0.04 and
        w["y_near"] - 0.02 <= p[1] <= w["y_far"] + 0.04
    )


# ----------------------------------------------------------------------
# Gripper geometry
# ----------------------------------------------------------------------

def wrist_xy(tcp_xy, base_xy=ARM_BASE_XY, length=GRIPPER_LENGTH):
    dx = tcp_xy[0] - base_xy[0]
    dy = tcp_xy[1] - base_xy[1]

    R = math.hypot(dx, dy)

    if R < 1e-6:
        return [base_xy[0], base_xy[1]]

    factor = max(0.0, (R - length) / R)

    return [
        base_xy[0] + dx * factor,
        base_xy[1] + dy * factor
    ]


def gripper_clear_at_tcp(tcp_xy, obstacles, lateral=GRIPPER_LATERAL_MARGIN):
    wr = wrist_xy(tcp_xy)

    for ox, oy, r in obstacles:
        if line_circle_dist(wr, tcp_xy, (ox, oy)) < (r + lateral):
            return False, (ox, oy, r), wr

    return True, None, wrist_xy(tcp_xy)


def path_gripper_clear(path, obstacles,
                       lateral=GRIPPER_LATERAL_MARGIN,
                       n_samples=GRIPPER_SAMPLES_PER_SEG):
    if not path:
        return True, None, None

    for wp in path:
        ok, bad, _ = gripper_clear_at_tcp(wp, obstacles, lateral)

        if not ok:
            return False, wp, bad

    for a, b in zip(path[:-1], path[1:]):
        for i in range(1, n_samples):
            t = i / float(n_samples)

            tcp = [
                a[0] + t * (b[0] - a[0]),
                a[1] + t * (b[1] - a[1])
            ]

            ok, bad, _ = gripper_clear_at_tcp(tcp, obstacles, lateral)

            if not ok:
                return False, tcp, bad

    return True, None, None


# ----------------------------------------------------------------------
# Waypoint shrink helpers
# ----------------------------------------------------------------------

def shrink_toward_base(p, amount=WAYPOINT_SHRINK, base=ARM_BASE_XY):
    dx = p[0] - base[0]
    dy = p[1] - base[1]

    R = math.hypot(dx, dy)

    if R < 1e-6 or amount <= 0.0:
        return list(p)

    f = max(0.0, (R - amount) / R)

    return [
        base[0] + dx * f,
        base[1] + dy * f
    ]


def shrink_intermediate(path, amount=WAYPOINT_SHRINK, base=ARM_BASE_XY):
    if not path or len(path) <= 2 or amount <= 0.0:
        return [list(p) for p in path]

    out = [list(path[0])]

    for p in path[1:-1]:
        out.append(shrink_toward_base(p, amount, base))

    out.append(list(path[-1]))

    return out


def densify_first_segment(path, max_step=0.030, threshold=0.045):
    """
    对路径起始第一段做加密。

    用途：place transit 阶段，抓取点 start_xy 到 shrink 后的第一个弧线入口点
    可能距离过大，机械臂会突然向后避障。将这一段拆成多个小段后，
    TCP 轨迹不会突然大幅跳变。

    返回：new_path, original_jump, inserted_count
    """
    if not path or len(path) < 2:
        return [list(p) for p in path], 0.0, 0

    a = list(path[0])
    b = list(path[1])

    d = math.hypot(b[0] - a[0], b[1] - a[1])

    if d <= threshold or max_step <= 1e-6:
        return [list(p) for p in path], d, 0

    pieces = int(math.ceil(d / max_step))
    pieces = max(2, pieces)

    out = [a]

    for k in range(1, pieces):
        t = k / float(pieces)
        out.append([
            a[0] + t * (b[0] - a[0]),
            a[1] + t * (b[1] - a[1])
        ])

    out.extend([list(p) for p in path[1:]])

    return out, d, pieces - 1


# ----------------------------------------------------------------------
# Arc planner
# ----------------------------------------------------------------------

def _build_arc(start, goal, obstacle, direction,
               n_arc=ARC_POINTS, margin=SAFETY_MARGIN, extra_inflate=0.0):
    ox, oy, r_min = obstacle

    r_safe = r_min + margin + extra_inflate

    a_s = math.atan2(start[1] - oy, start[0] - ox)
    a_g = math.atan2(goal[1] - oy, goal[0] - ox)

    delta = a_g - a_s

    while delta > math.pi:
        delta -= 2 * math.pi

    while delta < -math.pi:
        delta += 2 * math.pi

    if direction == "short":
        sweep = delta
    else:
        sweep = delta - 2 * math.pi if delta >= 0 else delta + 2 * math.pi

    enter = [
        ox + r_safe * math.cos(a_s),
        oy + r_safe * math.sin(a_s)
    ]

    exit_p = [
        ox + r_safe * math.cos(a_g),
        oy + r_safe * math.sin(a_g)
    ]

    path = [list(start), enter]

    for i in range(1, n_arc):
        t = i / float(n_arc)
        a = a_s + sweep * t

        path.append([
            ox + r_safe * math.cos(a),
            oy + r_safe * math.sin(a)
        ])

    path.append(exit_p)
    path.append(list(goal))

    return path


def _arc_dist_to_base(path, base_xy):
    bx, by = base_xy
    mx, my = path[len(path) // 2]

    return math.hypot(mx - bx, my - by)


def plan_around(start, goal, obstacles, margin=SAFETY_MARGIN, verbose=False):
    info = {
        "kind": "STRAIGHT",
        "d_short": None,
        "d_long": None,
        "chosen": None,
        "obs": None,
        "inflate": 0.0,
        "gripper_ok": True,
        "gripper_warn": None,
    }

    direct = [list(start), list(goal)]

    tcp_clear = segment_clear(start, goal, obstacles, margin)
    grip_ok, _, _ = path_gripper_clear(direct, obstacles)

    if tcp_clear and grip_ok:
        return direct, None, info

    blocking = None

    for ox, oy, r in obstacles:
        if line_circle_dist(start, goal, (ox, oy)) < (r + margin):
            blocking = (ox, oy, r)
            break

    if blocking is None:
        for ox, oy, r in obstacles:
            wr_s = wrist_xy(start)
            wr_g = wrist_xy(goal)

            if (
                line_circle_dist(wr_s, start, (ox, oy)) < r + GRIPPER_LATERAL_MARGIN or
                line_circle_dist(wr_g, goal, (ox, oy)) < r + GRIPPER_LATERAL_MARGIN
            ):
                blocking = (ox, oy, r)
                break

    if blocking is None:
        return direct, None, info

    chosen_inflate = 0.0

    path_short = None
    path_long = None

    short_grip_ok = False
    long_grip_ok = False

    for inflate in (GRIPPER_ARC_INFLATE, 0.04, 0.025, 0.01, 0.0):
        ps = _build_arc(
            start,
            goal,
            blocking,
            "short",
            margin=margin,
            extra_inflate=inflate
        )

        pl = _build_arc(
            start,
            goal,
            blocking,
            "long",
            margin=margin,
            extra_inflate=inflate
        )

        if not (
            in_play_area(ps[len(ps) // 2]) and
            in_play_area(pl[len(pl) // 2])
        ):
            continue

        s_ok, _, _ = path_gripper_clear(ps, obstacles)
        l_ok, _, _ = path_gripper_clear(pl, obstacles)

        if s_ok or l_ok:
            chosen_inflate = inflate

            path_short = ps
            path_long = pl

            short_grip_ok = s_ok
            long_grip_ok = l_ok

            break

        if path_short is None:
            path_short = ps
            path_long = pl
            chosen_inflate = inflate

    if path_short is None:
        path_short = _build_arc(
            start,
            goal,
            blocking,
            "short",
            margin=margin,
            extra_inflate=0.0
        )

        path_long = _build_arc(
            start,
            goal,
            blocking,
            "long",
            margin=margin,
            extra_inflate=0.0
        )

        chosen_inflate = 0.0

        short_grip_ok, _, _ = path_gripper_clear(path_short, obstacles)
        long_grip_ok, _, _ = path_gripper_clear(path_long, obstacles)

    d_short = _arc_dist_to_base(path_short, ARM_BASE_XY)
    d_long = _arc_dist_to_base(path_long, ARM_BASE_XY)

    candidates = [
        ("short", path_short, path_long, d_short, short_grip_ok),
        ("long", path_long, path_short, d_long, long_grip_ok),
    ]

    candidates.sort(key=lambda x: (0 if x[4] else 1, x[3]))

    dir_name, chosen, alt, _, chosen_ok = candidates[0]

    info["kind"] = "ARC"
    info["d_short"] = d_short
    info["d_long"] = d_long
    info["chosen"] = dir_name
    info["obs"] = blocking
    info["inflate"] = chosen_inflate
    info["gripper_ok"] = chosen_ok

    if not chosen_ok:
        ok, bad_wp, bad_obs = path_gripper_clear(chosen, obstacles)
        info["gripper_warn"] = (bad_wp, bad_obs)

    if verbose:
        ox, oy, r = blocking

        print("[plan] arc obs=({:.3f},{:.3f},r={:.3f}) inflate={:.3f}"
              .format(ox, oy, r, chosen_inflate))

        print("[plan] d_short={:.3f}(grip={}) d_long={:.3f}(grip={}) -> {}"
              .format(d_short, short_grip_ok, d_long, long_grip_ok, dir_name))

    return chosen, alt, info


# ======================================================================
# identify_GetTarget
# ======================================================================

class identify_GetTarget:
    def __init__(self):
        self.image = None
        self.xy = [90, 135]

        self.arm = Arm_Lib.Arm_Device()
        self.grap = identify_grap(arm=self.arm)

        self.n = rospy.init_node("dofbot_identify", anonymous=True)
        self.client = rospy.ServiceProxy("dofbot_kinemarics", kinemarics)

        self.tar_z = GRAB_Z
        self.tar_roll = -90.0
        self.tar_pitch = 0.0
        self.tar_yaw = 0.0

        self.target_geometry = {}
        self.color_obstacles = []

        self._frozen_obstacles = None
        self._ready_xy = None

        self.last_path = None
        self.last_alt_path = None
        self.preview_paths = []

        self._lock = threading.Lock()
        self._last_obs_count = -1

        model_path = "/home/yahboom/dofbot_ws/src/dofbot_color_identify/scripts/best.pt"

        self.yolo = YOLOFiberDetector(
            model_path=model_path,
            conf_thresh=0.45,
            iou_thresh=0.5,
            device="cpu"
        )

    # ------------------- color obstacle detection ---------------------

    def update_color_obstacles(self, image, verbose=False):
        if image is None:
            return

        h_img, _ = image.shape[:2]

        ignore_y_threshold = int(h_img * (1.0 - IGNORE_BOTTOM_RATIO))

        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

        kernel = np.ones((5, 5), np.uint8)

        mpp = meters_per_pixel()

        new_obs = []

        for color_name, ranges in COLOR_OBSTACLE_RANGES.items():
            mask = None

            for low, high in ranges:
                m = cv.inRange(
                    hsv,
                    np.array(low, dtype=np.uint8),
                    np.array(high, dtype=np.uint8)
                )

                mask = m if mask is None else cv.bitwise_or(mask, m)

            mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
            mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

            contours, _ = cv.findContours(
                mask,
                cv.RETR_EXTERNAL,
                cv.CHAIN_APPROX_SIMPLE
            )

            for cnt in contours:
                if cv.contourArea(cnt) < MIN_OBSTACLE_AREA_PX:
                    continue

                (cx, cy), r_px = cv.minEnclosingCircle(cnt)

                if cy > ignore_y_threshold:
                    continue

                wx, wy = pixel_to_world(cx, cy)
                wr = r_px * mpp

                if wr < 0.005 or wr > 0.05:
                    continue

                new_obs.append({
                    "world": [
                        float(wx),
                        float(wy),
                        float(wr * OBS_RADIUS_INFLATE)
                    ],
                    "pixel": (
                        int(cx),
                        int(cy),
                        int(r_px)
                    ),
                    "color": color_name,
                })

        with self._lock:
            self.color_obstacles = new_obs

        if verbose and len(new_obs) != self._last_obs_count:
            self._last_obs_count = len(new_obs)
            print("[obs] {} circle(s)".format(len(new_obs)))

    def clear_color_obstacles(self):
        with self._lock:
            self.color_obstacles = []

    def clear_preview_paths(self):
        with self._lock:
            self.preview_paths = []

    def get_obstacles_snapshot(self):
        with self._lock:
            return [o["world"][:] for o in self.color_obstacles]

    def get_active_obstacles(self):
        if self._frozen_obstacles is not None:
            return [o[:] for o in self._frozen_obstacles]

        return self.get_obstacles_snapshot()

    # ------------------- preview paths --------------------------------

    def update_preview_paths(self, targets_msg):
        obstacles = self.get_obstacles_snapshot()

        previews = []

        for name, pos in targets_msg.items():
            if pos is None:
                continue

            base_name = name.rsplit("_", 1)[0]

            servo1 = identify_grap.placement_servo1_for_class(base_name)

            placement_xy = identify_grap.placement_corridor_xy(servo1)

            target_xy = [pos[0], pos[1]]

            goal_grip_ok, _, _ = gripper_clear_at_tcp(target_xy, obstacles)

            path, alt, info = plan_around(
                target_xy,
                placement_xy,
                obstacles,
                verbose=False
            )

            display_path = (
                shrink_intermediate(path)
                if info["kind"] == "ARC"
                else path
            )

            color = CLASS_PREVIEW_COLOR.get(
                base_name,
                CLASS_PREVIEW_COLOR["default"]
            )

            previews.append({
                "name": name,
                "base": base_name,
                "path": display_path,
                "alt": alt,
                "color": color,
                "kind": info["kind"],
                "inflate": info["inflate"],
                "placement_xy": placement_xy,
                "target_xy": target_xy,
                "gripper_ok": info.get("gripper_ok", True),
                "goal_grip_ok": goal_grip_ok,
            })

        with self._lock:
            self.preview_paths = previews

    # ------------------- gripper rendering ----------------------------

    @staticmethod
    def _draw_gripper_at_tcp(image, tcp_xy, color=GRIPPER_BGR,
                             thick=3, draw_fork=False):
        wr = wrist_xy(tcp_xy)

        p_wr = world_to_pixel(wr[0], wr[1])
        p_tcp = world_to_pixel(tcp_xy[0], tcp_xy[1])

        cv.line(image, p_wr, p_tcp, color, thick, cv.LINE_AA)

        if draw_fork:
            dx = p_tcp[0] - p_wr[0]
            dy = p_tcp[1] - p_wr[1]

            n = math.hypot(dx, dy)

            if n > 1:
                ux = dx / n
                uy = dy / n

                px = -uy
                py = ux

                tine_len = 12
                tine_off = 6

                bx = p_tcp[0] - int(ux * 2)
                by = p_tcp[1] - int(uy * 2)

                lx0 = int(bx + px * tine_off)
                ly0 = int(by + py * tine_off)

                lx1 = int(lx0 + ux * tine_len)
                ly1 = int(ly0 + uy * tine_len)

                rx0 = int(bx - px * tine_off)
                ry0 = int(by - py * tine_off)

                rx1 = int(rx0 + ux * tine_len)
                ry1 = int(ry0 + uy * tine_len)

                cv.line(image, (lx0, ly0), (lx1, ly1),
                        GRIPPER_END_BGR, thick, cv.LINE_AA)

                cv.line(image, (rx0, ry0), (rx1, ry1),
                        GRIPPER_END_BGR, thick, cv.LINE_AA)

                cv.line(image, (lx0, ly0), (rx0, ry0),
                        GRIPPER_END_BGR, thick, cv.LINE_AA)

            cv.circle(image, p_wr, 4, GRIPPER_END_BGR, -1)

    def _draw_gripper_sweep(self, image, path, color=GRIPPER_BGR, stride=2):
        if not path:
            return

        for i, wp in enumerate(path):
            is_end = (i == 0 or i == len(path) - 1)

            if is_end or (i % stride == 0):
                self._draw_gripper_at_tcp(
                    image,
                    wp,
                    color,
                    thick=2 if not is_end else 3,
                    draw_fork=is_end
                )

    # ------------------- drawing --------------------------------------

    def draw_obstacles(self, image):
        with self._lock:
            obs_list = list(self.color_obstacles)
            path = list(self.last_path) if self.last_path else None
            alt_path = list(self.last_alt_path) if self.last_alt_path else None
            previews = list(self.preview_paths)

        mpp = meters_per_pixel()

        for o in obs_list:
            cx, cy, r_px = o["pixel"]

            color_bgr = {
                "red": (0, 0, 255),
                "blue": (255, 0, 0),
                "green": (0, 200, 0),
            }.get(o["color"], (0, 255, 255))

            cv.circle(image, (cx, cy), r_px, color_bgr, 2)
            cv.circle(image, (cx, cy), 4, color_bgr, -1)

            r_safe_px = int((o["world"][2] + SAFETY_MARGIN) / mpp)
            cv.circle(image, (cx, cy), r_safe_px, (0, 255, 255), 1)

            r_grip_px = int((o["world"][2] + GRIPPER_LATERAL_MARGIN) / mpp)
            cv.circle(image, (cx, cy), r_grip_px, GRIPPER_END_BGR, 1)

            cv.putText(
                image,
                "OBS:" + o["color"],
                (cx - 40, cy - r_px - 12),
                cv.FONT_HERSHEY_SIMPLEX,
                0.45,
                color_bgr,
                1
            )

        if path is None:
            for pv in previews:
                if not pv["path"] or len(pv["path"]) < 2:
                    continue

                pts = [
                    world_to_pixel(p[0], p[1])
                    for p in pv["path"]
                ]

                if not pv.get("goal_grip_ok", True):
                    color = (40, 40, 220)
                elif not pv.get("gripper_ok", True):
                    color = (60, 90, 230)
                else:
                    color = pv["color"]

                for a, b in zip(pts[:-1], pts[1:]):
                    cv.line(image, a, b, color, 2, cv.LINE_AA)

                for p in pts[1:-1]:
                    cv.circle(image, p, 3, color, -1)

                cv.circle(image, pts[0], 8, color, 2)
                cv.circle(image, pts[0], 2, color, -1)

                ex, ey = pts[-1]

                cv.rectangle(
                    image,
                    (ex - 7, ey - 7),
                    (ex + 7, ey + 7),
                    color,
                    2
                )

                self._draw_gripper_sweep(image, pv["path"])

                tag = "{} [{}]".format(pv["name"], pv["kind"])

                if pv["kind"] == "ARC":
                    tag += " inf={:.2f} shr={:.0f}mm".format(
                        pv["inflate"],
                        WAYPOINT_SHRINK * 1000
                    )

                if not pv.get("goal_grip_ok", True):
                    tag += " GRIPPER!"
                elif not pv.get("gripper_ok", True):
                    tag += " grip-risk"

                cv.putText(
                    image,
                    tag,
                    (pts[0][0] + 10, pts[0][1] - 10),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv.LINE_AA
                )

        if alt_path is not None and len(alt_path) >= 2:
            pts = [
                world_to_pixel(p[0], p[1])
                for p in alt_path
            ]

            for a, b in zip(pts[:-1], pts[1:]):
                cv.line(image, a, b, (140, 140, 140), 1, cv.LINE_AA)

        if path is not None and len(path) >= 2:
            pts = [
                world_to_pixel(p[0], p[1])
                for p in path
            ]

            for a, b in zip(pts[:-1], pts[1:]):
                cv.line(image, a, b, (0, 255, 255), 2)

            for p in pts:
                cv.circle(image, p, 4, (0, 255, 255), -1)

            self._draw_gripper_sweep(image, path)

        bp = world_to_pixel(*ARM_BASE_XY)

        if 0 <= bp[0] < WORKSPACE["img_w"] and 0 <= bp[1] < WORKSPACE["img_h"]:
            cv.drawMarker(
                image,
                bp,
                (255, 0, 255),
                cv.MARKER_CROSS,
                18,
                2
            )

        cv.putText(
            image,
            "GRIP L={:.0f}mm lat={:.0f}mm safe={:.0f}mm shrink={:.0f}mm"
            .format(
                GRIPPER_LENGTH * 1000,
                GRIPPER_LATERAL_MARGIN * 1000,
                SAFETY_MARGIN * 1000,
                WAYPOINT_SHRINK * 1000
            ),
            (28, 460),
            cv.FONT_HERSHEY_SIMPLEX,
            0.45,
            GRIPPER_BGR,
            1,
            cv.LINE_AA
        )

        cv.line(image, (10, 456), (24, 456), GRIPPER_BGR, 3, cv.LINE_AA)

        return image

    # ------------------- target selection -----------------------------

    def select_targets(self, image, target_classes=None):
        self.image = cv.resize(image, (640, 480))

        msg = {}

        self.target_geometry = {}

        self.update_color_obstacles(self.image, verbose=False)

        detections = self.yolo.detect(self.image)

        if target_classes is not None:
            target_dets = [
                d for d in detections
                if d["class_name"] in target_classes
            ]
        else:
            target_dets = detections

        target_dets.sort(key=lambda d: d["confidence"], reverse=True)

        class_counter = {}

        for det in target_dets:
            name = det["class_name"]

            class_counter[name] = class_counter.get(name, 0)

            key = "{}_{}".format(name, class_counter[name])

            class_counter[name] += 1

            msg[key] = det["pos"]

            self.target_geometry[key] = {
                "area": det["area"],
                "confidence": det["confidence"],
            }

        if self._frozen_obstacles is None:
            self.update_preview_paths(msg)

        self.image = self.yolo.draw_detections(self.image, detections)
        self.image = self.draw_obstacles(self.image)

        return self.image, msg

    # ------------------- helpers --------------------------------------

    def _is_goal_safe(self, goal_xy, obstacles,
                      margin=SAFETY_MARGIN + 0.005):
        for ox, oy, r in obstacles:
            if math.hypot(goal_xy[0] - ox, goal_xy[1] - oy) < (r + margin):
                return False, (ox, oy, r)

        return True, None

    def _is_goal_gripper_safe(self, goal_xy, obstacles):
        ok, bad, _ = gripper_clear_at_tcp(goal_xy, obstacles)

        return ok, bad

    def _pick_safe_dropdown(self, obstacles):
        candidates = [
            [0.000, 0.250],
            [0.000, WORKSPACE["y_far"] - 0.01],
            [0.000, WORKSPACE["y_near"] + 0.01],
            [WORKSPACE["x_min"] + 0.01, 0.250],
            [WORKSPACE["x_max"] - 0.01, 0.250],
            [WORKSPACE["x_min"] + 0.01, WORKSPACE["y_far"] - 0.01],
            [WORKSPACE["x_max"] - 0.01, WORKSPACE["y_far"] - 0.01],
        ]

        clearance = SAFETY_MARGIN + 0.01

        for c in candidates:
            ok = True

            for ox, oy, r in obstacles:
                if math.hypot(c[0] - ox, c[1] - oy) < (r + clearance):
                    ok = False
                    break

            if not ok:
                continue

            grip_ok, _, _ = gripper_clear_at_tcp(c, obstacles)

            if grip_ok:
                return c

        print("  [warn] no clean dropdown found, using default (0, 0.25)")

        return [0.000, 0.250]

    # ------------------- main XY motion with avoidance ----------------

    def move_xy_with_avoidance(self, start_xy, goal_xy, obstacles,
                               servo5_target, travel_z=TRAVEL_Z,
                               gripper=30, stage_tag="approach"):

        goal_grip_ok, goal_bad = self._is_goal_gripper_safe(goal_xy, obstacles)

        if not goal_grip_ok:
            print("  [{}][error] goal ({:.3f},{:.3f}) gripper clips obs {}; skip."
                  .format(stage_tag, goal_xy[0], goal_xy[1], goal_bad))
            return False

        effective_start = list(start_xy)

        for ox, oy, r in obstacles:
            d = math.hypot(
                effective_start[0] - ox,
                effective_start[1] - oy
            )

            forbid = r + SAFETY_MARGIN

            if d < forbid:
                if d < 1e-6:
                    vx, vy = 0.0, -1.0
                else:
                    vx = (effective_start[0] - ox) / d
                    vy = (effective_start[1] - oy) / d

                push = forbid + DETOUR_BONUS

                effective_start = [
                    ox + vx * push,
                    oy + vy * push
                ]

                print("  [{}][escape] start in obstacle zone, "
                      "shifted to ({:.3f},{:.3f})".format(
                          stage_tag,
                          effective_start[0],
                          effective_start[1]
                      ))

                break

        path, alt_path, info = plan_around(
            effective_start,
            goal_xy,
            obstacles,
            verbose=True
        )

        if path is None or len(path) < 2:
            print("  [{}][error] no path".format(stage_tag))
            return False

        executed_path = (
            shrink_intermediate(path)
            if info["kind"] == "ARC"
            else [p[:] for p in path]
        )

        # v8 smooth-place patch 1:
        # 只对持物放置阶段生效。
        # 原 ARC 路径长度一般是 17，其中最后一个点是 corridor_xy 目标点，
        # 该点没有 shrink，容易从倒数第2个点突然前冲到放置走廊。
        # 删除这个最后点后，避障结束在弧线出口附近，后续由 place rotate 的高位姿态慢速过渡。
        if (DROP_PLACE_ARC_FINAL_GOAL and
                info["kind"] == "ARC" and
                str(stage_tag).startswith("place transit") and
                len(executed_path) >= 3):
            last_wp = executed_path[-1]
            prev_wp = executed_path[-2]
            final_jump = math.hypot(last_wp[0] - prev_wp[0],
                                    last_wp[1] - prev_wp[1])

            if final_jump > DROP_PLACE_ARC_FINAL_THRESHOLD:
                print("  [{}][smooth] drop final corridor wp "
                      "({:.3f},{:.3f}); prev=({:.3f},{:.3f}); "
                      "jump={:.3f}m".format(
                          stage_tag,
                          last_wp[0], last_wp[1],
                          prev_wp[0], prev_wp[1],
                          final_jump
                      ))
                executed_path = executed_path[:-1]

        # v8 smooth-place patch 2:
        # 抓取结束后开始避障时，start_xy -> shrink 后入口点可能跨度较大，
        # 机械臂会表现为突然向后避障。这里把起始第一段拆成多个小段。
        if (SMOOTH_PLACE_ARC_ENTRY and
                info["kind"] == "ARC" and
                str(stage_tag).startswith("place transit") and
                len(executed_path) >= 2):
            executed_path, entry_jump, inserted_count = densify_first_segment(
                executed_path,
                max_step=SMOOTH_PLACE_ARC_ENTRY_MAX_STEP,
                threshold=SMOOTH_PLACE_ARC_ENTRY_THRESHOLD
            )

            if inserted_count > 0:
                print("  [{}][smooth] densify entry segment: "
                      "jump={:.3f}m, inserted={} wp, max_step={:.0f}mm".format(
                          stage_tag,
                          entry_jump,
                          inserted_count,
                          SMOOTH_PLACE_ARC_ENTRY_MAX_STEP * 1000
                      ))

        with self._lock:
            self.last_path = [p[:] for p in executed_path]
            self.last_alt_path = [p[:] for p in alt_path] if alt_path else None

        print("\n  [{}][plan] kind={}  obs_count={}  "
              "start=({:.3f},{:.3f}) -> goal=({:.3f},{:.3f})  "
              "z={:.3f} grip={}  waypoints={}  shrink={:.0f}mm".format(
                  stage_tag,
                  info["kind"],
                  len(obstacles),
                  effective_start[0],
                  effective_start[1],
                  goal_xy[0],
                  goal_xy[1],
                  travel_z,
                  gripper,
                  len(executed_path),
                  WAYPOINT_SHRINK * 1000 if info["kind"] == "ARC" else 0
              ))

        if info["kind"] == "ARC":
            ox, oy, r = info["obs"]

            print("  [{}][plan] obs ({:.3f},{:.3f},r={:.3f})  "
                  "inflate={:.3f}  d_short={:.4f}  d_long={:.4f}  "
                  "chosen={}  gripper_ok={}".format(
                      stage_tag,
                      ox,
                      oy,
                      r,
                      info["inflate"],
                      info["d_short"],
                      info["d_long"],
                      info["chosen"],
                      info["gripper_ok"]
                  ))

            if not info["gripper_ok"]:
                print("  [{}][WARN] gripper line still clips an obstacle"
                      .format(stage_tag))

        s5 = int(max(0, min(270, servo5_target)))
        g = int(max(0, min(180, gripper)))

        is_arc = info["kind"] == "ARC"

        first_t = 1800
        step_t = 600 if is_arc else 1200

        successful = 0
        skipped = 0

        last_idx = len(executed_path) - 1

        for i, wp in enumerate(executed_path):
            joints = self.server_joint(wp, tar_z=travel_z)

            if joints is None:
                joints = self.server_joint(wp, tar_z=travel_z + 0.02)

            if joints is None:
                joints = self.server_joint(wp, tar_z=travel_z + 0.04)

            if joints is None:
                skipped += 1

                print("  [{}][skip wp {}/{}] IK fail at ({:.3f},{:.3f})"
                      .format(
                          stage_tag,
                          i + 1,
                          len(executed_path),
                          wp[0],
                          wp[1]
                      ))

                continue

            if i == 0:
                move_t = first_t
            elif ADAPTIVE_ARC_STEP_TIME and is_arc:
                prev_wp = executed_path[i - 1]
                seg_d = math.hypot(wp[0] - prev_wp[0], wp[1] - prev_wp[1])
                move_t = int(ADAPTIVE_ARC_STEP_MIN_MS +
                             seg_d * ADAPTIVE_ARC_STEP_MS_PER_M)
                move_t = max(ADAPTIVE_ARC_STEP_MIN_MS,
                             min(ADAPTIVE_ARC_STEP_MAX_MS, move_t))
            else:
                move_t = step_t

            servo_cmd = [
                joints[0],
                joints[1],
                joints[2],
                joints[3],
                s5,
                g
            ]

            self.arm.Arm_serial_servo_write6_array(servo_cmd, move_t)

            sleep(move_t / 1000.0 + 0.1)

            successful += 1

            tag_extra = ""

            if is_arc and 0 < i < last_idx:
                if i < len(path):
                    orig = path[i]

                    tag_extra = " (shrunk/smooth: {:.3f},{:.3f}->{:.3f},{:.3f})".format(
                        orig[0],
                        orig[1],
                        wp[0],
                        wp[1]
                    )
                else:
                    tag_extra = " (smooth inserted)"

            print("  [{}][reach wp {}/{}] ({:.3f},{:.3f}) j1={:.0f} "
                  "j2={:.0f} j3={:.0f} j4={:.0f} t={}ms{}".format(
                      stage_tag,
                      i + 1,
                      len(executed_path),
                      wp[0],
                      wp[1],
                      joints[0],
                      joints[1],
                      joints[2],
                      joints[3],
                      move_t,
                      tag_extra
                  ))

        print("  [{}][summary] reached {}/{} skipped {}\n".format(
            stage_tag,
            successful,
            len(executed_path),
            skipped
        ))

        return successful > 0

    # ------------------- top-level grab loop --------------------------

    def target_run(self, msg, xy=None):
        if xy is not None:
            self.xy = xy

        valid_targets = {
            n: p for n, p in msg.items()
            if p is not None
        }

        if not valid_targets:
            return

        obstacles = self.get_obstacles_snapshot()

        self._frozen_obstacles = [o[:] for o in obstacles]

        self.clear_preview_paths()

        self.grap.move_status = True

        with self._lock:
            self.last_path = None
            self.last_alt_path = None

        print("\n[target_run] {} obstacles (frozen):".format(len(obstacles)))

        for i, (ox, oy, r) in enumerate(obstacles):
            print("   obs[{}] = ({:.3f}, {:.3f}, r={:.3f})"
                  .format(i, ox, oy, r))

        try:
            self.arm.Arm_Buzzer_On(1)
            sleep(0.3)
            self.arm.Arm_Buzzer_On(0)
        except Exception:
            pass

        target_list = list(valid_targets.items())

        total = len(target_list)

        staging_pos = self._pick_safe_dropdown(obstacles)

        self._ready_xy = staging_pos[:]

        print("[target_run] staging / ready XY = ({:.3f}, {:.3f})"
              .format(staging_pos[0], staging_pos[1]))

        try:
            for idx, (name, pos) in enumerate(target_list):
                try:
                    goal_pos = [pos[0], pos[1]]

                    angle = pos[2] if len(pos) >= 3 else 0.0

                    ok, bad = self._is_goal_safe(goal_pos, obstacles)

                    if not ok:
                        print("  [skip] {}: goal inside obs {}".format(name, bad))
                        continue

                    grip_ok, grip_bad = self._is_goal_gripper_safe(
                        goal_pos,
                        obstacles
                    )

                    if not grip_ok:
                        print("  [skip] {}: gripper line at goal clips obs {}"
                              .format(name, grip_bad))
                        continue

                    print("\n[target {}/{}] {} goal=({:.3f},{:.3f})"
                          .format(
                              idx + 1,
                              total,
                              name,
                              goal_pos[0],
                              goal_pos[1]
                          ))

                    final_joints = self.server_joint(goal_pos)

                    if final_joints is None:
                        print("  [skip] {}: final IK fail".format(name))
                        continue

                    servo5_target = self.grap.calc_servo5(
                        angle,
                        final_joints[0]
                    )

                    tag = "approach[{}]".format(name)

                    if not self.move_xy_with_avoidance(
                        staging_pos,
                        goal_pos,
                        obstacles,
                        servo5_target,
                        gripper=30,
                        stage_tag=tag
                    ):
                        print("  [skip] {}: approach failed".format(name))
                        continue

                    is_last = idx == total - 1

                    self.grap.identify_move(
                        name,
                        final_joints,
                        angle,
                        is_last,
                        path_followed=True,
                        parent=self,
                        target_xy=goal_pos,
                        home_xy=self.xy,
                        ready_xy=staging_pos
                    )

                except Exception as e:
                    print("  trajectory error: {}".format(e))
                    self.grap.move_status = True

        finally:
            try:
                hx, hy = self.xy[0], self.xy[1]

                # 最终正确停止位姿
                # 不再执行 [hx, hy, 30, 0, 90, 60]
                self.arm.Arm_serial_servo_write6_array(
                    [hx, hy, 0, 0, 90, 30],
                    800
                )

                sleep(0.6)

                print("  [target_run] final return to correct camera/start pose "
                      "[{}, {}, 0, 0, 90, 30]".format(hx, hy))

            except Exception as e:
                print("  [target_run] homing error: {}".format(e))

            with self._lock:
                self.last_path = None
                self.last_alt_path = None

            self._frozen_obstacles = None
            self._ready_xy = None
            self.grap.move_status = True

            print("[target_run] state cleaned, ready for next grab")

    # ------------------- IK -------------------------------------------

    def server_joint(self, posxy, tar_z=None):
        self.client.wait_for_service()

        request = kinemaricsRequest()

        request.tar_x = posxy[0]
        request.tar_y = posxy[1]
        request.tar_z = self.tar_z if tar_z is None else tar_z

        request.Roll = self.tar_roll
        request.Pitch = self.tar_pitch
        request.Yaw = self.tar_yaw

        request.cur_joint1 = 0.0
        request.cur_joint2 = 0.0
        request.cur_joint3 = 0.0
        request.cur_joint4 = 0.0
        request.cur_joint5 = 0.0
        request.cur_joint6 = 0.0

        request.kin_name = "ik"

        try:
            response = self.client.call(request)

            if isinstance(response, kinemaricsResponse):
                joints = [
                    response.joint1,
                    response.joint2,
                    response.joint3,
                    response.joint4,
                    response.joint5
                ]

                if joints[2] < 0:
                    joints[1] += joints[2] * 3 / 5
                    joints[3] += joints[2] * 3 / 5
                    joints[2] = 0

                for i in range(5):
                    v = joints[i]

                    if v < -3.0 or v > 183.0:
                        print("   [IK] joint{}={:.1f} OOB at "
                              "({:.3f},{:.3f},{:.3f})".format(
                                  i + 1,
                                  v,
                                  posxy[0],
                                  posxy[1],
                                  request.tar_z
                              ))
                        return None

                    joints[i] = max(0.0, min(180.0, v))

                return joints

        except Exception:
            rospy.loginfo("arg error")

        return None