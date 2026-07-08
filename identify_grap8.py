#!/usr/bin/env python
# coding: utf-8
"""
identify_grap8.py

适配：
- main_code8.py
- identify_target8.py
- RDK X5 + Docker dofbot_noetic + DOFBOT SE
+
30+
功能：
- pick / place 抓取放置
- 抓取点 -> 放置走廊：持物搬运阶段调用 parent.move_xy_with_avoidance() 走避障
- 支持 target8 中的 ready_xy / staging XY
- 复位段采用确定性动作：
    1) 放置侧垂直抬起
    2) 垂直状态转回中心
    3) 不再回到 ready/pre-grasp 位姿，避免归位阶段再次向前伸
- plastic / box / leaf 三类目标统一放到 box 的放置位置
- move_status 用 try/finally 兜底，避免上一轮异常后状态卡死
- 避障抓取后不再先走固定 lift local，改为从目标同 XY 平滑进入避障
"""

import math
import Arm_Lib
from time import sleep


class identify_grap:
    def __init__(self, arm=None):
        self.move_status = True
        self.arm = arm if arm is not None else Arm_Lib.Arm_Device()

        # DOFBOT SE 夹爪角度约定：
        # 30  = 张开
        # 180 = 闭合/夹持
        self.grap_joint = 180

        self.GRIPPER_OPEN = 30
        self.GRIPPER_READY = 60

        # 持物搬运时的 XY 移动高度。
        # 如果 identify_target8.py 的 move_xy_with_avoidance 支持 travel_z，
        # 会使用这个高度；否则自动回退到它自己的默认高度。
        self.CARRY_TRAVEL_Z = 0.210

    def calc_servo5(self, tilt_angle, servo1_angle):
        """
        根据目标倾角和底座 servo1 角度，计算夹爪旋转 servo5。
        servo5 范围：0~270。
        """
        relative_angle = tilt_angle - (servo1_angle - 90)

        while relative_angle > 90:
            relative_angle -= 180
        while relative_angle < -90:
            relative_angle += 180

        servo5 = 265 - relative_angle

        if servo5 > 270:
            servo5 -= 180
        elif servo5 < 0:
            servo5 += 180

        return int(max(0, min(270, servo5)))

    @staticmethod
    def placement_servo1_for_class(base_name):
        """
        三类杂质统一放到 box 的放置位置。
        box 原始放置 servo1 = 45。
        """
        if base_name in ("plastic", "box", "leaf"):
            return 45
        else:
            return 45

    @staticmethod
    def placement_corridor_xy(placement_servo1):
        """
        放置前的安全走廊 XY 点。
        这个点用于持物搬运阶段的中间目标点。
        """
        if placement_servo1 < 60:
            return [0.160, 0.200]
        elif placement_servo1 > 120:
            return [-0.060, 0.200]
        else:
            return [0.000, 0.200]

    @staticmethod
    def _placement_corridor_xy(placement_servo1):
        return identify_grap.placement_corridor_xy(placement_servo1)

    @staticmethod
    def _safe_xy_transit(parent, start_xy, goal_xy, servo5_for_grip,
                         tag="transit", gripper=180, travel_z=0.210):
        """
        持物搬运阶段避障转移。

        parent 一般是 identify_target8.identify_GetTarget 实例。
        障碍物来自 parent.get_active_obstacles()。
        target_run 期间 get_active_obstacles() 会返回冻结快照，避免机械臂遮挡导致障碍物变化。

        返回：
        - True：执行了避障路径
        - False：没有 parent、没有障碍物，或者避障失败
        """
        if parent is None:
            return False

        obstacles = parent.get_active_obstacles()

        if not obstacles:
            print("  [{}] no obstacles, skip avoidance".format(tag))
            return False

        print("  [{}] avoiding {} obstacles grip={}: "
              "({:.3f},{:.3f}) -> ({:.3f},{:.3f})".format(
                  tag,
                  len(obstacles),
                  gripper,
                  start_xy[0],
                  start_xy[1],
                  goal_xy[0],
                  goal_xy[1]
              ))

        # 兼容两种 identify_target8.py：
        # 1) move_xy_with_avoidance(..., travel_z=..., gripper=..., stage_tag=...)
        # 2) move_xy_with_avoidance(..., travel_z=..., gripper=...)
        try:
            return parent.move_xy_with_avoidance(
                start_xy,
                goal_xy,
                obstacles,
                servo5_target=servo5_for_grip,
                travel_z=travel_z,
                gripper=gripper,
                stage_tag=tag,
            )
        except TypeError:
            return parent.move_xy_with_avoidance(
                start_xy,
                goal_xy,
                obstacles,
                servo5_target=servo5_for_grip,
                travel_z=travel_z,
                gripper=gripper,
            )

    @staticmethod
    def _clamp(v, lo, hi):
        return max(lo, min(hi, v))

    def _get_ready_pose(self, parent, ready_xy, home_xy):
        """
        根据 ready_xy 计算复位后的准备抓取位姿。
        优先使用 IK，失败则回退到垂直安全位姿。
        """
        hx, hy = home_xy[0], home_xy[1]

        ready_pose = None

        if parent is not None and ready_xy is not None:
            ready_joints = None

            # 优先用较高的 travel_z 回 ready/pre-grasp，避免低位横扫。
            try:
                ready_joints = parent.server_joint(ready_xy, tar_z=0.235)
            except TypeError:
                try:
                    ready_joints = parent.server_joint(ready_xy)
                except Exception:
                    ready_joints = None
            except Exception:
                ready_joints = None

            # 如果高位 IK 失败，再试默认抓取高度。
            if ready_joints is None:
                try:
                    ready_joints = parent.server_joint(ready_xy)
                except Exception:
                    ready_joints = None

            if ready_joints is not None:
                ready_pose = [
                    ready_joints[0],
                    ready_joints[1],
                    ready_joints[2],
                    ready_joints[3],
                    90,
                    self.GRIPPER_READY,
                ]

        if ready_pose is None:
            print("  [return][warn] ready_xy IK failed, fallback to vertical start pose")
            ready_pose = [hx, hy, 0, 0, 90, self.GRIPPER_READY]

        return ready_pose

    def move(self, joints, joints_down, servo5_angle=265,
             is_last=True, skip_transit=False,
             parent=None, target_xy=None, place_name="default",
             home_xy=None, ready_xy=None):
        """
        完整抓取动作：

        1. 到目标点
        2. 张开夹爪
        3. 下探抓取
        4. 闭合夹爪
        5. 抬起
        6. 持物避障移动到放置走廊
        7. 分类放置
        8. 垂直抬起、转回中心、回 ready_xy
        """
        if home_xy is None:
            home_xy = [90, 130]

        hx, hy = home_xy[0], home_xy[1]

        servo5_angle = int(self._clamp(servo5_angle, 0, 270))

        joints_uu_local = [
            joints[0],
            80,
            50,
            50,
            servo5_angle,
            self.grap_joint,
        ]

        safe_above = [
            joints_down[0],
            80,
            50,
            50,
            servo5_angle,
            self.grap_joint,
        ]

        joints_up = [
            joints_down[0],
            80,
            50,
            50,
            265,
            self.GRIPPER_OPEN,
        ]

        joints_uu_center = [
            hx,
            80,
            50,
            50,
            servo5_angle,
            self.grap_joint,
        ]

        # 放置侧垂直抬起 -> 垂直转回中心
        vertical_at_place = [
            joints_down[0],
            hy,
            0,
            0,
            90,
            self.GRIPPER_OPEN,
        ]

        vertical_center = [
            hx,
            hy,
            0,
            0,
            90,
            self.GRIPPER_OPEN,
        ]

        # 本版不再计算 ready_pose，避免最后回到前方预抓取位姿。

        if not skip_transit:
            self.arm.Arm_serial_servo_write6_array(joints_uu_center, 1000)
            sleep(1.0)

        # ---------------- pick：抓取 ----------------
        print("  [pick] open gripper")
        self.arm.Arm_serial_servo_write(6, self.GRIPPER_OPEN, 500)
        sleep(0.5)

        print("  [pick] move to target")
        self.arm.Arm_serial_servo_write6_array(joints, 800)
        sleep(0.8)

        # 轻微下探/贴近目标，避免刚好悬空夹不到。
        servo4_down = int(self._clamp(joints[3] + 5, 0, 180))
        self.arm.Arm_serial_servo_write(4, servo4_down, 500)
        sleep(0.5)

        print("  [pick] close gripper")
        self.arm.Arm_serial_servo_write(6, self.grap_joint, 500)
        sleep(0.5)

        # v8 smooth-start patch:
        # 如果后面需要持物避障，不再先执行固定的 joints_uu_local 高位姿态。
        # 原来的 joints_uu_local = [joints[0], 80, 50, 50, ...] 会让机械臂
        # 抓住目标后先向前僵直一下，然后避障路径又从目标 XY 开始，造成“先前冲、再后撤”的突变。
        # 当前策略：有障碍物时直接进入 move_xy_with_avoidance()，它的第 1 个 waypoint
        # 是 target_xy 在 travel_z 高度下的同点抬升，然后再平滑进入弧线避障。
        has_carry_obstacles = False
        if parent is not None:
            try:
                has_carry_obstacles = bool(parent.get_active_obstacles())
            except Exception:
                has_carry_obstacles = False

        if has_carry_obstacles and target_xy is not None:
            print("  [pick] skip fixed lift local; smooth avoidance starts from same XY")
            sleep(0.2)
        else:
            print("  [pick] lift local")
            self.arm.Arm_serial_servo_write6_array(joints_uu_local, 1000)
            sleep(1.0)

        # ---------------- carry：持物搬运避障 ----------------
        corridor_xy = self.placement_corridor_xy(joints_down[0])

        if parent is not None and target_xy is not None:
            self._safe_xy_transit(
                parent,
                target_xy,
                corridor_xy,
                servo5_for_grip=servo5_angle,
                tag="place transit-1",
                gripper=self.grap_joint,
                travel_z=self.CARRY_TRAVEL_Z,
            )

        # ---------------- place：放置 ----------------
        print("  [place rotate] servo1 -> {} (high)".format(joints_down[0]))
        self.arm.Arm_serial_servo_write6_array(safe_above, 1200)
        sleep(1.2)

        print("  [place] down")
        self.arm.Arm_serial_servo_write6_array(joints_down, 1000)
        sleep(1.0)

        print("  [place] open gripper")
        self.arm.Arm_serial_servo_write(6, self.GRIPPER_OPEN, 500)
        sleep(0.5)

        print("  [place] lift")
        self.arm.Arm_serial_servo_write6_array(joints_up, 1000)
        sleep(1.0)

        # ---------------- return：安全复位 ----------------
        print("  [return] 1/3 lift vertical at place side")
        self.arm.Arm_serial_servo_write6_array(vertical_at_place, 1000)
        sleep(1.0)

        print("  [return] 2/3 rotate to center (still vertical)")
        self.arm.Arm_serial_servo_write6_array(vertical_center, 1000)
        sleep(1.0)

        # 关键修改：
        # 删除原来的 back to ready/pre-grasp pose 动作。
        # ready_pose 是根据 staging/ready_xy 通过 IK 计算出的前方预抓取位姿，
        # 容易导致机械臂归位阶段再次向前伸。
        # 当前处理：保持 vertical_center 垂直中心姿态，
        # 然后由 identify_target8.target_run() 的 final return 回到相机/起始位姿。
        print("  [return] 3/3 skip ready/pre-grasp pose, keep vertical center")
        sleep(0.5)

    def identify_move(self, name, joints, angle=0.0,
                      is_last=True, path_followed=False,
                      parent=None, target_xy=None, home_xy=None,
                      ready_xy=None):
        """
        identify_target8.py 会调用这个函数。

        参数：
        - name: plastic_0 / box_0 / leaf_0
        - joints: IK 求出的目标关节角
        - angle: YOLO 检测到的目标倾角
        - path_followed: True 表示 approach 阶段已经由 target8 避障走到目标附近
        - parent: identify_GetTarget 实例，用于调用 move_xy_with_avoidance()
        - target_xy: 当前抓取目标 XY
        - home_xy: 相机观察/启动位姿中的 servo1、servo2
        - ready_xy: 本轮 staging / ready XY
        """
        base_name = name.rsplit("_", 1)[0]

        servo5 = self.calc_servo5(angle, joints[0])

        # 自愈：如果上一轮异常导致状态卡在 False，强制恢复。
        if not self.move_status:
            print("  [warn] move_status stuck False, force reset before grab")
            self.move_status = True

        self.move_status = False

        try:
            servo1_place = self.placement_servo1_for_class(base_name)

            # 三类杂质统一放到 box 的放置位置。
            # box 原始放置姿态：[45, 50, 20, 60, 265, close]
            joints_down = [
                45,
                50,
                20,
                60,
                265,
                self.grap_joint,
            ]

            # 抓取姿态微调：
            # joint3 略减、joint4 略增，使末端更靠近目标。
            j1 = int(self._clamp(joints[0], 0, 180))
            j2 = int(self._clamp(joints[1], 0, 180))
            j3 = int(self._clamp(joints[2] - 13, 0, 180))
            j4 = int(self._clamp(joints[3] + 18, 0, 180))

            joints_cmd = [
                j1,
                j2,
                j3,
                j4,
                servo5,
                self.GRIPPER_OPEN,
            ]

            print("  [grap8-same-box-no-ready] {} ({})  path_followed={}  servo5={}  avoid_carry={}"
                  .format(
                      name,
                      base_name,
                      path_followed,
                      servo5,
                      parent is not None
                  ))

            self.move(
                joints_cmd,
                joints_down,
                servo5_angle=servo5,
                is_last=is_last,
                skip_transit=path_followed,
                parent=parent,
                target_xy=target_xy,
                place_name=base_name,
                home_xy=home_xy,
                ready_xy=ready_xy,
            )

        finally:
            # 无论是否异常，都恢复状态机，避免下一轮无法抓取。
            self.move_status = True

