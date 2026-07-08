#!/usr/bin/env python3
# coding: utf-8
"""
main_code8.py
RDK X5 Docker web-control runner for identify_target8.py + identify_grap8.py.

默认：
- HEADLESS=1：不弹 OpenCV 窗口，适合 Docker/SSH 运行。
- WEB_UI=1：开启浏览器界面，访问 http://RDK_IP:8080/

浏览器界面功能：
- 实时显示摄像头识别画面、障碍物、安全圈、路径预览。
- 选择 plastic / box / leaf。
- 开启检测、开关障碍物、抓取当前目标、一键检测并抓取。
"""

import os
import json
import cv2 as cv
import threading
import numpy as np
from time import sleep
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from dofbot_config import *
from identify_target8 import identify_GetTarget
import Arm_Lib


HEADLESS = os.environ.get("HEADLESS", "1").strip().lower() not in ("0", "false", "no", "off")
WEB_UI = os.environ.get("WEB_UI", "1").strip().lower() not in ("0", "false", "no", "off")
WEB_PORT = int(os.environ.get("WEB_PORT", "8080"))
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))
CAMERA_DEVICE = os.environ.get("CAMERA_DEVICE", "")

VALID_CLASSES = ["leaf", "plastic", "box"]


target = identify_GetTarget()
calibration = Arm_Calibration()
arm = target.arm

xy = [90, 130]
msg = {}
target_classes = []

is_grabbing = False
grap_request = False
detect_enabled = False
obstacle_enabled = True
exit_flag = False
camera_ok = False
last_error = ""

# 浏览器预览帧缓存。网页不会重新打开摄像头，只显示主程序正在处理的图像。
latest_frame = None
frame_lock = threading.Lock()

XYT_path = "/home/yahboom/dofbot_ws/src/dofbot_color_identify/scripts/XYT_config.txt"

dp = [
    np.array([110, 34],  dtype=np.float32),
    np.array([34, 427],  dtype=np.float32),
    np.array([625, 443], dtype=np.float32),
    np.array([554, 47],  dtype=np.float32)
]

try:
    xy, thresh = read_XYT(XYT_path)
except Exception as e:
    print("Read XYT_config Error:", e)

sleep(1)

# 程序启动 / 打开摄像头前的观察位姿
joints_0 = [xy[0], xy[1], 30, 0, 90, 60]
arm.Arm_serial_servo_write6_array(joints_0, 1000)
sleep(1)

try:
    arm.Arm_Buzzer_On(0)
except Exception:
    pass

try:
    arm.Arm_Beep(0)
except Exception:
    pass


def run_grap_safe(msg_in, xy_in):
    global is_grabbing, grap_request, detect_enabled, msg, last_error
    try:
        target.target_run(msg_in, xy_in)
    except Exception as e:
        last_error = "grab failed: {}".format(e)
        print("[error] grab failed:", e)
    finally:
        try:
            target.grap.move_status = True
        except Exception:
            pass

        is_grabbing = False
        grap_request = False
        detect_enabled = False
        msg.clear()
        print("[done] grab finished, back to idle")


def _open_camera():
    """优先用 CAMERA_DEVICE=/dev/videoX，否则用 CAMERA_INDEX。"""
    if CAMERA_DEVICE:
        print("[camera] opening", CAMERA_DEVICE)
        cap = cv.VideoCapture(CAMERA_DEVICE, cv.CAP_V4L2)
    else:
        print("[camera] opening index", CAMERA_INDEX)
        cap = cv.VideoCapture(CAMERA_INDEX, cv.CAP_V4L2)

    if cap.isOpened():
        cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv.CAP_PROP_FPS, 30)
    return cap


def _state_dict():
    return {
        "classes": list(target_classes),
        "detect": bool(detect_enabled),
        "obstacle": bool(obstacle_enabled),
        "grabbing": bool(is_grabbing),
        "targets": list(msg.keys()) if msg else [],
        "obstacles": len(target.color_obstacles),
        "headless": bool(HEADLESS),
        "web_ui": bool(WEB_UI),
        "camera_ok": bool(camera_ok),
        "last_error": last_error,
    }


def _delayed_grab_after_detection(delay_s=2.0):
    global grap_request, last_error
    sleep(delay_s)
    if exit_flag or is_grabbing:
        return
    if len(msg) > 0:
        grap_request = True
        last_error = ""
        print("[web] delayed go: start grab")
    else:
        last_error = "no target detected; keep detect on and retry grab"
        print("[web] delayed go: no target")


def handle_ui_command(action, params):
    global target_classes, detect_enabled, obstacle_enabled
    global grap_request, exit_flag, msg, last_error

    action = (action or "").strip().lower()

    if action == "target":
        classes_raw = params.get("classes", [""])[0]
        selected = []
        for c in classes_raw.replace(";", ",").split(","):
            c = c.strip().lower()
            if c in VALID_CLASSES and c not in selected:
                selected.append(c)
        target_classes = selected
        detect_enabled = bool(target_classes)
        last_error = ""
        print("[web] set target:", target_classes if target_classes else "none")

    elif action == "detect_on":
        detect_enabled = True
        last_error = ""
        print("[web] detect on")

    elif action == "detect_off":
        detect_enabled = False
        print("[web] detect off")

    elif action == "obstacle_toggle":
        obstacle_enabled = not obstacle_enabled
        print("[web] obstacle:", "on" if obstacle_enabled else "off")

    elif action == "obstacle_on":
        obstacle_enabled = True
        print("[web] obstacle on")

    elif action == "obstacle_off":
        obstacle_enabled = False
        target.clear_color_obstacles()
        print("[web] obstacle off")

    elif action == "grab":
        if not detect_enabled:
            last_error = "run detect first"
            print("[web] grab refused: detect off")
        elif len(target_classes) == 0:
            last_error = "select target first"
            print("[web] grab refused: no class")
        elif is_grabbing:
            last_error = "grabbing already running"
            print("[web] grab refused: busy")
        elif len(msg) == 0:
            last_error = "no target detected"
            print("[web] grab refused: no target")
        else:
            grap_request = True
            last_error = ""
            print("[web] grab command sent")

    elif action == "go":
        classes_raw = params.get("classes", [""])[0]
        selected = []
        for c in classes_raw.replace(";", ",").split(","):
            c = c.strip().lower()
            if c in VALID_CLASSES and c not in selected:
                selected.append(c)
        if not selected:
            last_error = "valid classes: {}".format(VALID_CLASSES)
        elif is_grabbing:
            last_error = "grabbing already running"
        else:
            target_classes = selected
            detect_enabled = True
            obstacle_enabled = True
            last_error = "detecting 2s, then grab if target exists"
            print("[web] go", target_classes)
            threading.Thread(target=_delayed_grab_after_detection, daemon=True).start()

    elif action == "reset":
        target_classes = []
        msg.clear()
        detect_enabled = False
        obstacle_enabled = False
        grap_request = False
        target.clear_color_obstacles()
        try:
            target.grap.move_status = True
        except Exception:
            pass
        last_error = ""
        print("[web] reset done")

    elif action == "quit":
        exit_flag = True
        print("[web] quit")

    else:
        last_error = "unknown action: {}".format(action)

    return _state_dict()


class WebHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def _send_json(self, obj, status=200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path in ("/", "/index.html"):
            html = WEB_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
            return

        if parsed.path == "/api/status":
            self._send_json(_state_dict())
            return

        if parsed.path == "/api/cmd":
            params = parse_qs(parsed.query)
            action = params.get("action", [""])[0]
            self._send_json(handle_ui_command(action, params))
            return

        if parsed.path.startswith("/stream.mjpg"):
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()

            while not exit_flag:
                with frame_lock:
                    frame = None if latest_frame is None else latest_frame.copy()

                if frame is None:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv.putText(frame, "Waiting for camera frame...", (40, 240),
                               cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                ok, jpg = cv.imencode(".jpg", frame, [int(cv.IMWRITE_JPEG_QUALITY), 75])
                if not ok:
                    sleep(0.05)
                    continue

                data = jpg.tobytes()
                try:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(("Content-Length: %d\r\n\r\n" % len(data)).encode())
                    self.wfile.write(data + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    break

                sleep(0.05)
            return

        self.send_error(404)


WEB_HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dofbot v8 Web Control</title>
<style>
  body { margin:0; background:#111; color:#eee; font-family:Arial,"Microsoft YaHei",sans-serif; }
  .wrap { max-width:1120px; margin:0 auto; padding:12px; }
  h2 { margin:8px 0 12px; font-size:22px; }
  .grid { display:grid; grid-template-columns: 1fr 320px; gap:12px; align-items:start; }
  .video { background:#000; border:1px solid #333; border-radius:10px; overflow:hidden; }
  .video img { width:100%; display:block; }
  .panel { background:#1b1b1b; border:1px solid #333; border-radius:10px; padding:12px; }
  .row { margin:10px 0; }
  button { width:100%; margin:4px 0; padding:10px; border:0; border-radius:8px; cursor:pointer; font-size:15px; }
  button.primary { background:#1e88e5; color:white; }
  button.good { background:#21a366; color:white; }
  button.warn { background:#f39c12; color:#111; }
  button.danger { background:#d9534f; color:white; }
  button.gray { background:#444; color:white; }
  .smallgrid { display:grid; grid-template-columns:1fr 1fr 1fr; gap:6px; }
  .kv { font-size:14px; line-height:1.65; background:#111; border-radius:8px; padding:10px; white-space:pre-wrap; }
  .hint { color:#aaa; font-size:13px; line-height:1.5; }
  @media(max-width:900px){ .grid{grid-template-columns:1fr;} .panel{order:-1;} }
</style>
</head>
<body>
<div class="wrap">
  <h2>DOFBOT v8 识别抓取控制界面</h2>
  <div class="grid">
    <div class="video"><img src="/stream.mjpg" alt="camera stream"></div>
    <div class="panel">
      <div class="row">
        <b>选择检测目标</b>
        <div class="smallgrid">
          <button class="primary" onclick="cmd('target','plastic')">plastic</button>
          <button class="primary" onclick="cmd('target','box')">box</button>
          <button class="primary" onclick="cmd('target','leaf')">leaf</button>
        </div>
      </div>
      <div class="row">
        <b>一键检测并抓取</b>
        <div class="smallgrid">
          <button class="good" onclick="cmd('go','plastic')">抓 plastic</button>
          <button class="good" onclick="cmd('go','box')">抓 box</button>
          <button class="good" onclick="cmd('go','leaf')">抓 leaf</button>
        </div>
      </div>
      <div class="row">
        <button class="good" onclick="api('/api/cmd?action=grab')">抓取当前识别目标</button>
        <button class="gray" onclick="api('/api/cmd?action=detect_on')">开启检测</button>
        <button class="gray" onclick="api('/api/cmd?action=detect_off')">关闭检测</button>
        <button class="warn" onclick="api('/api/cmd?action=obstacle_toggle')">障碍物开/关</button>
        <button class="danger" onclick="api('/api/cmd?action=reset')">重置</button>
      </div>
      <div class="row"><b>状态</b><div id="state" class="kv">loading...</div></div>
      <div class="hint">
        画面中会显示 YOLO 检测框、颜色障碍物、安全圈和路径预览。<br>
        如果画面不动，先检查摄像头是否被 RDK Studio 预览占用。<br>
        抓取前先确认机械臂周围没有人手和线缆干涉。
      </div>
    </div>
  </div>
</div>
<script>
async function api(url){
  try { await fetch(url); await refresh(); }
  catch(e){ console.log(e); }
}
function cmd(action, cls){
  api('/api/cmd?action=' + encodeURIComponent(action) + '&classes=' + encodeURIComponent(cls));
}
async function refresh(){
  try{
    const r = await fetch('/api/status');
    const s = await r.json();
    document.getElementById('state').textContent =
      '目标类别: ' + (s.classes.length ? s.classes.join(', ') : 'none') + '\n' +
      '检测: ' + (s.detect ? 'ON' : 'OFF') + '\n' +
      '障碍物: ' + (s.obstacle ? 'ON' : 'OFF') + ' / 数量: ' + s.obstacles + '\n' +
      '抓取状态: ' + (s.grabbing ? 'GRABBING' : 'IDLE') + '\n' +
      '当前目标: ' + (s.targets.length ? s.targets.join(', ') : 'none') + '\n' +
      '摄像头: ' + (s.camera_ok ? 'OK' : 'WAIT') + '\n' +
      '提示: ' + (s.last_error || '');
  }catch(e){}
}
setInterval(refresh, 800);
refresh();
</script>
</body>
</html>
'''


def web_server():
    try:
        httpd = ThreadingHTTPServer(("0.0.0.0", WEB_PORT), WebHandler)
        print("[web] open browser: http://RDK_X5_IP:{}/".format(WEB_PORT))
        print("[web] example: http://192.168.1.126:{}/".format(WEB_PORT))
        httpd.serve_forever()
    except Exception as e:
        print("[web][error]", e)


def camera():
    global msg, is_grabbing, grap_request
    global detect_enabled, obstacle_enabled, exit_flag
    global latest_frame, camera_ok, last_error

    capture = _open_camera()
    if not capture.isOpened():
        last_error = "cannot open camera"
        print("[error] cannot open camera")
        return

    camera_ok = True

    os.system("v4l2-ctl -d /dev/video0 --set-ctrl=exposure_dynamic_framerate=0 2>/dev/null")
    os.system("v4l2-ctl -d /dev/video0 --set-ctrl=brightness=0 2>/dev/null")
    os.system("v4l2-ctl -d /dev/video0 --set-ctrl=backlight_compensation=0 2>/dev/null")

    ret, img = capture.read()
    if ret:
        img = cv.resize(img, (640, 480))
        try:
            calibration.calibration_map(img, xy[:], 140)
            sleep(1.5)
        except Exception as e:
            print("calibration_map error:", e)

    print("[camera] started; HEADLESS={} WEB_UI={}".format(1 if HEADLESS else 0, 1 if WEB_UI else 0))

    while capture.isOpened():
        try:
            if exit_flag:
                break

            ret, img = capture.read()
            if not ret:
                last_error = "camera read failed"
                print("[error] read failed")
                break

            img = cv.resize(img, (640, 480))

            try:
                img = calibration.Perspective_transform(dp, img)
            except Exception:
                pass

            if obstacle_enabled and not is_grabbing:
                target.update_color_obstacles(img)
            elif not obstacle_enabled:
                target.clear_color_obstacles()

            if detect_enabled and len(target_classes) > 0 and not is_grabbing:
                img, msg = target.select_targets(img, target_classes)
            else:
                img = target.draw_obstacles(img)

            if grap_request and (not is_grabbing):
                if len(msg) != 0:
                    is_grabbing = True
                    msg_snapshot = msg.copy()
                    print("[grab] targets:", list(msg_snapshot.keys()))
                    threading.Thread(
                        target=run_grap_safe,
                        args=(msg_snapshot, xy[:]),
                        daemon=True
                    ).start()
                else:
                    last_error = "no target detected"
                    print("[info] no target detected")
                    grap_request = False

            if is_grabbing:
                state_text, state_color = "GRABBING", (0, 0, 255)
            elif detect_enabled:
                state_text, state_color = "DETECTING", (0, 255, 0)
            else:
                state_text, state_color = "IDLE", (128, 128, 128)

            cv.putText(img, state_text, (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 1, state_color, 2)

            if obstacle_enabled:
                cv.putText(img, "OBSTACLE ON ({})".format(len(target.color_obstacles)),
                           (10, 100), cv.FONT_HERSHEY_SIMPLEX, 0.6,
                           (0, 0, 255), 2)

            if len(msg) > 0 and not is_grabbing:
                cv.putText(img, "Targets: " + str(list(msg.keys())),
                           (10, 65), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                           (255, 255, 0), 2)

            with frame_lock:
                latest_frame = img.copy()

            if not HEADLESS:
                cv.imshow("Dofbot Camera", img)
                key = cv.waitKey(1) & 0xFF
                if key == ord('q'):
                    exit_flag = True
                    break

        except KeyboardInterrupt:
            break
        except Exception as e:
            last_error = "camera error: {}".format(e)
            print("camera error:", e)
            break

    camera_ok = False
    capture.release()
    if not HEADLESS:
        cv.destroyAllWindows()


def command_loop():
    global target_classes, msg, grap_request
    global detect_enabled, obstacle_enabled, exit_flag

    print("\n" + "=" * 50)
    print("  YOLO + Color Obstacle Grasping (Smart Fallback) v8 WEB")
    print("=" * 50)
    print("  Browser UI: http://RDK_X5_IP:{}/".format(WEB_PORT))
    print("  target plastic box leaf  Set targets")
    print("  detect                   Toggle YOLO detect")
    print("  obstacle                 Toggle color obstacle")
    print("  grab                     Send grab command")
    print("  go plastic leaf          Detect+grab in one shot")
    print("  status                   Show status")
    print("  reset                    Reset")
    print("  quit                     Exit\n")

    while not exit_flag:
        try:
            cmd = input(">>> ").strip().lower()
            if cmd == "":
                continue

            parts = cmd.split()
            action = parts[0]

            if action == "target":
                target_classes = []
                for c in parts[1:]:
                    if c in VALID_CLASSES and c not in target_classes:
                        target_classes.append(c)
                print("  set:", target_classes if target_classes else "none")

            elif action == "detect":
                detect_enabled = not detect_enabled
                print("  detect:", "on" if detect_enabled else "off")

            elif action == "obstacle":
                obstacle_enabled = not obstacle_enabled
                print("  obstacle:", "on" if obstacle_enabled else "off")

            elif action == "grab":
                if not detect_enabled:
                    print("  run detect first")
                elif len(target_classes) == 0:
                    print("  run target first")
                elif is_grabbing:
                    print("  grabbing...")
                elif len(msg) == 0:
                    print("  no target")
                else:
                    grap_request = True
                    print("  grab command sent")

            elif action == "go":
                if len(parts) < 2:
                    print("  usage: go plastic leaf")
                else:
                    target_classes = []
                    for c in parts[1:]:
                        if c in VALID_CLASSES and c not in target_classes:
                            target_classes.append(c)

                    if target_classes:
                        detect_enabled = True
                        obstacle_enabled = True
                        print("  classes:", target_classes)
                        print("  detecting, wait 2s...")
                        sleep(2)

                        if len(msg) > 0:
                            grap_request = True
                            print("  start grab!")
                        else:
                            print("  no target, type grab to retry")
                    else:
                        print("  valid:", VALID_CLASSES)

            elif action == "status":
                s = _state_dict()
                print("  classes:", s["classes"] if s["classes"] else "none")
                print("  detect:", "on" if s["detect"] else "off")
                print("  obstacle:", "on" if s["obstacle"] else "off")
                print("  grab:", "running" if s["grabbing"] else "idle")
                print("  targets:", s["targets"] if s["targets"] else "none")
                print("  obstacles:", s["obstacles"])
                print("  headless:", HEADLESS)
                print("  web:", "http://RDK_X5_IP:{}/".format(WEB_PORT))

            elif action == "reset":
                handle_ui_command("reset", {})
                print("  reset done")

            elif action in ["quit", "exit", "q"]:
                exit_flag = True
                break

            else:
                print("  commands: target/detect/obstacle/grab/go/status/reset/quit")

        except (EOFError, KeyboardInterrupt):
            exit_flag = True
            break


if __name__ == "__main__":
    print("[start] YOLO + Color Obstacle Grasping (Smart Fallback) v8 WEB")

    if WEB_UI:
        web_thread = threading.Thread(target=web_server, daemon=True)
        web_thread.start()

    cam_thread = threading.Thread(target=camera, daemon=True)
    cam_thread.start()

    try:
        command_loop()
    except Exception:
        pass
    finally:
        exit_flag = True
        sleep(1)
        if not HEADLESS:
            cv.destroyAllWindows()
        print("[exit] program ended")
