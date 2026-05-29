"""
Controls:
R = draw zone
Q = quit
"""

import cv2
import numpy as np
import os
import time
import wave
import struct
import math
import threading
from datetime import datetime
from ultralytics import YOLO

# SETTINGS 

VIDEO = "Screen Recording 2026-05-29 002054 (online-video-cutter.com).mp4"
OUTPUT_DIR = "crossing_output"
CONFIDENCE = 0.4
MAX_DISPLAY_W = 800
BEEP_EVERY = 1.5
SKIP_FRAMES = 1
MAX_CLIP_SECONDS = 10
ZONE_POINTS = 8

# OUTPUT DIR

os.makedirs(f"{OUTPUT_DIR}/screenshots", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/clips", exist_ok=True)

# IMPORT YOLO 

model = YOLO("yolov8n.pt")

PERSON = 0
zone_pts = None
zone_pts_display = None
active = {}
done = []
frame_idx = 0
fps = 25
scale = 1.0
# Stores last detections to prevent flicker
last_results = None

# BEEP 

def make_beep(path="beep.wav"):
    sr = 44100
    dur = 0.3
    n = int(sr * dur)
    with wave.open(path, "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        for i in range(n):
            fade = min(i, n - i, sr // 20) / (sr // 20)
            val = int(
                0.5
                * 32767
                * fade
                * math.sin(2 * math.pi * 880 * i / sr)
            )
            f.writeframes(struct.pack("<h", val))
    return path

BEEP_FILE = make_beep()


def beep():
    def _play():
        try:
            if os.name == "nt":
                import winsound
                winsound.PlaySound(
                    BEEP_FILE,
                    winsound.SND_FILENAME | winsound.SND_ASYNC
                )
            else:
                print("\a", end="", flush=True)

        except:
            pass

    threading.Thread(target=_play, daemon=True).start()


#  ZONE HELPERS

def in_zone(x1, y1, x2, y2):

    global zone_pts

    if zone_pts is None:
        return False

    foot = (
        float((x1 + x2) // 2),
        float(y2)
    )

    return cv2.pointPolygonTest(zone_pts, foot, False) >= 0


def draw_zone(frame, alert=False):
    global zone_pts_display
    if zone_pts_display is None:
        return
    if alert:
        overlay = frame.copy()
        cv2.fillPoly(
            overlay,
            [zone_pts_display],
            (0, 0, 100)
        )
        cv2.addWeighted(
            overlay,
            0.3,
            frame,
            0.7,
            0,
            frame
        )
    cv2.polylines(
        frame,
        [zone_pts_display],
        True,
        (0, 0, 255),
        2
    )

    for p in zone_pts_display:
        cv2.circle(
            frame,
            tuple(p),
            1,
            (0, 0, 255),
            -1
        )

#  SAVE HELPERS 

def save_screenshot(frame, pid):

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"{OUTPUT_DIR}/screenshots/person{pid}_{ts}.jpg"
    cv2.imwrite(path, frame)
    print(f"[SCREENSHOT] {path}")


def save_clip(pid, frames):
    global fps
    if not frames:
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dur = round(len(frames) / fps, 1)
    path = f"{OUTPUT_DIR}/clips/person{pid}_{dur}s_{ts}.mp4"
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )
    for f in frames:
        out.write(f)
    out.release()
    print(f"[CLIP] {path}")


# ───────────────── HUD ──────────────────────

def draw_hud(frame):
    global active
    global zone_pts
    h, w = frame.shape[:2]
    cv2.rectangle(
        frame,
        (0, 0),
        (w, 40),
        (20, 20, 20),
        -1
    )
    zone_status = (
        "ACTIVE"
        if zone_pts is not None
        else "NOT SET"
    )
    msg = (
        f"Crossing Detector | "
        f"Violations: {len(active)} | "
        f"Zone: {zone_status} | "
        f"R=Draw Zone | Q=Quit"
    )
    cv2.putText(
        frame,
        msg,
        (10, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

# ZONE PICKER

def pick_4_points(frozen_display, s):
    PICK_WIN = "ZONE SELECTOR"
    clicks = []
    base = frozen_display.copy()
    cv2.destroyAllWindows()
    cv2.namedWindow(PICK_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(PICK_WIN, 1200, 700)
    def redraw():
        d = base.copy()
        cv2.rectangle(
            d,
            (0, 0),
            (d.shape[1], 50),
            (20, 20, 20),
            -1
        )
        if len(clicks) < ZONE_POINTS:
            msg = (
                f"Click Point {len(clicks)+1}/{ZONE_POINTS} | "
                f"ENTER=Confirm | BACKSPACE=Undo | ESC=Cancel"
            )
        else:
            msg = "Press ENTER to confirm"
        cv2.putText(
            d,
            msg,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 100),
            2
        )
        for i, pt in enumerate(clicks):
            cv2.circle(d, pt, 8, (0, 0, 255), -1)
            cv2.putText(
                d,
                str(i + 1),
                (pt[0] + 10, pt[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )
        if len(clicks) >= 2:
            pts_arr = np.array(clicks, dtype=np.int32)
            cv2.polylines(
                d,
                [pts_arr],
                len(clicks) == ZONE_POINTS,
                (0, 0, 255),
                2
            )
        cv2.imshow(PICK_WIN, d)
    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(clicks) < ZONE_POINTS:
                clicks.append((x, y))
                redraw()
    cv2.setMouseCallback(PICK_WIN, mouse_cb)
    cv2.waitKey(1)
    redraw()
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            cv2.destroyWindow(PICK_WIN)
            return None, None
        elif key == 8 or key == 127:
            if clicks:
                clicks.pop()
                redraw()
        elif key == 13:
            if len(clicks) == ZONE_POINTS:
                break
    cv2.destroyWindow(PICK_WIN)
    disp_pts = np.array(clicks, dtype=np.int32)
    orig_pts = (disp_pts / s).astype(np.int32)
    return orig_pts, disp_pts

# MAIN 

def process(source):
    global frame_idx
    global fps
    global scale
    global zone_pts
    global zone_pts_display
    global active
    global done
    global last_results
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, MAX_DISPLAY_W / vid_w)
    disp_w = int(vid_w * scale)
    disp_h = int(vid_h * scale)
    WIN = "Crossing Detector"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, disp_w, disp_h)
    print(f"[INFO] {vid_w}x{vid_h} @ {fps:.0f}fps")
    last_display = None
    while True:
        ret, raw_frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        display = cv2.resize(
            raw_frame,
            (disp_w, disp_h)
        )
        last_display = display.copy()
        seen_ids = set()

        #  RUN YOLO ONLY SOMETIMES

        if frame_idx % (SKIP_FRAMES + 1) == 0:
            last_results = model.track(
                raw_frame,
                imgsz=640,
                persist=True,
                conf=CONFIDENCE,
                classes=[PERSON],
                verbose=False
            )
        results = last_results
        
        # DETECTION 
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                tid = (
                    int(box.id[0])
                    if box.id is not None
                    else -1
                )
                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )
                dx1 = int(x1 * scale)
                dy1 = int(y1 * scale)
                dx2 = int(x2 * scale)
                dy2 = int(y2 * scale)
                inside = (
                    cls == PERSON
                    and tid >= 0
                    and in_zone(x1, y1, x2, y2)
                )

                # COLORS 

                color = (0, 0, 255) if inside else (0, 255, 0)

                # BOX 
                cv2.rectangle(
                    display,
                    (dx1, dy1),
                    (dx2, dy2),
                    color,
                    1
                )
                cv2.rectangle(
                    display,
                    (dx1, dy1 - 28),
                    (dx1 + 140, dy1-10),
                    color,
                    -1
                )
                cv2.putText(
                    display,
                    f"Person #{tid}",
                    (dx1 + 5, dy1 - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 255, 255),   1
                )

                # ALERT 
                if inside:
                    seen_ids.add(tid)
                    cv2.putText(
                        display,
                        "INTRUSION",
                        (dx1, dy2 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 0, 255),
                        2
                    )
                    cv2.circle(
                        display,
                        (int((x1 + x2) / 2 * scale), dy2),
                        2,
                        (0, 0, 255),
                        -1
                    )
                    if tid not in active:
                        print(f"[ALERT] Person #{tid} entered zone")
                        active[tid] = {
                            "entry_time": time.time(),
                            "last_frame": frame_idx,
                            "frames": [],
                            "last_beep": 0
                        }
                        # CLEAN SCREENSHOT
                        save_screenshot(
                            raw_frame.copy(),
                            tid
                        )
                    p = active[tid]
                    p["last_frame"] = frame_idx
                    # Annotated clip frame
                    p["frames"].append(
                        display.copy()
                    )
                    if len(p["frames"]) > int(fps * MAX_CLIP_SECONDS):
                        p["frames"].pop(0)
                    if time.time() - p["last_beep"] >= BEEP_EVERY:
                        beep()
                        p["last_beep"] = time.time()

        # PERSON LEFT 
        for tid in [t for t in list(active) if t not in seen_ids]:
            p = active[tid]
            if frame_idx - p["last_frame"] > int(fps * 1.5):
                print(f"[INFO] Person #{tid} left")
                save_clip(
                    tid,
                    p["frames"]
                )
                done.append(tid)
                del active[tid]

        # DRAW ZONE 
        draw_zone(
            display,
            alert=bool(active)
        )
        draw_hud(display)
        cv2.imshow(WIN, display)
        key = cv2.waitKey(1) & 0xFF
        # QUIT 
        if key == ord("q"):
            break

        #  DRAW ZONE
        elif key == ord("r") or key == ord("R"):
            print("[ZONE] Draw 4 points")
            cv2.destroyAllWindows()
            new_orig, new_disp = pick_4_points(
                last_display,
                scale
            )

            cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(
                WIN,
                disp_w,
                disp_h
            )

            if new_orig is not None:
                zone_pts = new_orig
                zone_pts_display = new_disp
                active.clear()
                print("[ZONE] Updated")
            else:
                print("[ZONE] Cancelled")

    # CLEANUP 
    for tid, p in active.items():
        save_clip(
            tid,
            p["frames"]
        )
    cap.release()
    cv2.destroyAllWindows()
    print(
        f"[DONE] Violations: "
        f"{len(done)+len(active)}"
    )


#  ENTRY
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--video",
        default=VIDEO,
        help="Video file"
    )
    args = ap.parse_args()
    process(args.video or 0)