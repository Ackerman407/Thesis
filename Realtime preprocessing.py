# rpi_realtime_preprocess.py
import time
from collections import deque
import numpy as np
import cv2

USE_PICAMERA2 = True
try:
    from picamera2 import Picamera2
except Exception:
    USE_PICAMERA2 = False

# --- Config matching your old slice ---
TARGET_H, TARGET_W = 256, 320            # we force frames to this size
Y1, Y2 = 190, 236                         # -> height 46
X1, X2 = 80, 220                          # -> width 140
WINDOW_SIZE = 75                          # sequence length (T)

def get_camera():
    if USE_PICAMERA2:
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={"format": "RGB888", "size": (640, 480)}
        )
        picam2.configure(config)
        picam2.start()
        time.sleep(0.2)
        return picam2
    else:
        cap = cv2.VideoCapture(0)  # USB cam fallback
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        return cap

def read_frame_rgb(cam):
    if USE_PICAMERA2:
        return cam.capture_array()      # RGB (H,W,3)
    ok, bgr = cam.read()
    if not ok:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def main():
    cam = get_camera()
    buffer = deque(maxlen=WINDOW_SIZE)

    try:
        while True:
            rgb = read_frame_rgb(cam)
            if rgb is None:
                print("Camera frame read failed.")
                break

            # 1) Resize to ensure your original slice indices are valid
            rgb_resized = cv2.resize(rgb, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)

            # 2) Convert to grayscale (equivalent to tf.image.rgb_to_grayscale)
            gray = cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2GRAY)  # (256,320)

            # 3) Crop exactly like your code: frames[:, 190:236, 80:220, :]
            roi = gray[Y1:Y2, X1:X2]  # (46,140)

            # 4) Keep channel dim and float32
            roi = roi.astype(np.float32)[..., None]  # (46,140,1)

            # 5) Append to rolling sequence buffer
            buffer.append(roi)

            # 6) Compute sequence-wide normalization when we have at least 2 frames
            if len(buffer) > 1:
                x = np.stack(list(buffer), axis=0).astype(np.float32)   # [T,46,140,1]
                mean = x.mean()
                std = x.std()
                x_norm = (x - mean) / (std + 1e-6)                      # your normalization

                # For visualization: show the **latest normalized ROI**
                latest_norm = x_norm[-1, ..., 0]  # (46,140)
                # Map to 0..255 just for display (not needed for your model)
                vis = latest_norm
                vis = np.clip((vis - vis.min()) / (vis.ptp() + 1e-6), 0.0, 1.0)
                vis = (vis * 255).astype(np.uint8)

                # Enlarge display so it’s visible
                vis_big = cv2.resize(vis, (140*3, 46*3), interpolation=cv2.INTER_NEAREST)
                cv2.imshow("Normalized ROI (press 'q' to quit)", vis_big)
            else:
                # Show raw ROI until we have enough frames for normalization
                vis_big = cv2.resize(roi[..., 0].astype(np.uint8), (140*3, 46*3), interpolation=cv2.INTER_NEAREST)
                cv2.imshow("Normalized ROI (press 'q' to quit)", vis_big)

            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

    finally:
        if USE_PICAMERA2:
            try:
                cam.stop()
            except Exception:
                pass
        else:
            cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
