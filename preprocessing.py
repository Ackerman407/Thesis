# Real-time mouth ROI stream using webcam
# Notes:
# - Keep your Python 3.10–3.12 env. Install: mediapipe==0.10.14 opencv-python tensorflow
# - Press 'q' in the preview window to quit.

from typing import List, Optional, Deque
from collections import deque
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# --- MediaPipe setup (do this once) ---
_mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,   # improves lips/iris precision
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Outer+inner lip landmark indices (MediaPipe FaceMesh, 468 pts)
_LIP_IDX = list(set([
    # outer
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 
    291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
    # inner
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308
]))

def _mouth_bbox_from_landmarks(landmarks, w, h, pad=0.12, pad_top=0.30):
    xs, ys = [], []
    for idx in _LIP_IDX:
        lm = landmarks[idx]
        xs.append(lm.x * w)
        ys.append(lm.y * h)

    xmin, xmax = np.min(xs), np.max(xs)
    ymin, ymax = np.min(ys), np.max(ys)

    # base padding
    pw = (xmax - xmin) * pad
    ph = (ymax - ymin) * pad

    xmin = max(0, int(np.floor(xmin - pw)))
    xmax = min(w, int(np.ceil(xmax + pw)))

    # extra padding to the top (helps include upper lip/nose shadow)
    ymin = max(0, int(np.floor(ymin - ph - (ymax - ymin) * pad_top)))
    ymax = min(h, int(np.ceil(ymax + ph)))

    return xmin, ymin, xmax, ymax

def _process_frame_bgr(frame_bgr, target_h=46, target_w=140):
    """Returns (gray_crop, vis_crop). gray_crop is (46,140,1) float32 (unnormalized)."""
    # BGR -> RGB for MediaPipe
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    res = _mp_face_mesh.process(rgb)
    if res.multi_face_landmarks:
        lms = res.multi_face_landmarks[0].landmark
        x1, y1, x2, y2 = _mouth_bbox_from_landmarks(lms, w, h, pad=0.18)
        crop = frame_bgr[y1:y2, x1:x2] if (y2 > y1 and x2 > x1) else frame_bgr
    else:
        crop = frame_bgr

    # resize to your old ROI size
    crop = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # grayscale
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # (Optional) stability: CLAHE
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    # gray = clahe.apply(gray)

    vis = crop.copy()  # for display
    gray = gray[..., None].astype(np.float32)  # (H,W,1)
    return gray, vis

class RealTimeMouthStream:
    """
    Reads from a webcam and maintains a rolling window of mouth crops.
    Use .get_window_tensor() to retrieve a [T,46,140,1] normalized tensor.
    """
    def __init__(self, cam_index:int=0, window_size:int=75, stride:int=1, target_h:int=46, target_w:int=140):
        self.cap = cv2.VideoCapture(cam_index)
        self.window_size = window_size
        self.stride = stride
        self.target_h = target_h
        self.target_w = target_w
        self.buffer: Deque[np.ndarray] = deque(maxlen=window_size)  # store (H,W,1) float32 frames
        self.frame_count = 0

        # Try to set a reasonable FPS/size (optional)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

    def is_opened(self) -> bool:
        return self.cap.isOpened()

    def read_step(self) -> Optional[np.ndarray]:
        """Reads one frame from camera, processes to (46,140,1) float32 (unnormalized), returns it; None if failed."""
        ok, frame = self.cap.read()
        if not ok:
            return None
        gray, vis = _process_frame_bgr(frame, self.target_h, self.target_w)

        # Only append every 'stride' frames to control temporal sampling
        self.frame_count += 1
        if self.frame_count % self.stride == 0:
            self.buffer.append(gray)

        # show preview (mouth crop)
        cv2.imshow("Mouth ROI (press 'q' to quit)", vis)
        return gray

    def get_window_tensor(self) -> tf.Tensor:
        """
        Returns a normalized tensor [T,46,140,1] using (x - mean) / (std + 1e-6).
        If not enough frames yet, it will return what’s available (T < window_size).
        """
        if len(self.buffer) == 0:
            return tf.zeros([0, self.target_h, self.target_w, 1], dtype=tf.float32)
        x = tf.convert_to_tensor(np.stack(list(self.buffer), axis=0), dtype=tf.float32)  # [T,H,W,1]
        mean = tf.reduce_mean(x)
        std = tf.math.reduce_std(x)
        x = (x - mean) / (std + 1e-6)
        return x  # shape [T,46,140,1]

    def release(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    stream = RealTimeMouthStream(cam_index=0, window_size=75, stride=1)  # change cam_index if needed

    if not stream.is_opened():
        print("Could not open camera. Try a different cam_index (e.g., 1) or check permissions.")
        exit(1)

    try:
        while True:
            frame = stream.read_step()  # process one frame and update buffer
            if frame is None:
                print("Camera frame read failed.")
                break

            # Example: whenever you have enough frames, get a tensor for inference
            window_tensor = stream.get_window_tensor()  # [T,46,140,1], normalized
            # You can call your model here when len(buffer) == desired T
            # e.g., if window_tensor.shape[0] == 75: preds = model(window_tensor[None, ...])

            # quit on 'q'
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
    finally:
        stream.release()
