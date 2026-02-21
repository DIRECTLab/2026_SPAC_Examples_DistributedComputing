"""
Simple Webcam -> State Examples (OpenCV)

Three progressively richer "image -> state" pipelines:
1) Corridor Occupancy (binary FREE/OCCUPIED) using background subtraction
2) Edge Density (scalar risk score) using Canny edges
3) Free-Space Columns (vector state) using an obstacle mask, then binning

Requirements:
  pip install opencv-python numpy

Run:
  python webcam_image_to_state.py
Keys:
  1/2/3 : switch examples
  q     : quit
"""

import cv2
import numpy as np
from collections import deque

# ---------------------------
# Utilities
# ---------------------------

def draw_roi(frame, roi):
    x1, y1, x2, y2 = roi
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

def put_text(frame, lines, x=20, y=30, dy=28, scale=0.8):
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x, y + i * dy),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2)

def central_roi(h, w, w_frac=0.35, h_frac=0.55, y_start_frac=0.25):
    roi_w = int(w * w_frac)
    roi_h = int(h * h_frac)
    x1 = (w - roi_w) // 2
    y1 = int(h * y_start_frac)
    x2 = x1 + roi_w
    y2 = y1 + roi_h
    return (x1, y1, x2, y2)

def bottom_roi(h, w, w_frac=0.85, h_frac=0.35, y_end_frac=0.98):
    roi_w = int(w * w_frac)
    roi_h = int(h * h_frac)
    x1 = (w - roi_w) // 2
    y2 = int(h * y_end_frac)
    y1 = max(0, y2 - roi_h)
    x2 = x1 + roi_w
    return (x1, y1, x2, y2)

def clip01(x):
    return float(max(0.0, min(1.0, x)))

# ---------------------------
# Example 1: Corridor Occupancy (binary)
# ---------------------------

class CorridorOccupancy:
    """
    Uses background subtraction to detect foreground in a central ROI.
    Outputs:
      - occupied (bool)
      - confidence (0..1) via sliding window
      - occ_ratio (0..1) fraction of foreground pixels in ROI
    """
    def __init__(self, tau=0.03, conf_thresh=0.6, window=10):
        self.tau = tau
        self.conf_thresh = conf_thresh
        self.window = deque(maxlen=window)
        self.bg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=True)

    def step(self, frame, roi):
        fg = self.bg.apply(frame)

        # Clean up mask
        fg = cv2.medianBlur(fg, 5)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)  # remove shadow gray
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_DILATE, np.ones((7, 7), np.uint8), iterations=1)

        x1, y1, x2, y2 = roi
        roi_mask = fg[y1:y2, x1:x2]
        occ_ratio = np.count_nonzero(roi_mask) / roi_mask.size

        occupied = occ_ratio > self.tau
        self.window.append(1 if occupied else 0)
        confidence = sum(self.window) / len(self.window)

        stop = confidence >= self.conf_thresh
        state = "OCCUPIED" if stop else "FREE"

        return fg, occ_ratio, confidence, state

# ---------------------------
# Example 2: Edge Density (scalar)
# ---------------------------

class EdgeDensity:
    """
    Computes edge density in a central ROI.
    Outputs:
      - x_hat in [0,1] (normalized edge density)
      - confidence c_t via exponential smoothing
    """
    def __init__(self, d_max=0.15, alpha=0.3, occ_thresh=0.4):
        self.d_max = d_max
        self.alpha = alpha
        self.occ_thresh = occ_thresh
        self.c = 0.0

    def step(self, frame, roi):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 1.2)
        edges = cv2.Canny(blur, 60, 150)

        x1, y1, x2, y2 = roi
        roi_edges = edges[y1:y2, x1:x2]
        d = np.count_nonzero(roi_edges) / roi_edges.size
        x_hat = clip01(d / self.d_max)

        # Exponential smoothing (belief-like)
        self.c = self.alpha * x_hat + (1.0 - self.alpha) * self.c
        state = "OCCUPIED" if self.c > self.occ_thresh else "FREE"

        # Create a debug overlay image
        edges_vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return edges_vis, d, x_hat, self.c, state

# ---------------------------
# Example 3: Free-Space Columns (vector)
# ---------------------------

class FreeSpaceColumns:
    """
    Produces a K-dimensional free-space vector from an obstacle mask in a bottom ROI.
    Obstacle mask uses edges + dilation for a simple, general heuristic.
    Outputs:
      - x_vec in [0,1]^K
      - suggested direction (LEFT/STRAIGHT/RIGHT)
    """
    def __init__(self, K=9):
        self.K = K

    def obstacle_mask(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 1.2)
        edges = cv2.Canny(blur, 60, 150)
        # Make edges thicker to approximate occupied regions
        mask = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        return mask

    def step(self, frame, roi):
        mask = self.obstacle_mask(frame)
        x1, y1, x2, y2 = roi
        roi_mask = mask[y1:y2, x1:x2]

        H, W = roi_mask.shape[:2]
        bin_w = max(1, W // self.K)

        x_vec = []
        for i in range(self.K):
            bx1 = i * bin_w
            bx2 = W if i == self.K - 1 else (i + 1) * bin_w
            bin_mask = roi_mask[:, bx1:bx2]
            o_i = np.count_nonzero(bin_mask) / bin_mask.size  # obstacle fraction
            x_i = 1.0 - o_i                                   # free-space score
            x_vec.append(x_i)

        x_vec = np.array(x_vec, dtype=np.float32)

        # Suggested direction: pick the max free-space bin
        i_star = int(np.argmax(x_vec))
        center = (self.K - 1) / 2.0
        if i_star < center - 1:
            direction = "LEFT"
        elif i_star > center + 1:
            direction = "RIGHT"
        else:
            direction = "STRAIGHT"

        # Visualization: draw bars on the frame
        vis = frame.copy()
        # Draw ROI box
        draw_roi(vis, roi)

        # Draw bar chart inside the ROI bottom edge
        # (simple overlay; no fancy colors requested)
        bar_h = 80
        base_y = y2 - 10
        start_x = x1 + 10
        total_w = (x2 - x1) - 20
        bw = total_w / self.K

        for i, val in enumerate(x_vec):
            # bar height proportional to free-space score
            h_i = int(bar_h * val)
            px1 = int(start_x + i * bw)
            px2 = int(start_x + (i + 1) * bw - 4)
            py1 = base_y - h_i
            py2 = base_y
            cv2.rectangle(vis, (px1, py1), (px2, py2), (255, 255, 255), 2)

        return mask, x_vec, direction, vis

# ---------------------------
# Main loop: choose example
# ---------------------------

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try changing VideoCapture index (0/1).")

    mode = 1
    ex1 = CorridorOccupancy(tau=0.03, conf_thresh=0.6, window=10)
    ex2 = EdgeDensity(d_max=0.15, alpha=0.3, occ_thresh=0.4)
    ex3 = FreeSpaceColumns(K=9)

    print("Controls: 1=Corridor Occupancy, 2=Edge Density, 3=Free-Space Columns, q=quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        if mode in (1, 2):
            roi = central_roi(h, w)
        else:
            roi = bottom_roi(h, w)

        if mode == 1:
            fg, occ_ratio, confidence, state = ex1.step(frame, roi)
            vis = frame.copy()
            draw_roi(vis, roi)
            put_text(vis, [
                "Example 1: Corridor Occupancy (binary)",
                f"ROI occupancy r: {occ_ratio:.3f}",
                f"Confidence c: {confidence:.2f}",
                f"State: {state}",
                "Tip: stand still 5s to learn background"
            ])
            cv2.imshow("Webcam", vis)
            cv2.imshow("Foreground Mask", fg)

        elif mode == 2:
            edges_vis, d, x_hat, c, state = ex2.step(frame, roi)
            vis = frame.copy()
            draw_roi(vis, roi)
            put_text(vis, [
                "Example 2: Edge Density (scalar risk)",
                f"Edge density d: {d:.3f}",
                f"x_hat (normalized): {x_hat:.2f}",
                f"Smoothed c: {c:.2f}",
                f"State: {state}"
            ])
            cv2.imshow("Webcam", vis)
            cv2.imshow("Edges", edges_vis)

        elif mode == 3:
            mask, x_vec, direction, vis = ex3.step(frame, roi)
            put_text(vis, [
                "Example 3: Free-Space Columns (vector state)",
                f"x_vec: [{', '.join(f'{v:.2f}' for v in x_vec)}]",
                f"Suggested direction: {direction}",
                "Interpretation: higher values mean freer columns"
            ], scale=0.6)
            cv2.imshow("Webcam + Bars", vis)
            cv2.imshow("Obstacle Mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('1'):
            mode = 1
        if key == ord('2'):
            mode = 2
        if key == ord('3'):
            mode = 3

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
