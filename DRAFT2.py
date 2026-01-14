import cv2
import numpy as np
import os
import math
from dataclasses import dataclass


# CONFIG

# CAMERA GEOMETRY (LOCKED SETUP)
CAMERA_DISTANCE_IN = 15.0     # distance from camera to object plane (inches)
HFOV_DEG = 31.5               # empirically calibrated horizontal FOV for 640x480

CAM_INDEX = 0

OUT_DIR = "OUTPUT_MEASURE"
os.makedirs(OUT_DIR, exist_ok=True)

# Preprocess params
BLUR_KSIZE = (5, 5)
CANNY_T1 = 50    # low threshold
CANNY_T2 = 150   # high threshold

# Contour selection constraints
MIN_AREA_FRAC = 0.10
EPS_FRAC = 0.02

# UTILITIES

def compute_inches_per_pixel(distance_in, hfov_deg, image_width_px):
    """
    Compute physical scale using pinhole camera geometry.
    """
    scene_width_in = 2.0 * distance_in * math.tan(math.radians(hfov_deg / 2.0))
    return scene_width_in / float(image_width_px)


def order_corners(pts: np.ndarray) -> np.ndarray:
    pts = pts.astype(np.float32)

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def quad_edge_lengths(quad: np.ndarray):
    """
    Pixel edge lengths from ordered corners
    """
    tl, tr, br, bl = quad
    top = np.linalg.norm(tr - tl)
    right = np.linalg.norm(br - tr)
    bottom = np.linalg.norm(br - bl)
    left = np.linalg.norm(bl - tl)
    return top, right, bottom, left


def find_best_quad(frame_bgr: np.ndarray):
    """
    Detect the largest valid convex 4-corner polygon.
    """
    h, w = frame_bgr.shape[:2]
    img_area = float(h * w)

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, BLUR_KSIZE, 0)
    edges = cv2.Canny(blur, CANNY_T1, CANNY_T2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, (gray, edges, edges_closed)

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    for c in cnts[:15]:
        area = cv2.contourArea(c)
        if area < MIN_AREA_FRAC * img_area:
            break

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, EPS_FRAC * peri, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            quad = order_corners(approx.reshape(4, 2))
            top, right, bottom, left = quad_edge_lengths(quad)
            if min(top, right, bottom, left) >= 30:
                return quad, (gray, edges, edges_closed)

    return None, (gray, edges, edges_closed)


def compute_rectified(frame_bgr: np.ndarray, quad: np.ndarray):
    """
    Homography for visualization only.
    """
    quad = order_corners(quad)
    top, right, bottom, left = quad_edge_lengths(quad)

    outW = int(round(max(top, bottom)))
    outH = int(round(max(left, right)))
    outW = max(outW, 50)
    outH = max(outH, 50)

    dst = np.array([
        [0, 0],
        [outW - 1, 0],
        [outW - 1, outH - 1],
        [0, outH - 1]
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(frame_bgr, H, (outW, outH))

    return warped, H, (outW, outH)


def annotate_quad(frame_bgr: np.ndarray, quad: np.ndarray):
    vis = frame_bgr.copy()
    q = quad.astype(int)
    cv2.polylines(vis, [q], True, (0, 0, 255), 2)
    return vis


def measure_in_inches_from_scale(width_px, height_px, inches_per_px):
    return width_px * inches_per_px, height_px * inches_per_px


# MANUAL CORNER PICKER

@dataclass
class ClickState:
    pts: list


def pick_corners_gui(image_bgr: np.ndarray):
    state = ClickState(pts=[])

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(state.pts) < 4:
            state.pts.append([x, y])

    win = "Manual Corner Pick (TL → TR → BR → BL)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        vis = image_bgr.copy()
        for i, (x, y) in enumerate(state.pts):
            cv2.circle(vis, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(vis, str(i + 1), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 120, 0), 2)

        cv2.putText(
            vis,
            "Click 4 corners. ENTER=done, R=reset, ESC=cancel",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (10, 10, 10),
            2
        )

        cv2.imshow(win, vis)
        key = cv2.waitKey(20) & 0xFF

        if key == 27:
            cv2.destroyWindow(win)
            return None

        if key in (13, 10) and len(state.pts) == 4:
            cv2.destroyWindow(win)
            return order_corners(np.array(state.pts, dtype=np.float32))

        if key in (ord('r'), ord('R')):
            state.pts = []


# MAIN

def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    print("Controls: C=capture | Q/ESC=quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        cv2.imshow("Live Capture", frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            cap.release()
            cv2.destroyAllWindows()
            return

        if key in (ord('c'), ord('C')):
            captured = frame.copy()
            break

    cap.release()
    cv2.destroyAllWindows()

    cv2.imwrite(os.path.join(OUT_DIR, "01_captured.png"), captured)

    CAPTURE_WIDTH_PX = captured.shape[1]
    CAPTURE_HEIGHT_PX = captured.shape[0]
    print(f"[INFO] Captured resolution: {CAPTURE_WIDTH_PX} x {CAPTURE_HEIGHT_PX}")

    quad, debug_imgs = find_best_quad(captured)

    if quad is not None:
        vis_quad = annotate_quad(captured, quad)
        cv2.imshow("Auto Quad — ENTER=accept | M=manual | ESC=cancel", vis_quad)

        while True:
            k = cv2.waitKey(0) & 0xFF
            if k in (13, 10):
                cv2.destroyAllWindows()
                break
            if k in (ord('m'), ord('M')):
                cv2.destroyAllWindows()
                quad_manual = pick_corners_gui(captured)
                if quad_manual is None:
                    return
                quad = quad_manual
                break
            if k == 27:
                cv2.destroyAllWindows()
                return
    else:
        quad = pick_corners_gui(captured)
        if quad is None:
            return

    warped, H, _ = compute_rectified(captured, quad)
    cv2.imwrite(os.path.join(OUT_DIR, "06_rectified.png"), warped)

    inches_per_px = compute_inches_per_pixel(
        CAMERA_DISTANCE_IN,
        HFOV_DEG,
        CAPTURE_WIDTH_PX
    )

    top, right, bottom, left = quad_edge_lengths(order_corners(quad))
    width_px = 0.5 * (top + bottom)
    height_px = 0.5 * (left + right)

    width_in, height_in = measure_in_inches_from_scale(
        width_px, height_px, inches_per_px
    )

    print("\n=== Measurement Results ===")
    print(f"Width  : {width_in:.4f} in")
    print(f"Height : {height_in:.4f} in")

    annotated = warped.copy()
    cv2.putText(annotated, f"Width: {width_in:.4f} in",
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)
    cv2.putText(annotated, f"Height: {height_in:.4f} in",
                (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)

    cv2.imwrite(os.path.join(OUT_DIR, "07_rectified_annotated.png"), annotated)
    cv2.imshow("Rectified (Display Only)", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
