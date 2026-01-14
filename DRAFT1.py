import cv2
import numpy as np
import os
from dataclasses import dataclass


# CONFIG
REAL_WIDTH_IN = 4.000   # 4inches-from manual
CAM_INDEX = 0

OUT_DIR = "OUTPUT_MEASURE"
os.makedirs(OUT_DIR, exist_ok=True)

# Preprocess params
BLUR_KSIZE = (5, 5)
CANNY_T1 = 50   #low threshold
CANNY_T2 = 150  #high threshold

# Contour selection constraints
MIN_AREA_FRAC = 0.10  # quad area must be at least this fraction of image area
EPS_FRAC = 0.02       # approxPolyDP epsilon as fraction of contour perimeter
#decrease to add points

# Validation tolerance in rectified space (pixels)
EDGE_MATCH_TOL_PX = 8


# =========================
# UTILITIES
# =========================
def order_corners(pts: np.ndarray) -> np.ndarray:

    pts = pts.astype(np.float32)

    s = pts.sum(axis=1)          # x+y smallest for top left, largest for botom right
    diff = np.diff(pts, axis=1)  # x-y, small for bottom left, large for top right

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def quad_edge_lengths(quad: np.ndarray):
    """
   pixel length from ordered corners
    """
    tl, tr, br, bl = quad
    top = np.linalg.norm(tr - tl)
    right = np.linalg.norm(br - tr)
    bottom = np.linalg.norm(br - bl)
    left = np.linalg.norm(bl - tl)
    return top, right, bottom, left


def find_best_quad(frame_bgr: np.ndarray):
    """
    convert the image to edges, close gaps so boundaries become continuous,
    extract outer contours, simplify each contour into a polygon, and pick
    the largest valid convex 4-corner polygon as the face.
    """
    h, w = frame_bgr.shape[:2]
    img_area = float(h * w)

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, BLUR_KSIZE, 0)
    edges = cv2.Canny(blur, CANNY_T1, CANNY_T2)

    # Close gaps to stabilize the contour
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    #find contours-only external-ignore inner crevices
    cnts, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, (gray, edges, edges_closed)

    # Sort by area (largest first)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    best = None
    for c in cnts[:15]:  # scan top candidates
        area = cv2.contourArea(c)
        if area < MIN_AREA_FRAC * img_area:
            break

        peri = cv2.arcLength(c, True)   #perimeter of teh contour
        eps = EPS_FRAC * peri   #tolerance
        approx = cv2.approxPolyDP(c, eps, True) #simplified polygon version of contour

        if len(approx) == 4 and cv2.isContourConvex(approx):
            quad = approx.reshape(4, 2).astype(np.float32)
            quad = order_corners(quad)

            # edges should not be tiny
            top, right, bottom, left = quad_edge_lengths(quad)
            if min(top, right, bottom, left) < 30:
                continue

            best = quad
            break

    return best, (gray, edges, edges_closed)


def compute_rectified(frame_bgr: np.ndarray, quad: np.ndarray):
    """
    Compute homography and warp to a top-down view.
    Returns (warped, H, (outW, outH))
    """
    quad = order_corners(quad)
    top, right, bottom, left = quad_edge_lengths(quad)

    outW = int(round(max(top, bottom)))
    outH = int(round(max(left, right)))

    # Avoid zero or tiny outputs
    outW = max(outW, 50)
    outH = max(outH, 50)

    dst = np.array([[0, 0],
                    [outW - 1, 0],
                    [outW - 1, outH - 1],
                    [0, outH - 1]], dtype=np.float32)

    H = cv2.getPerspectiveTransform(quad, dst)  #matrix for perspective correction
    warped = cv2.warpPerspective(frame_bgr, H, (outW, outH), flags=cv2.INTER_LINEAR)

    return warped, H, (outW, outH)


def validate_rectified(warped_shape, tol_px=EDGE_MATCH_TOL_PX):
    """
    In rectified view, opposite edges should match closely in pixel length:
    width top vs bottom ~ outW, height left vs right ~ outH
    Since we warp to exact rectangle, this mostly validates corner ordering.
    """
    outH, outW = warped_shape[:2]
    # Here, outW/outH are definition of destination; this is a minimal check.
    # Real validation is more useful when measuring edges from the warped content.
    return True


def annotate_quad(frame_bgr: np.ndarray, quad: np.ndarray):
    vis = frame_bgr.copy()
    q = quad.astype(int)
    cv2.polylines(vis, [q], True, (0, 0, 255), 2)
    for i, (x, y) in enumerate(q):
        cv2.circle(vis, (x, y), 6, (0, 255, 0), -1)
        cv2.putText(vis, f"{i}", (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 120, 0), 2)
    return vis


def measure_in_inches(outW_px: int, outH_px: int, real_width_in: float):
    """
    Use known real width to compute inches_per_pixel and convert both dims.
    We assume outW corresponds to the known real width.
    """
    inches_per_px = real_width_in / float(outW_px)
    width_in = outW_px * inches_per_px
    height_in = outH_px * inches_per_px
    return width_in, height_in, inches_per_px


# MANUAL CORNER PICKER

@dataclass
class ClickState:
    pts: list

def pick_corners_gui(image_bgr: np.ndarray):
    """
    Manual fallback: user clicks 4 corners in order:
    TL -> TR -> BR -> BL (recommended).
    """
    state = ClickState(pts=[])

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(state.pts) < 4:
                state.pts.append([x, y])

    win = "Manual Corner Pick (Click 4 corners, then press ENTER)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        vis = image_bgr.copy()
        for i, (x, y) in enumerate(state.pts):
            cv2.circle(vis, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(vis, f"{i+1}", (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 120, 0), 2)
        cv2.putText(vis, "Click 4 corners (TL,TR,BR,BL). ENTER=done, R=reset, ESC=cancel",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 10), 2)

        cv2.imshow(win, vis)
        key = cv2.waitKey(20) & 0xFF

        if key == 27:  # ESC
            cv2.destroyWindow(win)
            return None
        if key in (13, 10):  # ENTER
            if len(state.pts) == 4:
                cv2.destroyWindow(win)
                return order_corners(np.array(state.pts, dtype=np.float32))
        if key in (ord('r'), ord('R')):
            state.pts = []


# MAIN: GUI CAPTURE + PIPELINE

def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Check CAM_INDEX or camera permissions.")

    print("Controls:")
    print("  C = capture frame")
    print("  Q / ESC = quit")

    captured = None

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        display = frame.copy()
        cv2.putText(display, "Press C to capture, Q/ESC to quit",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)

        cv2.imshow("Live Capture", display)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), ord('Q'), 27):
            cap.release()
            cv2.destroyAllWindows()
            return

        if key in (ord('c'), ord('C')):
            captured = frame.copy()
            break

    cap.release()
    cv2.destroyWindow("Live Capture")

    if captured is None:
        return

    cv2.imwrite(os.path.join(OUT_DIR, "01_captured.png"), captured)

    # Auto detect quad
    quad, debug_imgs = find_best_quad(captured)
    gray, edges, edges_closed = debug_imgs

    cv2.imwrite(os.path.join(OUT_DIR, "02_gray.png"), gray)
    cv2.imwrite(os.path.join(OUT_DIR, "03_edges.png"), edges)
    cv2.imwrite(os.path.join(OUT_DIR, "04_edges_closed.png"), edges_closed)

    if quad is None:
        print("[AUTO] Could not find a reliable 4-corner face.")
        print("Press M for manual corner selection, or ESC to exit.")
        cv2.imshow("Captured", captured)
        while True:
            k = cv2.waitKey(0) & 0xFF
            if k == 27:
                cv2.destroyAllWindows()
                return
            if k in (ord('m'), ord('M')):
                cv2.destroyAllWindows()
                quad = pick_corners_gui(captured)
                if quad is None:
                    print("Manual selection cancelled.")
                    return
                break
    else:
        vis_quad = annotate_quad(captured, quad)
        cv2.imwrite(os.path.join(OUT_DIR, "05_auto_quad.png"), vis_quad)

        # Optional: allow switching to manual if the auto quad looks wrong
        cv2.imshow("Auto Quad (press M to redo manually, any other key to accept)", vis_quad)
        k = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        if k in (ord('m'), ord('M')):
            quad2 = pick_corners_gui(captured)
            if quad2 is not None:
                quad = quad2

    # Rectify
    warped, H, (outW, outH) = compute_rectified(captured, quad)
    cv2.imwrite(os.path.join(OUT_DIR, "06_rectified.png"), warped)

    # Measure in pixels (in rectified plane, it’s simply output size)
    width_in, height_in, inches_per_px = measure_in_inches(outW, outH, REAL_WIDTH_IN)

    # Print results
    print("\n=== Measurement Results (Planar Face) ===")
    print(f"Rectified size: {outW}px × {outH}px")
    print(f"Inches per pixel: {inches_per_px:.8f} in/px")
    print(f"Measured width : {width_in:.4f} in  (reference set to {REAL_WIDTH_IN:.4f} in)")
    print(f"Measured height: {height_in:.4f} in")

    # Annotate rectified output with measurements
    annotated = warped.copy()
    cv2.putText(annotated, f"Width:  {width_in:.4f} in",
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)
    cv2.putText(annotated, f"Height: {height_in:.4f} in",
                (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)
    cv2.imwrite(os.path.join(OUT_DIR, "07_rectified_annotated.png"), annotated)

    cv2.imshow("Rectified (Top-Down)", annotated)
    print(f"\nSaved outputs in: {os.path.abspath(OUT_DIR)}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
