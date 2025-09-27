import cv2
import numpy as np
import logging
from typing import Tuple, Dict, Any, Optional
import mtl_ui as uicfg

# ---------------------- Utils (processing) ----------------------
TARGET_W, TARGET_H = uicfg.TARGET_W, uicfg.TARGET_H

def resize_to_target(img, w=TARGET_W, h=TARGET_H):
    ih, iw = img.shape[:2]
    interp = cv2.INTER_AREA if (iw > w or ih > h) else cv2.INTER_CUBIC
    return cv2.resize(img, (w, h), interpolation=interp)

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    return cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

def load_calibration_data(path):
    try:
        data = np.load(path)
        return data['mtx'], data['dist']
    except Exception:
        return None, None

def get_external_and_holes(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None:
        return None, []
    hierarchy = hierarchy[0]
    ext_indices = [i for i, h in enumerate(hierarchy) if h[3] == -1]
    if not ext_indices:
        return None, []
    largest_idx = max(ext_indices, key=lambda i: cv2.contourArea(contours[i]))
    ext_cnt = contours[largest_idx]
    hole_indices = [i for i, h in enumerate(hierarchy) if h[3] == largest_idx]
    holes = [contours[i] for i in hole_indices if cv2.contourArea(contours[i]) > 50]
    return ext_cnt, holes

def classify_shape(cnt, approx, holes, area_px):
    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    rect = cv2.minAreaRect(cnt)
    (rw, rh) = rect[1]
    rect_area = float(rw * rh) if rw > 0 and rh > 0 else 0.0
    area_px = float(area_px)
    solidity = (area_px / hull_area) if hull_area > 0 else 0.0
    extent   = (area_px / rect_area) if rect_area > 0 else 0.0
    v        = len(approx) if approx is not None else 0
    concavity = 1.0 - ((area_px / hull_area) if hull_area > 0 else 0.0)
    SOL_MIN, EXT_MIN, CONC_MAX = 0.95, 0.90, 0.05
    if holes:
        return "Asymmetrical", "#00d084"
    is_rect_like = (v in (4, 5)) and (solidity >= SOL_MIN) and (extent >= EXT_MIN) and (concavity <= CONC_MAX)
    return ("Symmetrical", "#00d084") if is_rect_like else ("Asymmetrical", "#00d084")

def _order_box_pts(pts4):
    pts = np.array(pts4, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def _warp_to_rect(mask, rect):
    """
    Warp สำหรับ MASK เท่านั้น (INTER_NEAREST) เพื่อไม่ให้ขอบเบลอ
    """
    (w, h) = rect[1]
    w = int(max(1, round(w)))
    h = int(max(1, round(h)))
    box = cv2.boxPoints(rect).astype(np.float32)
    box = _order_box_pts(box)
    dst = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(box, dst)
    warped = cv2.warpPerspective(mask, M, (w, h), flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    Minv = cv2.getPerspectiveTransform(dst, box)
    return warped, M, Minv, (w, h)

# ---------- Largest-rectangle decomposition ----------
def _largest_rect_in_binary(bin_u8: np.ndarray):
    # bin_u8: 255=โลหะ, 0=ไม่ใช่
    H, W = bin_u8.shape
    hist = np.zeros(W, dtype=np.int32)
    best = (0, 0, 0, 0, 0)  # area, x, y, w, h
    for i in range(H):
        row = (bin_u8[i] > 0).astype(np.int32)
        hist = (hist + row) * row
        stack = []
        j = 0
        while j <= W:
            cur = hist[j] if j < W else 0
            if not stack or cur >= hist[stack[-1]]:
                stack.append(j); j += 1
            else:
                h = hist[stack.pop()]
                l = stack[-1] if stack else -1
                w = j - l - 1
                area = h * w
                if area > best[0] and h > 0 and w > 0:
                    x = l + 1
                    y = i - h + 1
                    best = (area, x, y, w, h)
    return best  # area,x,y,w,h

def _decompose_into_rects(metal_bin: np.ndarray,
                          cover_ratio: float = 0.995,
                          max_rects: int = 8,
                          min_w: int = 12,
                          min_h: int = 12):
    rem = metal_bin.copy()
    total = float(cv2.countNonZero(metal_bin))
    rects = []
    while len(rects) < max_rects:
        area, x, y, w, h = _largest_rect_in_binary(rem)
        if area <= 0 or w < min_w or h < min_h:
            break
        rects.append((x, y, w, h))
        rem[y:y+h, x:x+w] = 0
        covered = total - cv2.countNonZero(rem)
        if total > 0 and covered / total >= cover_ratio:
            break
    return rects

# ---------- Axes & measurement ----------
def _axes_from_rect_pts(rect_pts: np.ndarray):
    ordp = _order_box_pts(rect_pts.astype(np.float32))
    tl, tr, br, bl = ordp
    e_u = tr - tl
    e_v = br - tr
    len_u = float(np.linalg.norm(e_u))
    len_v = float(np.linalg.norm(e_v))
    unit_u = e_u / (len_u + 1e-9)
    unit_v = e_v / (len_v + 1e-9)
    return tl.astype(np.float32), unit_u, unit_v, len_u, len_v

def _proj_span_mm(corners_xy: np.ndarray, origin_tl: np.ndarray,
                  unit_u: np.ndarray, unit_v: np.ndarray,
                  ratio_w: float, ratio_h: float) -> Tuple[float, float]:
    pts = corners_xy.astype(np.float32)
    rel = pts - origin_tl[None, :]
    u_coords = rel @ unit_u
    v_coords = rel @ unit_v
    du_px = float(u_coords.max() - u_coords.min())
    dv_px = float(v_coords.max() - v_coords.min())
    return du_px * float(ratio_w), dv_px * float(ratio_h)

# ---------- Draw helpers ----------
def _midpoint(p1, p2):
    p = ( (p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5 )
    return (int(round(p[0])), int(round(p[1])))

def _draw_label(img, text, org, color, scale=2.0, thickness=3):
    cv2.putText(img, text, (org[0]+1, org[1]+1),
                cv2.FONT_HERSHEY_PLAIN, scale, (0,0,0), thickness+2, cv2.LINE_AA)
    cv2.putText(img, text, org,
                cv2.FONT_HERSHEY_PLAIN, scale, color, thickness, cv2.LINE_AA)

def _draw_measure_lines(img, rect_pts, width_lr_mm, height_tb_mm):
    ordered = _order_box_pts(rect_pts.astype(np.float32))
    tl, tr, br, bl = [tuple(map(int, pt)) for pt in ordered]

    m_left  = _midpoint(tl, bl)
    m_right = _midpoint(tr, br)
    m_top   = _midpoint(tl, tr)
    m_bot   = _midpoint(bl, br)

    col_w = (0, 220, 255)  # Width
    col_h = (0, 180,  80)  # Height
    thick = 4

    cv2.line(img, m_left,  m_right, col_w, thick, lineType=cv2.LINE_AA)
    cv2.line(img, m_top,   m_bot,   col_h, thick, lineType=cv2.LINE_AA)

    def _cap(p, q, color):
        v = np.array([q[0]-p[0], q[1]-p[1]], dtype=np.float32)
        nrm = float(np.linalg.norm(v))
        if nrm < 1e-3: return
        v /= (nrm + 1e-9)
        n = np.array([-v[1], v[0]], dtype=np.float32)
        L = 16
        a1 = (int(round(p[0] + n[0]*L)), int(round(p[1] + n[1]*L)))
        a2 = (int(round(p[0] - n[0]*L)), int(round(p[1] - n[1]*L)))
        b1 = (int(round(q[0] + n[0]*L)), int(round(q[1] + n[1]*L)))
        b2 = (int(round(q[0] - n[0]*L)), int(round(q[1] - n[1]*L)))
        cv2.line(img, a1, a2, color, thick, lineType=cv2.LINE_AA)
        cv2.line(img, b1, b2, color, thick, lineType=cv2.LINE_AA)

    _cap(m_left, m_right, col_w)
    _cap(m_top,  m_bot,   col_h)

    mw = _midpoint(m_left, m_right)
    mh = _midpoint(m_top,  m_bot)
    _draw_label(img, f"W: {width_lr_mm:.1f} mm",  (mw[0]+500, mw[1]-20), col_w)
    _draw_label(img, f"H: {height_tb_mm:.1f} mm", (mh[0]+500, mh[1]+20), col_h)

def _put(img, text, org, scale=1.8):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_PLAIN, scale, (0,0,0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_PLAIN, scale, (255,255,255), 2, cv2.LINE_AA)

# ---------------------- Processor ----------------------
class Processor:
    def __init__(
        self,
        calibration_data_path: Optional[str],
        blue_lower=uicfg.blue_Lower,
        blue_upper=uicfg.blue_Upper,
        rotate_angle_deg=uicfg.ROTATE_ANGLE_DEG,
        crop_slice: Tuple[slice, slice] = uicfg.CROP_SLICE,
        pixel_mm_ratio_w: float = uicfg.PIXEL_MM_RATIO_W,
        pixel_mm_ratio_h: float = uicfg.PIXEL_MM_RATIO_H,
        logger: Optional[logging.Logger] = None
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.blue_lower = blue_lower
        self.blue_upper = blue_upper
        self.rotate_angle_deg = rotate_angle_deg
        self.crop_slice = crop_slice
        self.pixel_mm_ratio_w = float(pixel_mm_ratio_w)
        self.pixel_mm_ratio_h = float(pixel_mm_ratio_h)

        self.mtx, self.dist = (None, None)
        if calibration_data_path:
            self.mtx, self.dist = load_calibration_data(calibration_data_path)

        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

    def run_measurement(self, frame_bgr) -> Tuple[Optional[Dict[str, Any]], np.ndarray]:
        try:
            frame_HD = frame_bgr

            # 1) undistort + rotate + crop
            if self.mtx is not None and self.dist is not None:
                frame_HD = cv2.undistort(frame_HD, self.mtx, self.dist, None, self.mtx)
            frame_HD = rotate_image(frame_HD, self.rotate_angle_deg)
            yslice, xslice = self.crop_slice
            frame_HD = frame_HD[yslice, xslice].copy()
            output = frame_HD.copy()

            # 2) glare removal
            gray = cv2.cvtColor(frame_HD, cv2.COLOR_BGR2GRAY)
            _, thr = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            closed = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, self.kernel)
            frame_no_glare = cv2.inpaint(frame_HD, closed, 3, cv2.INPAINT_TELEA)

            # 3) background masking (โลหะ = not blue)
            hsv = cv2.cvtColor(frame_no_glare, cv2.COLOR_BGR2HSV)
            blue_mask = cv2.inRange(hsv, self.blue_lower, self.blue_upper)
            obj_mask = cv2.bitwise_not(blue_mask)
            obj_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_OPEN, self.kernel, iterations=1)
            obj_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)

            if cv2.countNonZero(obj_mask) <= 10000:
                return None, output

            # 4) contour + holes
            ext_cnt, hole_cnts = get_external_and_holes(obj_mask)
            if ext_cnt is None:
                return None, output

            # 5) พื้นที่รวม
            outer_area_px = cv2.contourArea(ext_cnt)
            holes_area_px = sum(cv2.contourArea(hc) for hc in hole_cnts)
            net_area_px = max(0.0, outer_area_px - holes_area_px)
            net_area_mm2 = float(net_area_px) * self.pixel_mm_ratio_w * self.pixel_mm_ratio_h
            if net_area_px < 1000:
                return None, output

            # 6) minAreaRect + warp (MASK)
            rect = cv2.minAreaRect(ext_cnt)
            warped_mask, M, Minv, (W, H) = _warp_to_rect(obj_mask, rect)  # 255 = metal (not-blue)
            warped_bgr  = cv2.warpPerspective(frame_no_glare, M, (W, H), flags=cv2.INTER_LINEAR)

            # กรอบใหญ่ในภาพเดิม + รู
            rect_pts = cv2.boxPoints(rect).astype(np.int32)
            cv2.polylines(output, [rect_pts], True, (0,0,0), 2, lineType=cv2.LINE_8)
            cv2.drawContours(output, [ext_cnt], -1, (255, 0, 255), 2, lineType=cv2.LINE_AA)
            for hc in hole_cnts:
                cv2.drawContours(output, [hc], -1, (0,0,0), 2, lineType=cv2.LINE_8)

            # 7) เตรียมในพิกัด warp
            k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

            metal_w = (warped_mask > 0).astype(np.uint8) * 255          # 255=โลหะ
            metal_solid_warp = metal_w.copy()
            metal_area_px = max(1, cv2.countNonZero(metal_solid_warp))

            # ---------- แกนเส้นวัด ----------
            origin_tl, unit_u, unit_v, len_u_px, len_v_px = _axes_from_rect_pts(rect_pts)
            big_w_mm = len_u_px * self.pixel_mm_ratio_w
            big_h_mm = len_v_px * self.pixel_mm_ratio_h

            # ---------- หา LOST เฉพาะรูจริงๆ ด้วย flood fill ----------
            inv = cv2.bitwise_not(metal_solid_warp)  # 255 = not metal (พื้นหลัง + รู)
            holes_only = inv.copy()
            hW, wW = holes_only.shape
            fmask = np.zeros((hW+2, wW+2), np.uint8)
            # fill จากทุกขอบให้พื้นหลังกลายเป็น 0
            for x in range(wW):
                if holes_only[0, x] == 255:     cv2.floodFill(holes_only, fmask, (x, 0), 0)
                if holes_only[hW-1, x] == 255:  cv2.floodFill(holes_only, fmask, (x, hW-1), 0)
            for y in range(hW):
                if holes_only[y, 0] == 255:     cv2.floodFill(holes_only, fmask, (0, y), 0)
                if holes_only[y, wW-1] == 255:  cv2.floodFill(holes_only, fmask, (wW-1, y), 0)
            holes_only = cv2.morphologyEx(holes_only, cv2.MORPH_OPEN, k3, iterations=1)
            holes_only = cv2.morphologyEx(holes_only, cv2.MORPH_CLOSE, k3, iterations=1)

            MIN_AREA_PX = max(1500, int(0.002 * metal_area_px))
            MIN_W = 12
            MIN_H = 12
            MIN_R_RATIO = 0.02

            num, labels, stats, _ = cv2.connectedComponentsWithStats(holes_only, connectivity=8)
            clean_lost = np.zeros_like(holes_only)
            for i in range(1, num):
                x, y, w, h, area = stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3], stats[i, 4]
                if area >= MIN_AREA_PX and w >= MIN_W and h >= MIN_H:
                    clean_lost[labels == i] = 255

            # ถ้า lost รวม < 3% ของโลหะทั้งหมด ให้ถือว่าไม่มี (กัน noise)
            lost_count_px = int(cv2.countNonZero(clean_lost))
            lost_ratio = lost_count_px / float(max(1, metal_area_px))
            if lost_ratio < 0.03:
                clean_lost[:] = 0

            # ---- คำนวณ concavity เพื่อเช็ค "แผ่นสี่เหลี่ยมปกติไม่ต้องมี R" ----
            cnts_metal, _ = cv2.findContours(metal_solid_warp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            concavity = 0.0
            if cnts_metal:
                c = max(cnts_metal, key=cv2.contourArea)
                hull = cv2.convexHull(c)
                area = float(cv2.contourArea(c))
                hull_area = float(cv2.contourArea(hull)) if hull is not None else 0.0
                concavity = 1.0 - (area / hull_area) if hull_area > 0 else 0.0

            # ทำ metal_core = โลหะ − lost เพื่อแตก RN และดันขอบเข้าเล็กน้อย
            metal_core = metal_solid_warp.copy()
            metal_core[clean_lost > 0] = 0
            metal_core = cv2.erode(metal_core, k3, iterations=1)

            # ----- แปลง lost เป็นพิกัดเดิม และเตรียม mask "กรอบแดง" -----
            h0, w0 = obj_mask.shape[:2]
            lost_orig = cv2.warpPerspective(clean_lost, Minv, (w0, h0), flags=cv2.INTER_NEAREST)
            lost_boxes_mask_orig = np.zeros((h0, w0), dtype=np.uint8)

            # ---------------- วาดกรอบแดง LOST + เติม mask กรอบแดง ----------------
            lost_boxes_pts_orig = []
            per_hole_details = []  # (idx, w_mm, h_mm, area_mm2, (cx_o, cy_o))
            cnts_lost, _ = cv2.findContours(clean_lost, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts_lost:
                if cv2.contourArea(c) < MIN_AREA_PX:
                    continue
                x,y,w,h = cv2.boundingRect(c)
                if w < MIN_W or h < MIN_H:
                    continue

                rect_lost = cv2.minAreaRect(c)
                box_warp  = cv2.boxPoints(rect_lost).astype(np.float32).reshape(-1, 1, 2)
                box_orig  = cv2.perspectiveTransform(box_warp, Minv).reshape(-1, 2)
                box_i32   = np.round(box_orig).astype(np.int32)
                # Compute W/H from the red minAreaRect box (projected back to original coords)
                try:
                    wmm_h, hmm_h = _proj_span_mm(
                        box_orig.astype(np.float32), origin_tl, unit_u, unit_v,
                        self.pixel_mm_ratio_w, self.pixel_mm_ratio_h
                    )
                except Exception:
                    wmm_h, hmm_h = 0.0, 0.0
                # Area from contour (in warp) -> mm^2
                area_px_h = float(cv2.contourArea(c))
                amm2_h = area_px_h * self.pixel_mm_ratio_w * self.pixel_mm_ratio_h
                # Centroid (warp) -> original
                Mc = cv2.moments(c)
                if Mc["m00"] != 0:
                    cx_w, cy_w = float(Mc["m10"]/Mc["m00"]), float(Mc["m01"]/Mc["m00"])
                else:
                    # fallback: center of box in warp space
                    cx_w = float((x + x + w) / 2.0); cy_w = float((y + y + h) / 2.0)
                pt = np.array([[[cx_w, cy_w]]], dtype=np.float32)
                pt_orig = cv2.perspectiveTransform(pt, Minv)[0,0]
                cx_o, cy_o = int(round(float(pt_orig[0]))), int(round(float(pt_orig[1])))
                per_hole_details.append((len(per_hole_details)+1, wmm_h, hmm_h, amm2_h, (cx_o, cy_o)))

                cv2.polylines(output, [box_i32], True, (0, 0, 255), 3, lineType=cv2.LINE_8)
                cv2.fillConvexPoly(lost_boxes_mask_orig, box_i32, 255, lineType=cv2.LINE_8)
                lost_boxes_pts_orig.append(box_i32)

            # โลหะที่ยังใช้งานได้ (ไว้เป็นตัวหาร % ของ R)  **ต้องคำนวณหลังเติมกรอบแดง**
            usable_metal_orig = cv2.bitwise_and(obj_mask, cv2.bitwise_not(lost_orig))
            usable_metal_orig = cv2.bitwise_and(usable_metal_orig, cv2.bitwise_not(lost_boxes_mask_orig))
            total_usable_px = max(1, cv2.countNonZero(usable_metal_orig))

            # ---------- ไม่แตก R ถ้าเป็นแผ่นสี่เหลี่ยมปกติ ----------
            is_plain_rect = (len(hole_cnts) == 0) and (concavity <= 0.02)
            if is_plain_rect:
                rects_ax = []
            else:
                rects_ax = _decompose_into_rects(
                    metal_core, cover_ratio=0.995, max_rects=8, min_w=12, min_h=12
                )

            # ---------- Project RN กลับพิกัดเดิม พร้อมกันไม่ให้ทับ lost/lost-box ----------
            polys_back = []
            r_masks_orig = []
            colors = [(0,255,0), (90,150,255), (255,180,0), (180,120,255),
                      (120,255,120), (255,120,120), (120,200,255), (200,120,255)]
            dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

            for (rx, ry, rw, rh) in rects_ax:
                corners_warp = np.array(
                    [[rx,ry],[rx+rw,ry],[rx+rw,ry+rh],[rx,ry+rh]], dtype=np.float32
                ).reshape(-1,1,2)
                back = cv2.perspectiveTransform(corners_warp, Minv).reshape(-1,2)
                back_int = np.round(back).astype(np.int32)

                rmask = np.zeros((h0, w0), dtype=np.uint8)
                cv2.fillConvexPoly(rmask, back_int, 255, lineType=cv2.LINE_8)
                rmask = cv2.dilate(rmask, dilate_kernel, iterations=1)

                # ห้ามทับโลหะที่ไม่มีจริง และห้ามทับกรอบแดง
                rmask = cv2.bitwise_and(rmask, obj_mask)
                rmask = cv2.bitwise_and(rmask, cv2.bitwise_not(lost_orig))
                rmask = cv2.bitwise_and(rmask, cv2.bitwise_not(lost_boxes_mask_orig))

                area_R_px = cv2.countNonZero(rmask)
                # เกณฑ์ 2% ของโลหะที่ใช้งานได้
                if area_R_px / float(total_usable_px) < MIN_R_RATIO:
                    continue
                if area_R_px == 0:
                    continue

                polys_back.append(back)
                r_masks_orig.append(rmask)

            if r_masks_orig:
                overlay = output.copy()
                for i, rmask in enumerate(r_masks_orig):
                    col = np.zeros_like(overlay); col[:] = colors[i % len(colors)]
                    overlay = np.where(rmask[...,None].astype(bool), overlay + col, overlay)
                output = cv2.addWeighted(overlay, 0.35, output, 0.65, 0)

                for i, poly in enumerate(polys_back):
                    poly_i32 = np.round(poly).astype(np.int32)
                    cv2.polylines(output, [poly_i32], True, colors[i % len(colors)], 3, lineType=cv2.LINE_8)

            # ----- คำนวณขนาด RN ตามแกนวัด -----
            rects_mm_for_result = []
            log_sn = []
            for idx, poly_float in enumerate(polys_back, start=1):
                w_mm_piece, h_mm_piece = _proj_span_mm(
                    np.asarray(poly_float), origin_tl, unit_u, unit_v,
                    self.pixel_mm_ratio_w, self.pixel_mm_ratio_h
                )
                cx, cy = map(int, np.mean(poly_float, axis=0))
                _put(output, f"S{idx}",(cx - 0, cy - 0), 2.5)

                area_px_piece = float(cv2.countNonZero(r_masks_orig[idx-1]))
                area_mm2_piece = area_px_piece * self.pixel_mm_ratio_w * self.pixel_mm_ratio_h
                rects_mm_for_result.append((w_mm_piece, h_mm_piece, area_mm2_piece))

            # ---------- วัด Lost W/H (จากรูจริงๆ เท่านั้น) ----------
            if cv2.countNonZero(clean_lost) > 0:
                xL, yL, wL, hL = cv2.boundingRect(clean_lost)
                lost_corners_warp = np.array(
                    [[xL,yL],[xL+wL,yL],[xL+wL,yL+hL],[xL,yL+hL]], dtype=np.float32
                ).reshape(-1,1,2)
                lost_corners_orig = cv2.perspectiveTransform(lost_corners_warp, Minv).reshape(-1,2)
                lost_w_mm, lost_h_mm = _proj_span_mm(
                    lost_corners_orig, origin_tl, unit_u, unit_v,
                    self.pixel_mm_ratio_w, self.pixel_mm_ratio_h
                )
            else:
                lost_w_mm = lost_h_mm = 0.0

            lost_area_px_warp = float(cv2.countNonZero(clean_lost))
            lost_area_mm2 = lost_area_px_warp * self.pixel_mm_ratio_w * self.pixel_mm_ratio_h


            # ---------- ระบายช่องว่างภายในกรอบใหญ่ (ทึบ) ----------
            rect_mask = np.zeros_like(obj_mask)
            cv2.fillConvexPoly(rect_mask, rect_pts, 255, lineType=cv2.LINE_8)
            void_inside_rect_mask = cv2.bitwise_and(rect_mask, cv2.bitwise_not(obj_mask))
            overlay_void = output.copy()
            overlay_void[void_inside_rect_mask.astype(bool)] = (0, 0, 0)
            output = cv2.addWeighted(overlay_void, 0.35, output, 0.65, 0)

            # ---------- ตัวเลขรวม ----------
            Mmom = cv2.moments(ext_cnt)
            if Mmom["m00"] != 0:
                cX, cY = int(Mmom["m10"]/Mmom["m00"]), int(Mmom["m01"]/Mmom["m00"])
            else:
                cX, cY = 20, 20

            epsilon = 0.01 * cv2.arcLength(ext_cnt, True)
            approx  = cv2.approxPolyDP(ext_cnt, epsilon, True)
            type, class_color = classify_shape(ext_cnt, approx, hole_cnts, net_area_px)

            try:
                _draw_measure_lines(output, rect_pts, big_w_mm, big_h_mm)
            except Exception:
                pass

            Mh = cv2.moments(cv2.convexHull(ext_cnt))
            if Mh["m00"] != 0:
                hX, hY = int(Mh["m10"]/Mh["m00"]), int(Mh["m01"]/Mh["m00"])
            else:
                hX, hY = 20, 20

            # List per-hole lost area below
            y_offset = 150
            if per_hole_details:
                for idx_h, (hid, wmm_h, hmm_h, amm2_h, (cx_o, cy_o)) in enumerate(per_hole_details, start=1):
                    _draw_label(output, f"R{idx_h}: W={wmm_h:.1f} H={hmm_h:.1f} A={amm2_h:.0f}", (cx_o+5, cy_o-5), (255,255,255))

            # _put(output, f"Width:  {big_w_mm:.1f} mm",  (cX - 140, cY - 20), 2.0)
            # _put(output, f"Height: {big_h_mm:.1f} mm", (cX - 140, cY + 14), 2.0)
            
            # ---------- มุมขวาล่าง: แสดง Type + S1..Sn ----------
            lines = [f"Type: {type}"]

            if rects_mm_for_result:  # มี S จึงค่อยแสดง
                # สร้างข้อความ S1..Sn แบบสั้น: S1(W=..,H=..,A=..)
                s_parts = [
                    f"S{i}(W={wmm:.1f} mm,H={hmm:.1f} mm,A={amm2:.0f} mm2)"
                    for i, (wmm, hmm, amm2) in enumerate(rects_mm_for_result, start=1)
                ]
                # แบ่งเป็นหลายบรรทัดเพื่อไม่ให้ยาวเกินไป (เช่น บรรทัดละ 2–3 ชิ้น)
                max_per_line = 2
                s_lines = ["  |  ".join(s_parts[i:i+max_per_line]) for i in range(0, len(s_parts), max_per_line)]
                lines.extend(s_lines)

            # คำนวณขนาดรวม เพื่อจัดชิดขวาล่างให้ทั้งบล็อก
            scale = 2.5
            thick = 4
            h_img, w_img = output.shape[:2]
            margin = 200

            # หา width สูงสุดของทุกบรรทัด
            widths = []
            heights = []
            for txt in lines:
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_PLAIN, scale, thick)
                widths.append(tw); heights.append(th)

            block_w = max(widths) if widths else 0
            line_h = (max(heights) if heights else 0) + 30  # เผื่อระยะห่างระหว่างบรรทัด
            start_x = w_img - block_w - margin
            start_y = h_img - margin

            # วาดจากล่างขึ้นบน เพื่อให้ "Type" อยู่บรรทัดล่างสุดตามตำแหน่งที่คุ้นเคย
            y = start_y
            for txt in reversed(lines):
                _draw_label(output, txt, (start_x, y), (255, 255, 255), scale=scale, thickness=thick)
                y -= line_h

            # --------- สรุปผลลัพธ์ ---------
            return {
                "width_mm":   big_w_mm,
                "height_mm":  big_h_mm,
                "area_mm2":   net_area_mm2,
                "rects_mm":   rects_mm_for_result,  # [(W,H,Area_mm2), ...] ของ R1..RN
                "contour":    ext_cnt,
                "holes":      hole_cnts,
                "type": type,
                "class_color": class_color,
                "lost_area_px":  float(cv2.countNonZero(clean_lost)),
                "lost_area_mm2": lost_area_mm2,
                "lost_width_mm":  lost_w_mm,
                "lost_height_mm": lost_h_mm,
                "lost_boxes_points": lost_boxes_pts_orig,
                "void_width_mm":  0.0,
                "void_height_mm": 0.0
            }, output

        except Exception as e:
            logging.exception("Measurement error")
            return None, frame_bgr