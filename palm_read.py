#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
palm_read.py —— 掌纹结构化提取（Python CLI 版，对应浏览器工具 palm-reader.html）

流程：MediaPipe 手部关键点(21) → 自适应肤色/皱褶分割 → 线条剖面/标记/梯度 →
      拓扑节点 → 财富图案(封闭口袋+高分叉) → 流年 → 事业线纵脊 → 画布标注。
输出：palm.json (结构化纹路，供 AI 受约束解读) + annotated.png (红框断口/青点拓扑/金框财富/蓝框ROI)。

依赖（venv）：numpy opencv-python-headless mediapipe==0.10.14
用法：python palm_read.py <图片路径> [-o <输出目录>]
"""
import sys, os, json, argparse, subprocess
import numpy as np
import cv2

try:
    import mediapipe as mp
    MP_OK = True
except Exception as e:
    mp = None
    MP_OK = False
    sys.stderr.write(
        '[WARN] 未检测到 mediapipe 库，手部关键点(21)检测当前不可用，将尝试自动安装（见下方）。\n'
        '        若自动安装失败，结果降级为「简易分区」，且这并非照片质量问题。也可手动执行：\n'
        '        pip install -r requirements.txt   # 需 mediapipe==0.10.14（>=0.10.15 已移除 mp.solutions.hands 接口）\n'
    )
    sys.stderr.flush()


def _try_install_mediapipe():
    """缺依赖时自举安装 mediapipe，装好并 import 成功返回 True，否则 False。"""
    global mp, MP_OK
    req = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'requirements.txt')
    if os.path.exists(req):
        cmd = [sys.executable, '-m', 'pip', 'install', '-r', req]
    else:
        # requirements.txt 不在脚本旁时（如脚本被拷贝/软链），直接装锁定版本
        cmd = [sys.executable, '-m', 'pip', 'install', 'mediapipe==0.10.14']
    sys.stderr.write('[INFO] 尝试自动安装缺失依赖 mediapipe==0.10.14 ...\n')
    sys.stderr.flush()
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        sys.stderr.write(
            f'[WARN] 自动安装失败：{e}\n'
            '        请手动在该 venv 中执行：pip install -r requirements.txt\n'
        )
        sys.stderr.flush()
        return False
    try:
        import mediapipe as mp  # noqa: F811
        MP_OK = True
        sys.stderr.write('[OK] mediapipe 已自动安装，手部关键点(21)检测可用。\n')
        sys.stderr.flush()
        return True
    except Exception as e:
        sys.stderr.write(f'[WARN] 安装后 import mediapipe 仍失败：{e}\n')
        sys.stderr.flush()
        return False

MAX = 520
MARKCN = {'chain': '锁链纹', 'island': '岛纹', 'break': '断口'}
WEALTH_MEANING = {
    '元宝纹': '掌心封闭如金元宝，传统主财库丰盈、中年易成巨富；封口越严，财越守得住。',
    '田字纹': '田字形封闭纹，主房产/不动产/土地之财，宜从事实业、地产、矿业。',
    '口字纹': '方形封闭纹，主凭口才与专业（培训、咨询、法律）得财。',
    '三角纹': '清晰三角纹，主事业通达、能聚财，掌中大三角尤为吉。',
    '井字纹': '井字形交叉，主事业影响力带来的巨大财富，多为领导管理者。',
    '十字纹': '十字纹（非主线上），主偏财/贵人财；掌心神秘十字主投资直觉敏锐。',
    '星纹': '星/米字纹，主横财、名气财或权力财（依所在丘位而定）。',
    '米字纹': '米字纹（金钱纹），主偏财、意外之财、外财。',
}
# 仅用于判定 clarityLabel/lengthLabel/markType 的结构（文本释义由 AI 基于 knowledge md 生成）
KB_LEN = {'生命线': True, '智慧线': True, '感情线': True, '事业线': True, '太阳线': True}

EDGES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),
         (9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]


# ---------------- 基础图像处理 ----------------
def skin_ok(r, g, b):
    """HSV 宽区间 + RGB 秩序约束（与 JS skinOK 完全一致）。"""
    mx = max(r, g, b); mn = min(r, g, b); c = mx - mn
    if c <= 0:
        h = 0.0
    else:
        if mx == r:
            h = ((g - b) / c) % 6
        elif mx == g:
            h = (b - r) / c + 2
        else:
            h = (r - g) / c + 4
        h *= 60.0
        if h < 0:
            h += 360
    s = (c / mx) if mx > 0 else 0.0
    v = mx / 255.0
    return (h <= 50 or h >= 330) and (0.18 <= s <= 0.70) and (0.20 <= v <= 0.98) and (r >= g >= b) and (c >= 12)


def dilate_mask(mask, r):
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(mask.astype(np.uint8), k)


def clean_small_cc(img, min_area):
    """img: uint8 0/1；去掉面积 < min_area 的连通域。"""
    if img.sum() == 0:
        return img
    num, labels = cv2.connectedComponents(img.astype(np.uint8), connectivity=8)
    out = np.zeros_like(img)
    for i in range(1, num):
        if (labels == i).sum() >= min_area:
            out[labels == i] = 1
    return out


def filter_crease_components(keep, min_area=6, min_elong=2.2, big_area=40):
    """形态学过滤：保留「线」结构、丢弃「团块噪点」，且不切断线的渐淡末端。

    掌纹线是细长、有方向的结构；噪点是又短又团的点块。
    第一遍（形状过滤）：对 keep(0/1) 的每个 8-连通组件：
      - 面积 < min_area               → 删（碎屑噪点）
      - 面积 >= big_area              → 留（线密集区 / 大口袋，避免误删真结构）
      - 细长度 max(bbW,bbH)^2 / 面积 >= min_elong → 留（细长 = 线/弧/口袋边界）
      - 其余                          → 删（又短又团 = 噪点）
    第二遍（组件级生长）：剩余小组件若 8-邻接于已保留的线组件，则并入——
      它是线的渐淡末端 / 分叉尾巴（"能连成线"），而非孤立噪点。
    """
    if keep.sum() == 0:
        return keep
    num, labels = cv2.connectedComponents(keep.astype(np.uint8), connectivity=8)
    out = np.zeros_like(keep)
    for lab in range(1, num):
        comp = labels == lab
        area = int(comp.sum())
        if area < min_area:
            continue
        if area >= big_area:
            out[comp] = 1
            continue
        ys, xs = np.nonzero(comp)
        bw = int(xs.max() - xs.min()) + 1
        bh = int(ys.max() - ys.min()) + 1
        elong = max(bw, bh) ** 2 / float(area)
        if elong >= min_elong:
            out[comp] = 1
    # 组件级生长：邻接线组件的小尾巴并入（迭代，避免连锁漏接）
    for _ in range(2):
        keep_d = cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        added = False
        for lab in range(1, num):
            if out[labels == lab].any():
                continue
            comp = labels == lab
            if (comp & (keep_d > 0)).any():
                out[comp] = 1
                added = True
        if not added:
            break
    return out


def zhang_suen(img):
    """向量化 Zhang-Suen 细化，img: uint8 0/1，返回 uint8 0/1 骨架。"""
    A = np.pad(img.astype(np.uint8), 1, mode='constant')
    changed = True
    while changed:
        changed = False
        p2 = A[0:-2, 1:-1]; p3 = A[0:-2, 2:]; p4 = A[1:-1, 2:]; p5 = A[2:, 2:]
        p6 = A[2:, 1:-1]; p7 = A[2:, 0:-2]; p8 = A[1:-1, 0:-2]; p9 = A[0:-2, 0:-2]
        N = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
        seq = np.stack([p2, p3, p4, p5, p6, p7, p8, p9, p2], axis=0)
        trans = ((seq[:-1] == 0) & (seq[1:] == 1)).sum(axis=0)
        cond = (N >= 2) & (N <= 6) & (trans == 1) & (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0) & (A[1:-1, 1:-1] == 1)
        if cond.any():
            A[1:-1, 1:-1][cond] = 0
            changed = True
        p2 = A[0:-2, 1:-1]; p3 = A[0:-2, 2:]; p4 = A[1:-1, 2:]; p5 = A[2:, 2:]
        p6 = A[2:, 1:-1]; p7 = A[2:, 0:-2]; p8 = A[1:-1, 0:-2]; p9 = A[0:-2, 0:-2]
        N = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
        seq = np.stack([p2, p3, p4, p5, p6, p7, p8, p9, p2], axis=0)
        trans = ((seq[:-1] == 0) & (seq[1:] == 1)).sum(axis=0)
        cond = (N >= 2) & (N <= 6) & (trans == 1) & (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0) & (A[1:-1, 1:-1] == 1)
        if cond.any():
            A[1:-1, 1:-1][cond] = 0
            changed = True
    return A[1:-1, 1:-1]


def eight_neighbor_sum(S):
    """S: uint8 0/1，返回每像素 8 邻域骨架像素数。"""
    Sp = np.pad(S, 1, mode='constant')
    s = (Sp[0:-2, 1:-1] + Sp[0:-2, 2:] + Sp[1:-1, 2:] + Sp[2:, 2:] + Sp[2:, 1:-1] +
         Sp[2:, 0:-2] + Sp[1:-1, 0:-2] + Sp[0:-2, 0:-2])
    return s


# ---------------- 几何 / 区域 ----------------
def build_frame(lm):
    P = lambda i: (lm[i]['x'], lm[i]['y'])
    wrist = P(0)
    mcp = [(P(5)[0] + P(9)[0] + P(13)[0] + P(17)[0]) / 4,
           (P(5)[1] + P(9)[1] + P(13)[1] + P(17)[1]) / 4]
    lx = mcp[0] - wrist[0]; ly = mcp[1] - wrist[1]
    L = (lx * lx + ly * ly) ** 0.5 or 1e-6
    lx /= L; ly /= L
    ax = -ly; ay = lx
    Cx = (wrist[0] + mcp[0]) / 2; Cy = (wrist[1] + mcp[1]) / 2
    thumb = ((P(1)[0] + P(2)[0]) / 2, (P(1)[1] + P(2)[1]) / 2)
    thumb_sign = 1 if ((thumb[0] - Cx) * ax + (thumb[1] - Cy) * ay) >= 0 else -1
    return {'lx': lx, 'ly': ly, 'ax': ax, 'ay': ay, 'Cx': Cx, 'Cy': Cy,
            'L': L, 'thumbSign': thumb_sign, 'landmarks': lm}


def local_uv(frame, nx, ny):
    dx = nx - frame['Cx']; dy = ny - frame['Cy']
    return [(dx * frame['lx'] + dy * frame['ly']) / frame['L'],
            (dx * frame['ax'] + dy * frame['ay']) / frame['L']]


def landmark_regions(fr, W, H):
    ts = fr['thumbSign']
    def mk(u_lo, u_hi, v_pred=None):
        def pred(x, y):
            u, v = local_uv(fr, x / W, y / H)
            if u < u_lo or u > u_hi:
                return False
            if v_pred is not None:
                return v_pred(u, v)
            return True
        return pred
    return {
        'heart': mk(0.10, 0.58, lambda u, v: abs(v) < 0.62),
        'mind': mk(-0.10, 0.32, lambda u, v: abs(v) < 0.62),
        'life': mk(-1.0, 0.25, lambda u, v: (ts * v) > 0.12),
        'fate': mk(-1.0, 0.7, lambda u, v: abs(v) < 0.18),
        'sun': mk(-1.0, -0.35, lambda u, v: abs(v) < 0.28),
    }


def naive_regions(W, H, skin):
    ys, xs = np.where(skin)
    if xs.size == 0:
        return {k: (lambda x, y: False) for k in ('heart', 'mind', 'life', 'fate', 'sun')}
    minX, maxX, minY, maxY = xs.min(), xs.max(), ys.min(), ys.max()
    ph2 = max(1, maxY - minY); pw2 = max(1, maxX - minX); cx = (minX + maxX) / 2
    def heart(x, y): return (y >= minY + ph2 * 0.36 and y < minY + ph2 * 0.66 and minX <= x <= maxX)
    def mind(x, y): return (y >= minY + ph2 * 0.36 and y < minY + ph2 * 0.66 and minX <= x <= maxX)
    def life(x, y): return (y >= minY + ph2 * 0.66 and x < cx)
    def fate(x, y): return abs(x - cx) < pw2 * 0.16
    def sun(x, y): return (y >= minY + ph2 * 0.66 and abs(x - cx) < pw2 * 0.22)
    return {'heart': heart, 'mind': mind, 'life': life, 'fate': fate, 'sun': sun}


def locate_mount(frame, x, y, W, H):
    u, v = local_uv(frame, x / W, y / H)
    wv = frame['thumbSign'] * v
    if u < 0.4 and abs(wv) < 0.3:
        return '明堂'
    if u < 0.4:
        return '金星丘' if wv > 0 else '月丘'
    if wv > 0.22:
        return '木星丘'
    if wv < -0.22:
        return '水星丘'
    return '太阳丘'


def build_palm_roi(frame, W, H):
    L = frame['landmarks']
    if not L or len(L) < 18:
        return None
    base = [(L[i]['x'] * W, L[i]['y'] * H) for i in (0, 5, 9, 13, 17)]
    sx = sum(p[0] for p in base) / len(base); sy = sum(p[1] for p in base) / len(base)
    hand = sum(((p[0] - sx) ** 2 + (p[1] - sy) ** 2) ** 0.5 for p in base) / len(base)
    buf = hand * 0.22
    poly = []
    for (px, py) in base:
        d = ((px - sx) ** 2 + (py - sy) ** 2) ** 0.5 or 1
        poly.append((int(px + (px - sx) / d * buf), int(py + (py - sy) / d * buf)))
    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, [np.array(poly, np.int32)], 1)
    return mask.astype(bool)


# ---------------- 测量函数 ----------------
def region_metric(pred, skin, crease, W, H):
    s_count = c_sum = c_count = 0
    for y in range(H):
        for x in range(W):
            if skin[y, x] and pred(x, y):
                s_count += 1
                c = crease[y, x]
                c_sum += c
                if c > 0:
                    c_count += 1
    if s_count == 0:
        return {'strength': 0.0, 'density': 0.0, 'sCount': 0}
    return {'strength': c_sum / s_count, 'density': c_count / s_count, 'sCount': s_count}


def line_profile(pred, axis, frame, skin, crease, W, H):
    BINS = 24
    t_min = np.inf; t_max = -np.inf; crease_n = 0; total_n = 0
    for y in range(H):
        for x in range(W):
            if not (skin[y, x] and pred(x, y)):
                continue
            total_n += 1
            if crease[y, x] > 0:
                u, v = local_uv(frame, x / W, y / H)
                t = u if axis == 'u' else v
                t_min = min(t_min, t); t_max = max(t_max, t); crease_n += 1
    if crease_n < 5 or (t_max - t_min) < 0.05:
        return {'has': False}
    present = np.zeros(BINS, np.uint8)
    bin_total = np.zeros(BINS, np.float32)
    bin_crease = np.zeros(BINS, np.float32)
    bin_sum_x = np.zeros(BINS, np.float64)
    bin_sum_y = np.zeros(BINS, np.float64)
    bin_cnt = np.zeros(BINS, np.int32)
    for y in range(H):
        for x in range(W):
            if not (skin[y, x] and pred(x, y)):
                continue
            u, v = local_uv(frame, x / W, y / H)
            t = u if axis == 'u' else v
            b = int((t - t_min) / (t_max - t_min) * BINS)
            b = 0 if b < 0 else (BINS - 1 if b >= BINS else b)
            bin_total[b] += 1; bin_sum_x[b] += x; bin_sum_y[b] += y; bin_cnt[b] += 1
            if crease[y, x] > 0:
                present[b] = 1; bin_crease[b] += 1
    runs = max_gap = cur = 0; in_run = False; covered = 0
    for b in range(BINS):
        if present[b]:
            covered += 1
            if not in_run:
                in_run = True; runs += 1
            cur = 0
        else:
            if in_run:
                cur += 1
                if cur > max_gap:
                    max_gap = cur
            else:
                cur = 0
    gap_frac = max_gap / BINS
    frag = (runs - 1) / (covered + 3) if covered > 0 else 1
    density = crease_n / max(1, total_n)
    return {'has': True, 'present': present, 'BINS': BINS, 'tMin': t_min, 'tMax': t_max,
            'gapFrac': gap_frac, 'frag': frag, 'density': density, 'covered': covered,
            'runs': runs, 'lengthRatio': covered / BINS, 'binCrease': bin_crease,
            'binTotal': bin_total, 'binSumX': bin_sum_x, 'binSumY': bin_sum_y, 'binCnt': bin_cnt}


def trace_line_path(pred, axis, frame, skin, crease, W, H, steps=60, smooth=4):
    """沿轴线把区域切成多片，每片取皱褶像素(crease)的加权质心，连成平滑折线。
    这样描出的线追着真实皱褶谷走，而不是区域中线。"""
    ts = []; xs = []; ys = []; cv = []
    for y in range(H):
        for x in range(W):
            if skin[y, x] and pred(x, y):
                u, v = local_uv(frame, x / W, y / H)
                t = u if axis == 'u' else v
                ts.append(t); xs.append(x); ys.append(y); cv.append(crease[y, x])
    if len(cv) < 30:
        return []
    t = np.array(ts, np.float32); x = np.array(xs, np.float32)
    y = np.array(ys, np.float32); c = np.array(cv, np.float32)
    tmin, tmax = float(t.min()), float(t.max())
    if tmax - tmin < 0.04:
        return []
    nb = steps
    edges = np.linspace(tmin, tmax, nb + 1)
    raw = [None] * nb
    for i in range(nb):
        m = (t >= edges[i]) & (t < edges[i + 1])
        cm = c[m]
        s = float(cm.sum())
        if s < 1.0:
            continue
        w = cm
        cx = float((x[m] * w).sum() / s)
        cy = float((y[m] * w).sum() / s)
        raw[i] = (cx, cy)
    # 插值填补空隙（断口处用前后邻点线性补）
    pts = []
    for i in range(nb):
        if raw[i] is not None:
            pts.append(raw[i])
        else:
            prev = nxt = None
            for j in range(i - 1, -1, -1):
                if raw[j] is not None:
                    prev = j; break
            for j in range(i + 1, nb):
                if raw[j] is not None:
                    nxt = j; break
            if prev is not None and nxt is not None:
                f = (i - prev) / (nxt - prev)
                a, b = raw[prev], raw[nxt]
                pts.append((a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f))
            elif prev is not None:
                pts.append(raw[prev])
            elif nxt is not None:
                pts.append(raw[nxt])
    if len(pts) < 4:
        return []
    # 移动平均平滑
    if smooth > 1:
        sw = smooth
        sm = []
        n = len(pts)
        for i in range(n):
            a = max(0, i - sw); b = min(n, i + sw + 1)
            k = b - a
            sm.append((sum(p[0] for p in pts[a:b]) / k, sum(p[1] for p in pts[a:b]) / k))
        pts = sm
    return [[int(round(px)), int(round(py))] for px, py in pts]


def classify_mark(p, line_key):
    if not p or not p['has']:
        return {'type': 'none', 'sev': 0, 'pos': None, 'run': 0, 'label': None, 'b0': None, 'b1': None}
    if p['density'] <= 0.03:
        return {'type': 'none', 'sev': 0, 'pos': None, 'run': 0, 'label': None, 'b0': None, 'b1': None}
    B = p['BINS']; present = p['present']
    best_run = best_start = cur_run = cur_start = -1
    cur_run = 0; cur_start = -1
    for b in range(B):
        if not present[b]:
            if cur_run == 0:
                cur_start = b
            cur_run += 1
            if cur_run > best_run:
                best_run = cur_run; best_start = cur_start
        else:
            cur_run = 0
    if best_start <= 0 or best_start + best_run >= B - 1:
        best_run = 0; best_start = -1
    pos = (best_start + best_run / 2) / B if best_run > 0 else None
    b0 = best_start if best_start >= 0 else None
    b1 = best_start + best_run - 1 if best_start >= 0 else None
    label = None
    if pos is not None:
        if line_key == 'life':
            age_c = round((1 - pos) * 80); half = max(1, round(best_run / B * 80 / 2))
            label = f'约 {age_c} 岁（±{half} 年）'
        else:
            label = '起点侧前段' if pos < 0.34 else ('中段' if pos < 0.67 else '末端侧后段')
    if best_run >= 4:
        return {'type': 'break', 'sev': 3, 'pos': pos, 'run': best_run, 'label': label, 'b0': b0, 'b1': b1}
    if best_run >= 2:
        return {'type': 'island', 'sev': 2, 'pos': pos, 'run': best_run, 'label': label, 'b0': b0, 'b1': b1}
    if p['frag'] >= 0.50:
        return {'type': 'chain', 'sev': 1, 'pos': None, 'run': 0, 'label': None, 'b0': None, 'b1': None}
    return {'type': 'clean', 'sev': 0, 'pos': None, 'run': 0, 'label': None, 'b0': None, 'b1': None}


def depth_gradient(p, line_key, frame):
    if not p or not p['has']:
        return None
    B = p['BINS']; bc = p['binCrease']; bt = p['binTotal']
    q = np.where(bt > 2, bc / bt, -1.0)
    q1 = max(1, int(B * 0.25)); q3 = max(1, int(B * 0.75))

    def win(a, b):
        w_sum = w_w = n = pc = 0
        for i in range(a, b):
            if q[i] >= 0:
                w_sum += q[i] * bt[i]; w_w += bt[i]; n += 1
            if p['present'][i]:
                pc += 1
        return {'avg': (w_sum / w_w if w_w >= 5 else None), 'cov': n / (b - a), 'pc': pc / (b - a)}
    w0 = win(0, q1); w1 = win(q3, B); wm = win(0, B)
    if w0['avg'] is None or w1['avg'] is None or wm['avg'] is None:
        return None
    f0 = w0['avg']; f1 = w1['avg']; m = wm['avg']
    reliable = w0['pc'] >= 0.4 and w1['pc'] >= 0.4
    thumb_last = frame['thumbSign'] > 0 if frame else True
    mp = {
        'life': ('f1', 'f0', '起点（指端 / 幼年）', '末端（腕端 / 老年）'),
        'fate': ('f0', 'f1', '起点（腕端）', '末端（指端）'),
        'sun': ('f0', 'f1', '起点（腕端）', '末端（指端）'),
        'mind': ('f1' if thumb_last else 'f0', 'f0' if thumb_last else 'f1', '起点（拇指侧）', '末端（小指侧）'),
        'heart': ('f0' if thumb_last else 'f1', 'f1' if thumb_last else 'f0', '起点（小指侧）', '末端（拇指侧）'),
    }.get(line_key, ('f0', 'f1', '起端', '末端'))
    sq = f1 if mp[0] == 'f1' else f0
    eq = f1 if mp[1] == 'f1' else f0
    mx = max(sq, eq, m, 1e-6); sq_n = sq / mx; eq_n = eq / mx; m_n = m / mx
    tol = 0.16
    if abs(sq_n - eq_n) < tol:
        typ = 'uniform'; desc = '深浅均匀（全程凹陷强度一致）'
    elif sq_n >= m_n and eq_n < sq_n - tol:
        typ = 'fadeEnd'; desc = f'{mp[2]}深、{mp[3]}渐浅'
    elif eq_n >= m_n and sq_n < eq_n - tol:
        typ = 'deepenEnd'; desc = f'{mp[3]}渐深（{mp[2]}相对浅）'
    elif m_n < min(sq_n, eq_n) - tol:
        typ = 'midShallow'; desc = '中段较两端明显变浅，首尾相对深'
    elif sq_n > eq_n:
        typ = 'fadeEnd'; desc = f'{mp[2]}深、{mp[3]}渐浅'
    else:
        typ = 'deepenEnd'; desc = f'{mp[3]}渐深（{mp[2]}相对浅）'
    if not reliable:
        desc += '（末端/起点样本偏少，梯度结论仅供参考）'
    return {'type': typ, 'desc': desc, 'startQ': round(float(sq), 3), 'endQ': round(float(eq), 3),
            'midQ': round(float(m), 3), 'startN': round(float(sq_n), 2), 'endN': round(float(eq_n), 2),
            'reliable': reliable}


def career_spine(frame, skin, crease, W, H):
    if not frame:
        return None
    u0, u1, v_half, BINS = -0.95, 0.55, 0.085, 40
    bins = [{'mx': 0, 'sum': 0, 'n': 0} for _ in range(BINS)]
    for y in range(H):
        for x in range(W):
            if not skin[y, x]:
                continue
            u, v = local_uv(frame, x / W, y / H)
            if u < u0 or u > u1 or v < -v_half or v > v_half:
                continue
            b = int((u - u0) / (u1 - u0) * BINS)
            b = 0 if b < 0 else (BINS - 1 if b >= BINS else b)
            c = crease[y, x]; bins[b]['sum'] += c; bins[b]['n'] += 1
            if c > bins[b]['mx']:
                bins[b]['mx'] = c
    n = spine_hit = bg_sum = hit = run = max_run = 0
    for b in bins:
        if b['n'] > 0:
            avg = b['sum'] / b['n']; sig = b['mx'] / max(1, avg)
            n += 1; bg_sum += avg
            if b['mx'] > 13 and sig > 1.7:
                hit += 1; run += 1; spine_hit += b['mx']
                if run > max_run:
                    max_run = run
            else:
                run = 0
    if n == 0:
        return None
    spine_mean = spine_hit / hit if hit > 0 else 0
    bg_mean = bg_sum / n; coverage = hit / n
    cont = max_run / hit if hit > 0 else 0
    if coverage >= 0.5 and spine_mean > 15:
        visible = True; label = '纵脊清晰'
    elif coverage >= 0.28 and spine_mean > 13:
        visible = True; label = '纵脊断续'
    else:
        visible = False; label = '纵脊微弱'
    return {'visible': visible, 'label': label, 'spineMean': round(spine_mean, 1),
            'bgMean': round(bg_mean, 1), 'coverage': round(coverage, 2),
            'contRatio': round(cont, 2), 'maxRun': max_run, 'hit': hit}


def detect_topology(pred, skin, vis_mask, W, H):
    if vis_mask is None:
        return {'nodes': 0, 'bifurcations': 0, 'endings': 0, 'points': []}
    sk = np.zeros((H, W), np.uint8)
    n = 0
    for y in range(H):
        for x in range(W):
            if skin[y, x] and vis_mask[y, x] and pred(x, y):
                sk[y, x] = 1; n += 1
    if n < 24:
        return {'nodes': 0, 'bifurcations': 0, 'endings': 0, 'points': []}
    sk = clean_small_cc(sk, 12)
    skel = zhang_suen(sk)
    deg = eight_neighbor_sum(skel)
    nodes = bifurcations = endings = 0
    points = []
    ys, xs = np.where(skel > 0)
    for y, x in zip(ys, xs):
        cs = int(deg[y, x])
        if cs == 1:
            endings += 1
        elif cs >= 3:
            nodes += 1; bifurcations += 1; points.append({'x': int(x), 'y': int(y)})
    return {'nodes': nodes, 'bifurcations': bifurcations, 'endings': endings, 'points': points}


def detect_wealth_marks(frame, skin, vis_mask, crease, W, H):
    roi = build_palm_roi(frame, W, H)
    roi_mask = roi if roi is not None else np.ones((H, W), bool)
    marks = []
    gmask = (skin & vis_mask & roi_mask).astype(np.uint8)
    gn = int(gmask.sum())
    deg = None
    if gn >= 40:
        gmask = clean_small_cc(gmask, 12)
        skel = zhang_suen(gmask)
        deg = eight_neighbor_sum(skel)
    # A. 封闭口袋
    barrier = dilate_mask(vis_mask, 2)
    bg = (skin & (barrier == 0) & roi_mask).astype(np.uint8)
    padded = np.pad(bg, 1, constant_values=1)
    fmask = np.zeros((H + 4, W + 4), np.uint8)
    cv2.floodFill(padded, fmask, (0, 0), 2)
    reached = padded[1:-1, 1:-1] == 2
    pocket = (bg == 1) & (~reached)
    if pocket.sum() > 0:
        num, labels = cv2.connectedComponents(pocket.astype(np.uint8), connectivity=8)
        barrier_sum = cv2.filter2D(barrier.astype(np.float32), -1, np.ones((3, 3)))
        open_mask = reached | (~skin.astype(bool))
        open_sum = cv2.filter2D(open_mask.astype(np.float32), -1, np.ones((3, 3)))
        for i in range(1, num):
            comp = (labels == i)
            area = int(comp.sum())
            if area == 0:
                continue
            ys, xs = np.where(comp)
            cx = float(xs.mean()); cy = float(ys.mean())
            loc = locate_mount(frame, cx, cy, W, H)
            bn = float(barrier_sum[comp].sum()); on = float(open_sum[comp].sum())
            enc = bn / (bn + on) if (bn + on) > 0 else 0
            has_cross = bool(deg is not None and (deg[comp] >= 4).any())
            if area >= 25 and enc >= 0.45:
                typ = '田字纹' if has_cross else ('元宝纹' if loc == '明堂' else '三角纹')
                conf = 0.7 if enc >= 0.65 else (0.55 if enc >= 0.55 else 0.4)
                marks.append({'type': typ, 'location': loc, 'confidence': round(conf, 2),
                              'detail': f'封闭区{area}px·封口率{int(enc * 100)}%', 'hasCross': has_cross,
                              'cx': round(cx, 1), 'cy': round(cy, 1)})
    # B. 高分叉/放射节点
    if deg is not None:
        seeds_y, seeds_x = np.where(deg >= 4)
        used = np.zeros((H, W), np.uint8)
        raw = []
        for y, x in zip(seeds_y, seeds_x):
            if used[y, x]:
                continue
            stack = [(y, x)]; used[y, x] = 1; cxs = cys = ncc = maxd = 0
            while stack:
                jy, jx = stack.pop(); cxs += jx; cys += jy; ncc += 1
                if deg[jy, jx] > maxd:
                    maxd = int(deg[jy, jx])
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        aa = jx + dx; bb = jy + dy
                        if 0 <= aa < W and 0 <= bb < H and not used[bb, aa] and deg[bb, aa] >= 4 and abs(aa - jx) <= 14 and abs(bb - jy) <= 14:
                            used[bb, aa] = 1; stack.append((bb, aa))
            cx = cxs / ncc; cy = cys / ncc
            raw.append({'cx': cx, 'cy': cy, 'maxd': maxd, 'loc': locate_mount(frame, cx, cy, W, H)})
        merged = []
        for c in raw:
            hit = False
            for m in merged:
                if m['loc'] == c['loc'] and ((c['cx'] - m['cx']) ** 2 + (c['cy'] - m['cy']) ** 2) ** 0.5 <= 25:
                    if c['maxd'] > m['maxd']:
                        m['maxd'] = c['maxd']
                    hit = True
                    break
            if not hit:
                merged.append({'cx': c['cx'], 'cy': c['cy'], 'maxd': c['maxd'], 'loc': c['loc']})
        per_mount = {}
        for c in merged:
            if per_mount.get(c['loc'], 0) >= 2:
                continue
            per_mount[c['loc']] = per_mount.get(c['loc'], 0) + 1
            busy = sum(1 for o in merged if o is not c and ((o['cx'] - c['cx']) ** 2 + (o['cy'] - c['cy']) ** 2) ** 0.5 <= 40)
            typ = '星纹' if c['maxd'] >= 5 else '井字纹'
            conf = 0.6 if c['maxd'] >= 6 else (0.45 if c['maxd'] >= 5 else 0.35)
            if busy >= 2:
                conf = min(conf, 0.3)
            marks.append({'type': typ, 'location': c['loc'], 'confidence': round(conf, 2),
                          'detail': f'节点放射度{c["maxd"]}' + ('·交汇密集区' if busy >= 2 else ''),
                          'hasCross': c['maxd'] >= 4, 'cx': round(c['cx'], 1), 'cy': round(c['cy'], 1)})
    return marks


def life_timeline(p):
    if not p or not p['has']:
        return None
    B = p['BINS']; present = p['present']; bc = p['binCrease']; bt = p['binTotal']
    tot_c = tot_t = 0
    for b in range(B):
        tot_c += bc[b]; tot_t += bt[b]
    avg_q = tot_c / tot_t if tot_t else 0
    age_at = lambda b: (p['tMax'] - (p['tMin'] + (b + 0.5) / B * (p['tMax'] - p['tMin']))) / (p['tMax'] - p['tMin']) * 80
    N = 8; per = max(1, B // N)
    segs = []
    for s in range(N):
        c = t = pres = 0; b0 = s * per; b1 = min(B, (s + 1) * per)
        for b in range(b0, b1):
            c += bc[b]; t += bt[b]; pres += present[b]
        q = c / t if t else 0; cov = pres / (b1 - b0)
        st = 'low' if (cov < 0.45 or q < avg_q * 0.5) else ('good' if (q > avg_q * 1.25 and cov > 0.6) else 'mid')
        a0 = round(age_at(b0)); a1 = round(age_at(b1 - 1))
        segs.append({'lo': min(a0, a1), 'hi': max(a0, a1), 'st': st, 'q': round(float(q), 3), 'cov': round(float(cov), 2)})
    MIN_BREAK = 2
    breaks = []; in_gap = False; gs = -1
    for b in range(B):
        if not present[b]:
            if not in_gap:
                in_gap = True; gs = b
        else:
            if in_gap:
                w = b - gs
                if w >= MIN_BREAK and gs >= 1:
                    breaks.append({'b0': gs, 'b1': b - 1, 'age0': round(age_at(gs)), 'age1': round(age_at(b))})
                in_gap = False
    if in_gap:
        w = B - gs
        if w >= MIN_BREAK and gs >= 1:
            breaks.append({'b0': gs, 'b1': B - 1, 'age0': round(age_at(gs)), 'age1': round(age_at(B))})
    return {'segs': segs, 'breaks': breaks, 'avgQ': round(float(avg_q), 3)}


# ---------------- 进阶纹向检测（川字掌 / 双太阳线 / 感情线羽毛纹） ----------------
# 说明：以下均为基于区域皱褶掩码 + 手部关键点的几何启发式，非医学/相学权威判定。
# 只输出原始几何量 + 中/低置信，解读时须诚实带过，不得伪装成精确结论。

def _region_crease_mask(pred, skin, crease, W, H):
    """返回该线区域内「肤色 ∩ 皱褶」的布尔掩码 (H,W)。"""
    mask = np.zeros((H, W), np.uint8)
    for y in range(H):
        for x in range(W):
            if pred(x, y):
                mask[y, x] = 1
    return (mask.astype(bool) & skin.astype(bool) & (crease > 0))


def _hand_scale_px(frame, W, H):
    """以手腕(0)到中指根(9)的像素距离作为「一手长」归一化尺度。"""
    L = frame['landmarks']
    if not L or len(L) < 10:
        return max(W, H) * 0.4
    x0, y0 = L[0]['x'] * W, L[0]['y'] * H
    x9, y9 = L[9]['x'] * W, L[9]['y'] * H
    return max(1.0, ((x0 - x9) ** 2 + (y0 - y9) ** 2) ** 0.5)


def _skeleton(mask):
    """形态学骨架（细线化），用于在细线上正确判定端点/分叉点，避免把粗线内部像素误判。"""
    img = mask.astype(np.uint8)
    if img.sum() == 0:
        return img
    skel = np.zeros_like(img)
    elem = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(img, elem)
        temp = cv2.dilate(eroded, elem)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skel


def _pinky_end_centroid(mask, frame, W, H):
    """该线 pinky 侧（thumbSign*v 最小）那 25% 皱褶像素的质心，用于绘图标注。"""
    if mask.sum() < 8:
        return None
    ws = np.full((H, W), np.inf)
    for y in range(H):
        for x in range(W):
            if mask[y, x]:
                u, v = local_uv(frame, x / W, y / H)
                ws[y, x] = frame['thumbSign'] * v
    wmin = ws[mask].min(); wmax = ws[mask].max()
    th = wmin + (wmax - wmin) * 0.25
    sel = mask & (ws <= th)
    ys, xs = np.where(sel)
    if xs.size == 0:
        return None
    return {'x': int(xs.mean()), 'y': int(ys.mean())}


def chuan_analysis(frame, skin, crease, regs, W, H):
    """川字掌 / 起点关系：检测 智慧线(mind) 与 感情线(heart) 起点是否相连。
    川字掌 = 三主线各自独立，典型特征是智慧线与感情线起点不相连（存在明显间隙）。

    注意：mind/heart 区域在 u∈(0.10,0.32) 有重叠，直接用距离变换会把重叠区算成「间距 0」。
    故先用「互斥掩码」（各自扣掉对方）测真实间隙，避免区域重叠污染。"""
    scale = _hand_scale_px(frame, W, H)
    mm = _region_crease_mask(regs['mind'], skin, crease, W, H)
    hm = _region_crease_mask(regs['heart'], skin, crease, W, H)
    lm = _region_crease_mask(regs['life'], skin, crease, W, H)
    if mm.sum() < 20 or hm.sum() < 20:
        return None
    # 互斥掩码：去掉与对方重叠的区域，仅保留各自独有部分
    mm_only = mm & ~hm
    hm_only = hm & ~mm
    if mm_only.sum() < 8 or hm_only.sum() < 8:
        mh_gap = 0.0  # 两线在独有区域几乎不存 → 视为相连
    else:
        dt = cv2.distanceTransform((~mm_only).astype(np.uint8), cv2.DIST_L2, 5)
        mh_gap = float(dt[hm_only].min()) / scale
    if lm.sum() < 20:
        lm_gap = 0.0
    else:
        lm_only = lm & ~mm
        if lm_only.sum() < 8:
            lm_gap = 0.0
        else:
            dt2 = cv2.distanceTransform((~lm_only).astype(np.uint8), cv2.DIST_L2, 5)
            lm_gap = float(dt2[mm_only].min()) / scale
    # 阈值：间距 > 0.08 手长 视为「分开」（川字特征）；否则相连
    mh_sep = mh_gap > 0.08
    lm_sep = lm_gap > 0.08
    is_chuan = bool(mh_sep and lm_sep)
    conf = '中' if (mh_gap > 0.12 or mh_gap < 0.04) else '低'
    return {
        'mindHeartGap': round(float(mh_gap), 3),
        'lifeMindGap': round(float(lm_gap), 3),
        'mindHeartSeparate': bool(mh_sep),
        'lifeMindSeparate': bool(lm_sep),
        'isChuan': is_chuan,
        'confidence': conf,
        'mindEnd': _pinky_end_centroid(mm, frame, W, H),
        'heartEnd': _pinky_end_centroid(hm, frame, W, H),
    }


def sun_double_line(frame, skin, crease, regs, W, H):
    """双太阳线：太阳线区域内是否存在两条并列、细长、竖直延展的皱褶主脊。"""
    scale = _hand_scale_px(frame, W, H)
    sm = _region_crease_mask(regs['sun'], skin, crease, W, H)
    if sm.sum() < 20:
        return {'isDouble': False, 'count': 0, 'components': [], 'confidence': '低'}
    num, labels = cv2.connectedComponents(sm.astype(np.uint8), connectivity=8)
    comps = []
    best_len = 0.0
    for lab in range(1, num):
        comp = (labels == lab)
        ys, xs = np.where(comp)
        if ys.size < 8:
            continue
        hgt = int(ys.max() - ys.min()); wid = int(xs.max() - xs.min())
        aspect = hgt / max(1, wid)
        len_uv = hgt / scale
        if len_uv > best_len:
            best_len = len_uv
        if len_uv > 0.12 and aspect > 1.0:  # 细长、竖直延展足够（放宽以容纳真实单/双太阳线）
            comps.append({'len': round(float(len_uv), 3), 'aspect': round(float(aspect), 2),
                          'cx': int(xs.mean()), 'cy': int(ys.mean())})
    is_double = len(comps) >= 2
    return {'isDouble': bool(is_double), 'count': len(comps), 'bestLen': round(float(best_len), 3),
            'components': comps, 'confidence': '中' if is_double else '低'}


def heart_feathering(frame, skin, crease, regs, W, H):
    """感情线羽毛纹：感情线 pinky 侧末端是否存在短小分叉/支线（羽毛状）。
    羽毛/树枝状末端传统视为情感外放、人缘好；此处仅做几何检出，置信偏低。

    做法：对整条感情线骨架化；主末端 = pinky 侧端点。在主末端周围 ROI 内，
    取「皱褶减骨架」= 线宽光晕；真正羽毛是**从末端向外分离出来的独立小支**
    （不与末端相连的独立连通块）。主线自身的线宽光晕连在末端、不算。
    干净收尾：无独立小支 → 无羽毛；有 → 羽毛（可能多条）。"""
    hm = _region_crease_mask(regs['heart'], skin, crease, W, H)
    if hm.sum() < 20:
        return None
    skel = _skeleton(hm)
    if skel.sum() < 4:
        return {'has': False, 'forks': 0, 'tips': 0, 'tipPts': [], 'confidence': '低'}
    k = np.ones((3, 3), np.uint8)
    neigh = cv2.filter2D(skel, -1, k)
    degree = neigh - 1
    tip_mask = (degree == 1)
    ty, tx = np.where(tip_mask)
    if tx.size == 0:
        return {'has': False, 'forks': 0, 'tips': 0, 'tipPts': [], 'confidence': '低'}
    ws_tip = np.array([frame['thumbSign'] * local_uv(frame, x / W, y / H)[1] for x, y in zip(tx, ty)])
    mi = int(np.argmin(ws_tip))
    mx, my = int(tx[mi]), int(ty[mi])
    scale = _hand_scale_px(frame, W, H)
    R = max(6, int(0.06 * scale))
    # ROI 圆盘（全图布尔掩码）
    yy, xx = np.mgrid[0:H, 0:W]
    roi = ((xx - mx) ** 2 + (yy - my) ** 2) <= R * R
    off = (hm & ~skel) & roi  # 线宽光晕（ROI 内）
    if off.sum() < 4:
        return {'has': False, 'forks': 0, 'tips': 0, 'tipPts': [{'x': mx, 'y': my}], 'confidence': '低'}
    num, labels = cv2.connectedComponents(off.astype(np.uint8), connectivity=8)
    max_branch = max(8, int(hm.sum() * 0.02))
    exclude_d = max(3.0, 0.02 * scale)
    branches = []
    for lab in range(1, num):
        comp = (labels == lab)
        sz = int(comp.sum())
        if not (2 <= sz <= max_branch):
            continue
        cys, cxs = np.where(comp)
        # 该小支到主末端的最近距离：很近 → 属主线自身线宽，排除
        cd2 = (cxs - mx) ** 2 + (cys - my) ** 2
        if cd2.min() ** 0.5 < exclude_d:
            continue
        branches.append({'x': int(cxs.mean()), 'y': int(cys.mean()), 'size': sz})
    has = len(branches) >= 1
    tip_pts = [{'x': mx, 'y': my}] + [{'x': b['x'], 'y': b['y']} for b in branches]
    return {'has': bool(has), 'forks': len(branches), 'tips': len(branches),
            'tipPts': tip_pts, 'confidence': '中' if has else '低'}


def heart_end_trend(frame, heart_pred, skin, crease, W, H):
    pts = []
    for y in range(H):
        for x in range(W):
            if skin[y, x] and heart_pred(x, y) and crease[y, x] > 0:
                u, v = local_uv(frame, x / W, y / H)
                pts.append({'u': u, 'w': frame['thumbSign'] * v})
    n = len(pts)
    if n < 14:
        return None
    pts.sort(key=lambda p: p['u'])
    k = max(2, n // 7)  # 与 JS 0.15 比例接近
    ws = sum(p['w'] for p in pts[:k]) / k
    we = sum(p['w'] for p in pts[-k:]) / k
    us = pts[0]['u']; ue = pts[-1]['u']
    sl = (we - ws) / max(1e-3, ue - us)
    m = int(n * 0.7)
    dev = sum(p['w'] - (ws + sl * (p['u'] - us)) for p in pts[m:]) / (n - m)
    typ = 'up' if dev > 0.08 else ('down' if dev < -0.08 else 'flat')
    return {'type': typ, 'dev': round(float(dev), 3)}


def gap_centroid(p, b0, b1):
    if not p or not p['has'] or b0 is None:
        return None
    B = p['BINS']; sx = p['binSumX']; sy = p['binSumY']; sc = p['binCnt']
    x = y = n = 0
    for b in range(max(0, b0 - 1), min(B - 1, b1 + 1) + 1):
        if sc[b] > 0:
            x += sx[b]; y += sy[b]; n += sc[b]
    return {'x': x / n, 'y': y / n} if n > 0 else None


def line_confidence(p, has_frame, grad):
    if not has_frame or not p or not p['has']:
        return {'lvl': '低', 'note': '无AI关键点，仅区域清晰度'}
    cov = p['covered'] / p['BINS']; dens = p['density']
    if cov < 0.12:
        return {'lvl': '低', 'note': '区域覆盖不足'}
    if grad and grad.get('reliable') is False:
        if cov > 0.6 and dens > 0.03:
            return {'lvl': '中', 'note': '覆盖充分但端点梯度不可靠'}
    if cov > 0.6 and dens > 0.03:
        return {'lvl': '高', 'note': '覆盖充分且纹理清晰'}
    if cov > 0.25 and dens > 0.015:
        return {'lvl': '中', 'note': '检测一般'}
    return {'lvl': '低', 'note': '纹理过淡或覆盖不足'}


def clarity_of(m, avg_s, avg_d):
    if m['strength'] > avg_s * 1.3 or m['density'] > avg_d * 1.3:
        return 'deep'
    if m['strength'] > avg_s * 0.7:
        return 'mid'
    return 'shallow'


def interpret(name, m, mark, len_ratio, with_ai, avg_s, avg_d):
    c = clarity_of(m, avg_s, avg_d)
    mk = mark['type'] if mark and mark['type'] in ('chain', 'island', 'break') else 'none'
    clarity_label = {'deep': '深长清晰', 'mid': '中等', 'shallow': '浅淡'}[c]
    length_label = None
    if KB_LEN.get(name):
        length_label = (('绵长' if len_ratio >= 0.6 else ('适中' if len_ratio >= 0.42 else '偏短')) if with_ai else '长度依图估算')
    return {'clarityLabel': clarity_label, 'lengthLabel': length_label, 'markType': mk}


# ---------------- 标注绘制 ----------------
def draw_annotations(img, cards, prof, topo, life_tl, wealth_marks, palm_roi, W, H, landmarks=None,
                    crease=None, vis_mask=None, extras=None):
    S = max(9, int(min(W, H) * 0.028))
    out = img.copy()
    # 金色高亮：检测到的掌纹皱褶（肤色分割 + 纹理凹陷）。与浏览器版一致：
    # 每个皱褶像素按强度算透明度，向金色 (BGR 120,186,222) 混合。骨架随后叠加其上。
    if crease is not None:
        cm = crease.astype(np.float32)
        a = np.clip(0.5 + cm / 100.0, 0.0, 0.95)
        if vis_mask is not None:
            m = vis_mask.astype(bool) & (cm > 0)
        else:
            m = cm > 0
        gold = np.array([120, 186, 222], dtype=np.float32)
        out_f = out.astype(np.float32)
        out_f[m] = out_f[m] * (1 - a[m, None]) + gold * a[m, None]
        out = out_f.astype(np.uint8)
    # 青色 AI 骨架（21 关键点 + 连线），让手部朝向/分区可见
    if landmarks is not None:
        pts = [(int(p['x'] * W), int(p['y'] * H)) for p in landmarks]
        for (a, b) in EDGES:
            cv2.line(out, pts[a], pts[b], (255, 255, 0), 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(out, p, 3, (255, 255, 0), -1)
    if palm_roi is not None:
        cnts, _ = cv2.findContours(palm_roi.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (120, 170, 230), 2)
    # 注：主线的彩色折线（trace）经实测与真实掌纹贴合度差、易误导，已按用户要求不再叠加到图上。
    # 识别结论以 palm.json 的数据为准；图面只保留准确的骨架 / ROI / 财富图案 / 拓扑 / 断口标注。
    for c in cards:
        m = c['mark']
        if m and m['type'] in ('break', 'island') and m['b0'] is not None and prof.get(c['key']):
            pt = gap_centroid(prof[c['key']], m['b0'], m['b1'])
            if pt:
                col = (0, 0, 255) if m['type'] == 'break' else (0, 165, 255)
                cv2.rectangle(out, (int(pt['x'] - S), int(pt['y'] - S)), (int(pt['x'] + S), int(pt['y'] + S)), col, 2)
                cv2.putText(out, '断' if m['type'] == 'break' else '岛', (int(pt['x'] - S), int(pt['y'] - S - 7)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        tp = topo.get(c['key'], {}).get('points') if topo else None
        if tp:
            for p in tp[:8]:
                cv2.circle(out, (p['x'], p['y']), 3, (224, 208, 127), -1)
    if life_tl and life_tl['breaks'] and prof.get('life'):
        for br in life_tl['breaks']:
            pt = gap_centroid(prof['life'], br['b0'], br['b1'])
            if pt:
                cv2.circle(out, (int(pt['x']), int(pt['y'])), 3, (0, 0, 255), -1)
    if wealth_marks:
        for mk in wealth_marks[:8]:
            if mk.get('cx') is not None:
                cx = int(mk['cx']); cy = int(mk['cy'])
                cv2.rectangle(out, (cx - S, cy - S), (cx + S, cy + S), (160, 210, 232), 2)
                cv2.putText(out, (mk['type'] or '财')[0], (cx - S, cy - S - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 210, 232), 1, cv2.LINE_AA)
    # 进阶纹向标记：川字掌间隙、双太阳线、感情线羽毛末端
    if extras:
        ch, sd, hf = extras.get('chuan'), extras.get('sunDouble'), extras.get('heartFeather')
        if ch and ch.get('mindHeartSeparate'):
            for key in ('mindEnd', 'heartEnd'):
                p = ch.get(key)
                if p:
                    cv2.circle(out, (p['x'], p['y']), 4, (255, 0, 255), -1)
            if ch.get('isChuan'):
                cv2.putText(out, '川', (ch['mindEnd']['x'] + 6, ch['mindEnd']['y'] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2, cv2.LINE_AA)
        if sd and sd.get('components'):
            for comp in sd['components']:
                cv2.circle(out, (comp['cx'], comp['cy']), 4, (255, 200, 0), -1)
            if sd.get('isDouble'):
                cv2.putText(out, '双太阳', (sd['components'][0]['cx'] + 6, sd['components'][0]['cy'] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2, cv2.LINE_AA)
        if hf and hf.get('tipPts'):
            for p in hf['tipPts']:
                cv2.circle(out, (p['x'], p['y']), 3, (0, 255, 255), -1)
            if hf.get('has'):
                cv2.putText(out, '羽', (hf['tipPts'][0]['x'] + 5, hf['tipPts'][0]['y'] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
    return out


# ---------------- 主流程 ----------------
def analyze(img_path):
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise ValueError(f'无法读取图片：{img_path}')
    ph, pw = bgr.shape[:2]
    scale = min(1.0, MAX / max(pw, ph))
    W = max(1, round(pw * scale)); H = max(1, round(ph * scale))
    bgr = cv2.resize(bgr, (W, H))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # 向量化肤色分割（与 JS skinOK 一致）
    R = rgb[:, :, 0].astype(np.int32); G = rgb[:, :, 1].astype(np.int32); B = rgb[:, :, 2].astype(np.int32)
    mx = np.maximum.reduce([R, G, B]).astype(np.float64)
    mn = np.minimum.reduce([R, G, B]).astype(np.float64)
    c = mx - mn
    nz = c > 0
    h = np.zeros((H, W), np.float64)
    hr = np.where((mx == R) & nz, ((G - B).astype(np.float64) / np.where(nz, c, 1)), 0.0)
    hg = np.where((mx == G) & nz, ((B - R).astype(np.float64) / np.where(nz, c, 1)) + 2, 0.0)
    hb = np.where((mx == B) & nz, ((R - G).astype(np.float64) / np.where(nz, c, 1)) + 4, 0.0)
    h = np.where(mx == R, hr, np.where(mx == G, hg, hb)) * 60.0
    h = np.where(h < 0, h + 360, h)
    s = np.where(mx > 0, c / mx, 0.0)
    v = mx / 255.0
    skin = (((h <= 50) | (h >= 330)) & (s >= 0.18) & (s <= 0.70) & (v >= 0.20) & (v <= 0.98)
            & (R >= G) & (G >= B) & (c >= 12)).astype(np.uint8)
    # 向量化皱褶（B-3 对比度增强 + B-1 多尺度脊线增强，提升细纹/浅淡纹读取）
    lum = (0.299 * R + 0.587 * G + 0.114 * B).astype(np.float32)
    # --- B-3：仅皮肤区做限制对比度直方图均衡(CLAHE)，拉起浅淡纹路信号，
    #     使其在自适应阈值前就被抬高，避免被切掉（这正是图2感情线"浅淡"的主因之一）。---
    lum_u8 = np.clip(lum, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    skin_mask = skin.astype(bool)
    lum_eq = lum_u8.copy()
    lum_eq[skin_mask] = clahe.apply(lum_u8)[skin_mask]
    lum = lum_eq.astype(np.float32)
    # --- B-1：多尺度差-of-高斯(DoG)。单尺度 9x9 只能检出≈9px 量级纹路，更细的掌纹直接被模糊掉。
    #     改为 5/9/15 三尺度，逐像素取最大响应：小尺度保细纹、大尺度保粗纹，互补覆盖。零额外依赖。---
    diff5 = cv2.blur(lum, (5, 5)) - lum
    diff9 = cv2.blur(lum, (9, 9)) - lum
    diff15 = cv2.blur(lum, (15, 15)) - lum
    diff_flat = np.maximum.reduce([diff5, diff9, diff15])
    vals = diff_flat[skin_mask]
    dmean = float(vals.mean()) if vals.size else 0.0
    dstd = float(vals.std()) if vals.size else 0.0
    crease_th = max(4, min(30, dmean + 1.4 * dstd))
    lum_ok = (lum > 20) & (lum < 245)
    strong = skin_mask & (diff_flat > crease_th) & lum_ok
    # 连通域生长：弱皱褶(>0.5*th)若与强皱褶 8-连通，则保留(细线尾巴连着主线)；
    # 孤立弱噪声(组件内无强像素)丢弃，避免满屏噪点。这样生命线等细处也能读出。
    weak_th = max(3.0, crease_th * 0.5)
    weak = skin_mask & (diff_flat > weak_th) & lum_ok
    union = (strong | weak).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(union, connectivity=8)
    keep = np.zeros((H, W), np.uint8)
    for lab in range(1, num_labels):
        comp = labels == lab
        if (comp & strong).any():
            keep |= comp.astype(np.uint8)
    # 第二道：形态学形状过滤——去掉「又短又团」的噪点组件，保留「细长」的线结构。
    # 线判定(line_profile/region_metric)、拓扑、财富图案、金色高亮全部统一用过滤后的 keep2。
    keep2 = filter_crease_components(keep)
    # 金色高亮层(draw_annotations)直接以本 crease 为唯一数据源绘制，二者同源、天然一致，切勿改用其他中间量。
    crease = np.where(keep2 > 0, diff_flat, 0.0).astype(np.float32)
    vis_mask = dilate_mask(keep2, 1)

    # 手部关键点
    frame = None; hand_info = None; skeleton_drawn = False; landmarks_norm = None
    if MP_OK:
        try:
            with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, model_complexity=1) as hnd:
                res = hnd.process(rgb)
                if res.multi_hand_landmarks and len(res.multi_hand_landmarks):
                    lm = [{'x': p.x, 'y': p.y, 'z': p.z} for p in res.multi_hand_landmarks[0].landmark]
                    frame = build_frame(lm)
                    landmarks_norm = res.multi_hand_landmarks[0]
                    try:
                        hd_list = res.multi_handedness[0].classification if res.multi_handedness else None
                        hd = hd_list[0] if hd_list else None
                        label = getattr(hd, 'category_name', None) or getattr(hd, 'display_name', None) or getattr(hd, 'label', None) if hd else '未知'
                        conf = round(float(hd.score) * 100) if hd else 0
                        # ---- 左右手判定：三重证据链（不依赖 MediaPipe 手性）----
                        # ① 几何法：掌心朝上时拇指朝身体外侧 → 图像上「拇指在左=左手、在右=右手」
                        # ② z 深度：掌凹朝上（中指根 z > 两侧 MCP 均值 z，差值>阈值）⇒ 掌心朝上
                        # ③ MediaPipe 官方规则：非镜像输入需 swap 其手性输出（仅作参考交叉验证）
                        src = []
                        if len(lm) >= 21:
                            thumb_x = lm[4]['x']; pinky_x = lm[20]['x']
                            geo = 'Left' if thumb_x < pinky_x else 'Right'
                            z9 = lm[9]['z']; zmcp = (lm[5]['z'] + lm[13]['z'] + lm[17]['z']) / 3.0
                            palm_up = (z9 - zmcp) > 0.015  # 掌凹朝上（掌心朝上）
                            final = geo if palm_up else ('Right' if geo == 'Left' else 'Left')
                            src.append(f'几何拇指位({geo})')
                            src.append('z验证掌心朝上' if palm_up else 'z判定疑似手背朝上!')
                            if label in ('Left', 'Right'):
                                swap = 'Left' if label == 'Right' else 'Right'
                                if swap == final:
                                    src.append('MediaPipe官方swap一致')
                                else:
                                    src.append(f'MediaPipe原始{label}未对齐(忽略)')
                            hand_info = {'label': final, 'confidence': conf, 'palmUp': bool(palm_up),
                                         'source': '·'.join(src)}
                        else:
                            hand_info = {'label': label, 'confidence': conf, 'source': 'MediaPipe'}
                    except Exception:
                        hand_info = {'label': '未知', 'confidence': 0, 'source': None}
                    skeleton_drawn = True
        except Exception as e:
            sys.stderr.write(f'[warn] detection failed: {e}\n')

    if frame is not None:
        regs = landmark_regions(frame, W, H)
    else:
        regs = naive_regions(W, H, skin)

    m_life = region_metric(regs['life'], skin, crease, W, H)
    m_mind = region_metric(regs['mind'], skin, crease, W, H)
    m_heart = region_metric(regs['heart'], skin, crease, W, H)
    m_fate = region_metric(regs['fate'], skin, crease, W, H)
    m_sun = region_metric(regs['sun'], skin, crease, W, H)

    prof = {}
    if frame:
        prof['life'] = line_profile(regs['life'], 'u', frame, skin, crease, W, H)
        prof['mind'] = line_profile(regs['mind'], 'v', frame, skin, crease, W, H)
        prof['heart'] = line_profile(regs['heart'], 'v', frame, skin, crease, W, H)
        prof['fate'] = line_profile(regs['fate'], 'u', frame, skin, crease, W, H)
        prof['sun'] = line_profile(regs['sun'], 'u', frame, skin, crease, W, H)
        # 描线轨迹：皱褶加权质心（追真实皱褶，非区域中线）
        _axis = {'life': 'u', 'mind': 'v', 'heart': 'v', 'fate': 'u', 'sun': 'u'}
        for _k in ('life', 'mind', 'heart', 'fate', 'sun'):
            if prof[_k].get('has'):
                prof[_k]['trace'] = trace_line_path(regs[_k], _axis[_k], frame, skin, crease, W, H)
            else:
                prof[_k]['trace'] = []

    def nov():
        return {'type': 'none', 'sev': 0, 'pos': None, 'run': 0, 'label': None, 'b0': None, 'b1': None}

    mark_life = classify_mark(prof.get('life'), 'life') if prof.get('life') else nov()
    mark_mind = classify_mark(prof.get('mind'), 'mind') if prof.get('mind') else nov()
    mark_heart = classify_mark(prof.get('heart'), 'heart') if prof.get('heart') else nov()
    mark_fate = classify_mark(prof.get('fate'), 'fate') if prof.get('fate') else nov()
    mark_sun = classify_mark(prof.get('sun'), 'sun') if prof.get('sun') else nov()

    topo = None
    if frame:
        topo = {
            'life': detect_topology(regs['life'], skin, vis_mask, W, H),
            'mind': detect_topology(regs['mind'], skin, vis_mask, W, H),
            'heart': detect_topology(regs['heart'], skin, vis_mask, W, H),
            'fate': detect_topology(regs['fate'], skin, vis_mask, W, H),
            'sun': detect_topology(regs['sun'], skin, vis_mask, W, H),
        }
    life_tl = life_timeline(prof['life']) if prof.get('life') else None
    heart_trend = heart_end_trend(frame, regs['heart'], skin, crease, W, H) if frame else None
    wealth_marks = detect_wealth_marks(frame, skin, vis_mask, crease, W, H) if frame else None
    career_scan = career_spine(frame, skin, crease, W, H) if frame else None
    # 进阶纹向检测（川字掌 / 双太阳线 / 感情线羽毛纹）
    chuan = chuan_analysis(frame, skin, crease, regs, W, H) if frame else None
    sun_double = sun_double_line(frame, skin, crease, regs, W, H) if frame else None
    heart_feather = heart_feathering(frame, skin, crease, regs, W, H) if frame else None

    cards_def = [
        {'name': '生命线', 'key': 'life', 'm': m_life, 'mark': mark_life,
         'topo': topo['life'] if topo else None},
        {'name': '智慧线', 'key': 'mind', 'm': m_mind, 'mark': mark_mind,
         'topo': topo['mind'] if topo else None},
        {'name': '感情线', 'key': 'heart', 'm': m_heart, 'mark': mark_heart,
         'topo': topo['heart'] if topo else None},
        {'name': '事业线', 'key': 'fate', 'm': m_fate, 'mark': mark_fate,
         'topo': topo['fate'] if topo else None},
        {'name': '太阳线', 'key': 'sun', 'm': m_sun, 'mark': mark_sun,
         'topo': topo['sun'] if topo else None},
    ]
    avg_s = (m_life['strength'] + m_mind['strength'] + m_heart['strength'] + m_fate['strength']) / 4
    avg_d = (m_life['density'] + m_mind['density'] + m_heart['density'] + m_fate['density']) / 4
    report_cards = []; total = 0
    for c in cards_def:
        len_r = prof[c['key']]['lengthRatio'] if (prof.get(c['key']) and prof[c['key']].get('has')) else 0.5
        grad = depth_gradient(prof.get(c['key']), c['key'], frame) if prof.get(c['key']) else None
        conf = line_confidence(prof.get(c['key']), frame is not None, grad)
        interp = interpret(c['name'], c['m'], c['mark'], len_r, frame is not None, avg_s, avg_d)
        clarity_shown = interp['clarityLabel']
        base = 80 if interp['clarityLabel'] == '深长清晰' else (68 if interp['clarityLabel'] == '中等' else 56)
        sev = 14 if interp['markType'] == 'break' else (8 if interp['markType'] == 'island' else (4 if interp['markType'] == 'chain' else 0))
        score = max(40, min(98, base - sev))
        if c['key'] == 'fate' and career_scan:
            clarity_shown = '纵脊可见' if career_scan['visible'] else '微弱'
            if career_scan['visible'] and interp['clarityLabel'] == '浅淡':
                score = min(98, score + 12)
        if c['key'] in ('life', 'mind', 'heart', 'fate'):
            total += score
        report_cards.append({
            'name': c['name'], 'clarity': clarity_shown,
            'length': interp['lengthLabel'] or '-', 'mark': interp['markType'],
            'markPos': c['mark']['label'] if (c['mark'] and c['mark']['label']) else None,
            'topoNodes': c['topo']['nodes'] if c['topo'] else 0,
            'strength': round(c['m']['strength'], 1), 'density': round(c['m']['density'], 4),
            'confidence': conf['lvl'], 'confNote': conf['note'],
            'depthGradient': ({'type': grad['type'], 'desc': grad['desc'], 'startQ': grad['startQ'],
                               'endQ': grad['endQ'], 'reliable': grad.get('reliable', True)} if grad else None),
            'path': (prof.get(c['key']) or {}).get('trace') or [],
        })

    total = round(total / 4)
    if total >= 85:
        q_label = '优质 · 纹路清晰'
    elif total >= 70:
        q_label = '良好'
    elif total >= 55:
        q_label = '一般'
    else:
        q_label = '偏低 · 建议重拍'

    skin_ratio = skin.sum() / (W * H)
    crease_ratio = (crease > 0).sum() / max(1, skin.sum())
    vis_ratio = (vis_mask > 0).sum() / max(1, skin.sum())
    low_conf = skin_ratio < 0.08 or crease_ratio < 0.012
    if frame:
        region_mode = 'AI关键点精准分区'
    elif not MP_OK:
        region_mode = '简易分区(降级·缺mediapipe)'
    else:
        region_mode = '简易分区(降级)'
    mark_list = []
    for nm, mk in [('生命线', mark_life), ('智慧线', mark_mind), ('感情线', mark_heart), ('事业线', mark_fate)]:
        if mk['type'] in ('chain', 'island', 'break'):
            mark_list.append(f"{nm}·{MARKCN[mk['type']]}" + (f"({mk['label']})" if mk['label'] else ''))
    mark_note = ('标记检测：' + ('、'.join(mark_list) + '（按纹理断裂模式估算）' if mark_list else '各主线连续、无明显标记')) if frame else '标记检测需AI关键点'
    topo_sum = 0
    if topo:
        for k in topo:
            topo_sum += topo[k]['nodes']
    topo_note = ('拓扑特征检测：' + (f'检测到 {topo_sum} 处分叉/交汇节点' if topo_sum else '各主线拓扑平滑、未见明显分叉/交汇节点')) if frame else '拓扑特征检测需AI关键点'
    grad_note = '起止深浅对比需AI关键点'
    if frame:
        gl = []
        for ln in report_cards:
            if ln['depthGradient'] and ln['depthGradient']['type'] != 'uniform':
                gl.append(f"{ln['name']}·{ln['depthGradient']['desc']}")
        grad_note = '起止深浅对比：' + ('、'.join(gl) if gl else '四条主线全程深浅均匀')
    wealth_note = ('财富图案征兆：' + ('、'.join(f"{m['location']}·{m['type']}(置信{m['confidence']})" for m in (wealth_marks or [])[:6]) if wealth_marks else '未检出明显财富吉纹')) if frame else '财富图案征兆检测需AI关键点'
    if hand_info:
        hand_note = f"检测到手，判定为 {hand_info['label']}手（置信 {hand_info['confidence']}%，依据：{hand_info.get('source') or 'MediaPipe'}）"
    elif not MP_OK:
        hand_note = '未检出手部：运行环境缺 mediapipe，关键点检测不可用，已降级——并非照片问题，请 pip install -r requirements.txt'
    else:
        hand_note = '未定位到手部（已尝试 AI 关键点检测但未命中，可能角度/光线/手指并拢导致）'

    palm_roi = build_palm_roi(frame, W, H) if frame else None
    annotated = draw_annotations(bgr, cards_def, prof, topo, life_tl, wealth_marks, palm_roi, W, H,
                                  (frame['landmarks'] if frame else None), crease=crease, vis_mask=vis_mask,
                                  extras={'chuan': chuan, 'sunDouble': sun_double, 'heartFeather': heart_feather})

    report = {
        'hand': hand_info,
        'lines': report_cards,
        'lifeTimeline': ({'segments': [{'age': f"{s['lo']}-{s['hi']}", 'state': s['st']} for s in life_tl['segs']],
                          'breaks': [{'from': b['age0'], 'to': b['age1'], 'bins': [b['b0'], b['b1']]} for b in life_tl['breaks']]} if life_tl else None),
        'heartEndTrend': heart_trend['type'] if heart_trend else None,
        'wealthMarks': wealth_marks,
        'careerSpine': career_scan,
        'extraFeatures': {
            'chuan': chuan,
            'sunDouble': sun_double,
            'heartFeather': heart_feather,
        },
        'qualityScore': {'score': total, 'label': q_label,
                         'note': '仅衡量照片掌纹提取质量，分数低=图糊/光影差，建议重拍，绝不代表命运'},
        'diagnostics': {
            'skinRatio': round(float(skin_ratio), 4), 'creaseRatio': round(float(crease_ratio), 4),
            'visRatio': round(float(vis_ratio), 4), 'lowConf': bool(low_conf),
            'regionMode': region_mode, 'markNote': mark_note, 'topoNote': topo_note,
            'gradNote': grad_note, 'wealthNote': wealth_note, 'handNote': hand_note,
            'skeletonDrawn': skeleton_drawn,
        },
        'imageInfo': {'width': pw, 'height': ph, 'resized': [W, H], 'localProcessing': True},
        'landmarks': (to_native(frame['landmarks']) if frame else None),
    }
    return report, annotated


def to_native(obj):
    """递归把 numpy 类型转成 Python 原生 JSON 可序列化类型。"""
    if isinstance(obj, np.ndarray):
        return [to_native(x) for x in obj.tolist()]
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    return obj


def main():
    ap = argparse.ArgumentParser(description='掌纹结构化提取（Python CLI）')
    ap.add_argument('image', help='手掌照片路径')
    ap.add_argument('-o', '--out', default='.', help='输出目录（默认当前目录）')
    args = ap.parse_args()
    if not MP_OK:
        _try_install_mediapipe()
    report, annotated = analyze(args.image)
    os.makedirs(args.out, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.image))[0]
    json_path = os.path.join(args.out, 'palm.json')
    png_path = os.path.join(args.out, 'annotated.png')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(to_native(report), f, ensure_ascii=False, indent=2)
    cv2.imwrite(png_path, annotated)
    # 简要摘要（便于 agent 快速确认）
    print(f'[OK] json={json_path} png={png_path}')
    print(f"[质量分] {report['qualityScore']['score']} ({report['qualityScore']['label']})")
    print(f"[手部] {report['diagnostics']['handNote']}")
    if not MP_OK:
        print('[降级] 因环境缺 mediapipe，已跳过手部关键点(21)检测；上述线条清晰度仅为区域估算，左右手/流年/拓扑/财富图案均不可用，结论置信低，非照片问题。')
    for ln in report['lines']:
        print(f"  - {ln['name']}: 清晰度={ln['clarity']} 标记={ln['mark']}{(' @'+ln['markPos']) if ln['markPos'] else ''} 置信={ln['confidence']}")
    ef = report.get('extraFeatures', {})
    if ef.get('chuan'):
        c = ef['chuan']
        print(f"[川字掌] isChuan={c['isChuan']} mind-heart间距={c['mindHeartGap']} life-mind间距={c['lifeMindGap']} 置信={c['confidence']}")
    if ef.get('sunDouble'):
        s = ef['sunDouble']
        print(f"[双太阳线] isDouble={s['isDouble']} 主脊数={s['count']} 最长主脊={s.get('bestLen')} 置信={s['confidence']}")
    if ef.get('heartFeather'):
        h = ef['heartFeather']
        print(f"[感情线羽毛纹] has={h['has']} 分叉点={h['forks']} 末端数={h['tips']} 置信={h['confidence']}")


if __name__ == '__main__':
    main()
