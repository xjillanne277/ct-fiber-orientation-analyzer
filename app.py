import streamlit as st
import cv2
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os

st.set_page_config(
    page_title="CT Scan Analyzer", 
    page_icon=":material/layers:", 
    layout="wide"
)

# Base directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CT_IMAGES_DIR = os.path.join(BASE_DIR, "CT Scan Images")

PRESET_DICT = {
    "XZ Plane: Cross-Section (Skin-Core-Skin)": os.path.join(CT_IMAGES_DIR, "black, 90deg, xz plane with core.png"),
    "XZ Plane: Cross-Section #2": os.path.join(CT_IMAGES_DIR, "black, 90deg, xz plane with core2.png"),
    "XY Plane: Surface Slice (Only Skin)": os.path.join(CT_IMAGES_DIR, "black, 90deg, xy plane, only skin.png"),
    "XY Plane: Surface Slice #2": os.path.join(CT_IMAGES_DIR, "black, 90deg, xy plane, only skin2.png"),
    "0° Cut Reference Scan": os.path.join(BASE_DIR, "sample_ct_0deg.png"),
    "45° Cut Reference Scan": os.path.join(BASE_DIR, "sample_ct_45deg.png"),
    "90° Cut Reference Scan": os.path.join(BASE_DIR, "sample_ct_90deg.png")
}

# ---------------------------------------------------------
# SIDEBAR CONTROLS (Scale & Thickness Calibration)
# ---------------------------------------------------------
with st.sidebar:
    st.header(":material/straighten: Scale & Thickness Calibration")
    scale_mode = st.radio(
        "Calibration Method:",
        ["Sample Thickness (mm)", "CT Pixel Size (µm/px)", "Field of View Width (mm)"],
        index=0,
        help="Choose how pixel measurements are converted to physical millimeters"
    )
    
    if scale_mode == "Sample Thickness (mm)":
        user_thickness_mm = st.number_input(
            "Known Sample Thickness (mm)",
            min_value=0.1,
            max_value=50.0,
            value=2.0,
            step=0.1,
            help="Physical thickness of the specimen measured with calipers/micrometer"
        )
        user_pixel_size_um = None
        user_fov_width_mm = None
    elif scale_mode == "CT Pixel Size (µm/px)":
        user_pixel_size_um = st.number_input(
            "CT Voxel / Pixel Size (µm/px)",
            min_value=0.01,
            max_value=500.0,
            value=5.21,
            step=0.1,
            help="Micro-CT scan voxel resolution (e.g. 5.21 µm/pixel)"
        )
        user_thickness_mm = None
        user_fov_width_mm = None
    else:
        user_fov_width_mm = st.number_input(
            "CT Field of View Width (mm)",
            min_value=0.5,
            max_value=500.0,
            value=19.13,
            step=0.5,
            help="Total physical width from CT scan scale bar (e.g. 19.13 mm / 1.913e+04 µm)"
        )
        user_thickness_mm = None
        user_pixel_size_um = None


# ---------------------------------------------------------
# CORE IMAGE ANALYSIS PIPELINE (With 4-Way Trimming)
# ---------------------------------------------------------
def detect_sample_boundaries(img_gray):
    from scipy.ndimage import gaussian_filter1d
    h, w = img_gray.shape
    
    # Smooth vertical profile to get approximate core location
    row_means_all = np.mean(img_gray, axis=1)
    smooth_all = gaussian_filter1d(row_means_all, sigma=2.0)
    c_min = int(0.30 * h)
    c_max = int(0.70 * h)
    core_peak = c_min + int(np.argmax(smooth_all[c_min:c_max]))
    
    # 1. Detect Left & Right Sample Boundaries (X-direction)
    # Analyze the central vertical region where the specimen is guaranteed to exist
    sample_y_band = img_gray[max(0, core_peak - 100):min(h, core_peak + 100), :]
    col_means = np.mean(sample_y_band, axis=0)
    
    # Search from x=0 to w//2: specimen begins where column mean is consistently > 62 for 15+ cols
    x_part_left = 0
    for x in range(w // 2):
        if x + 15 < w and np.all(col_means[x:x+15] > 62.0):
            x_part_left = x
            break
            
    # Search from x=w-1 down to w//2: specimen ends where column mean drops
    x_part_right = w - 1
    for x in range(w - 1, w // 2, -1):
        if x - 15 >= 0 and np.all(col_means[x-15:x] > 62.0):
            x_part_right = x
            break
            
    # 2. Slice strictly within the physical sample width [x_part_left:x_part_right + 1]
    sample_x_img = img_gray[:, x_part_left:x_part_right + 1]
    row_means = np.mean(sample_x_img, axis=1)
    smooth_means = gaussian_filter1d(row_means, sigma=2.0)
    d_means = np.gradient(smooth_means)
    
    # 3. Locate Core Peak in the center region
    core_peak = c_min + int(np.argmax(smooth_means[c_min:c_max]))
    core_val = smooth_means[core_peak]
    
    skin_top_sample = smooth_means[max(0, core_peak - 100)]
    skin_bot_sample = smooth_means[min(h - 1, core_peak + 100)]
    skin_baseline = (skin_top_sample + skin_bot_sample) / 2.0
    
    # Core thickness: thin lighter region centered at core_peak
    core_thresh = skin_baseline + 0.35 * (core_val - skin_baseline)
    y_core_top = core_peak
    while y_core_top > 10 and smooth_means[y_core_top] > core_thresh:
        y_core_top -= 1
    y_core_bot = core_peak
    while y_core_bot < h - 10 and smooth_means[y_core_bot] > core_thresh:
        y_core_bot += 1
        
    if y_core_bot - y_core_top < 8:
        y_core_top = max(0, core_peak - 8)
        y_core_bot = min(h - 1, core_peak + 8)
        
    # 4. Detect Top Surface (moving upward from core until sharp transition into dark background)
    y_part_top = 0
    for y in range(core_peak - 25, 0, -1):
        if (d_means[y] > 1.5 and smooth_means[y] < 72.0) or (smooth_means[y] < 63.0):
            y_part_top = y
            break
            
    # 5. Detect Bottom Surface (moving downward from core until sharp transition into dark background)
    y_part_bot = h - 1
    for y in range(core_peak + 25, h - 1):
        if (d_means[y] < -1.5 and smooth_means[y] < 72.0) or (smooth_means[y] < 63.0):
            y_part_bot = y
            break
            
    y_part_top = max(0, min(y_part_top, h - 4))
    y_core_top = max(y_part_top + 1, min(y_core_top, h - 3))
    y_core_bot = max(y_core_top + 1, min(y_core_bot, h - 2))
    y_part_bot = max(y_core_bot + 1, min(y_part_bot, h - 1))
    
    return x_part_left, x_part_right, y_part_top, y_core_top, y_core_bot, y_part_bot


def analyze_sample(img_gray, x_part_left, x_part_right, y_part_top, y_core_top, y_core_bot, y_part_bot, 
                   scale_mode="Sample Thickness (mm)", user_thickness_mm=2.0, user_pixel_size_um=5.21, user_fov_width_mm=19.13,
                   blur_ksize=21):
    h, w = img_gray.shape
    part_h_px = max(y_part_bot - y_part_top + 1, 1)
    sample_w_px = max(x_part_right - x_part_left + 1, 1)
    
    # 1. Compute Physical Scale (mm per pixel and um per pixel)
    if scale_mode == "Sample Thickness (mm)":
        actual_thickness_mm = float(user_thickness_mm)
        mm_per_px = actual_thickness_mm / part_h_px
        um_per_px = mm_per_px * 1000.0
    elif scale_mode == "CT Pixel Size (µm/px)":
        um_per_px = float(user_pixel_size_um)
        mm_per_px = um_per_px / 1000.0
        actual_thickness_mm = part_h_px * mm_per_px
    else: # Field of View Width
        mm_per_px = float(user_fov_width_mm) / float(w)
        um_per_px = mm_per_px * 1000.0
        actual_thickness_mm = part_h_px * mm_per_px
        
    # Physical layer dimensions
    top_nonpart_px = y_part_top
    top_skin_px = y_core_top - y_part_top
    core_px = y_core_bot - y_core_top
    bot_skin_px = y_part_bot - y_core_bot
    bot_nonpart_px = h - 1 - y_part_bot
    total_skin_px = top_skin_px + bot_skin_px
    
    top_nonpart_mm = top_nonpart_px * mm_per_px
    top_skin_mm = top_skin_px * mm_per_px
    core_mm = core_px * mm_per_px
    bot_skin_mm = bot_skin_px * mm_per_px
    bot_nonpart_mm = bot_nonpart_px * mm_per_px
    total_skin_mm = total_skin_px * mm_per_px
    
    top_skin_pct = (top_skin_px / part_h_px) * 100.0
    core_pct = (core_px / part_h_px) * 100.0
    bot_skin_pct = (bot_skin_px / part_h_px) * 100.0
    total_skin_pct = (total_skin_px / part_h_px) * 100.0
    
    # 2. Structure tensor for fiber orientation mapping
    Ix = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    
    Jxx = cv2.GaussianBlur(Ix**2, (blur_ksize, blur_ksize), 0)
    Jyy = cv2.GaussianBlur(Iy**2, (blur_ksize, blur_ksize), 0)
    Jxy = cv2.GaussianBlur(Ix * Iy, (blur_ksize, blur_ksize), 0)
    
    denom = Jxx + Jyy + 1e-8
    a11_map = Jyy / denom
    a22_map = Jxx / denom
    
    theta_rad = 0.5 * np.arctan2(2 * Jxy, Jyy - Jxx) + (np.pi / 2.0)
    theta_deg = (np.degrees(theta_rad)) % 180.0
    
    # 3. Direct Fiber Volume Fraction (Vf) Estimation (Sample region only)
    sample_roi = img_gray[y_part_top:y_part_bot + 1, x_part_left:x_part_right + 1]
    fiber_mask = cv2.adaptiveThreshold(sample_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 0)
    global_fiber_pct = float(np.mean(fiber_mask > 0) * 100.0)
    
    # Full image fiber mask for visualization
    full_fiber_mask = np.zeros_like(img_gray, dtype=np.uint8)
    full_fiber_mask[y_part_top:y_part_bot + 1, x_part_left:x_part_right + 1] = fiber_mask
    
    # Global orientation
    global_a11 = float(np.mean(a11_map[y_part_top:y_part_bot + 1, x_part_left:x_part_right + 1]))
    global_a22 = 1.0 - global_a11
    
    # 4. Through-thickness profile dataframe
    num_slices = 50
    step = part_h_px / num_slices
    profile_rows = []
    
    for i in range(num_slices):
        ys = y_part_top + int(round(i * step))
        ye = y_part_top + int(round((i + 1) * step))
        if ye <= ys:
            ye = ys + 1
        ye = min(ye, h)
        
        slice_fib = full_fiber_mask[ys:ye, x_part_left:x_part_right + 1]
        slice_a11 = a11_map[ys:ye, x_part_left:x_part_right + 1]
        slice_a22 = a22_map[ys:ye, x_part_left:x_part_right + 1]
        
        f_pct = float(np.mean(slice_fib > 0) * 100.0)
        a11_v = float(np.mean(slice_a11))
        a22_v = float(np.mean(slice_a22))
        
        y_mid = (ys + ye) / 2.0
        if y_mid < y_core_top:
            zone_name = "Top Skin Layer"
        elif y_mid < y_core_bot:
            zone_name = "Core Layer (Center)"
        else:
            zone_name = "Bottom Skin Layer"
            
        z_norm = 1.0 - 2.0 * (i + 0.5) / num_slices
        depth_mm = (i + 0.5) * (actual_thickness_mm / num_slices)
        
        profile_rows.append({
            'Sub_Slice': i + 1,
            'Y_Start': ys,
            'Y_End': ye,
            'Normalized_Z': z_norm,
            'Depth_mm': depth_mm,
            'A11': np.clip(a11_v, 0.0, 1.0),
            'A22': np.clip(a22_v, 0.0, 1.0),
            'Fiber_Pct': f_pct,
            'Zone': zone_name
        })
        
    df_profile = pd.DataFrame(profile_rows)
    
    return {
        'h': h,
        'w': w,
        'x_part_left': x_part_left,
        'x_part_right': x_part_right,
        'sample_w_px': sample_w_px,
        'y_part_top': y_part_top,
        'y_core_top': y_core_top,
        'y_core_bot': y_core_bot,
        'y_part_bot': y_part_bot,
        'part_h_px': part_h_px,
        'actual_thickness_mm': actual_thickness_mm,
        'mm_per_px': mm_per_px,
        'um_per_px': um_per_px,
        'top_nonpart_px': top_nonpart_px,
        'top_skin_px': top_skin_px,
        'core_px': core_px,
        'bot_skin_px': bot_skin_px,
        'bot_nonpart_px': bot_nonpart_px,
        'total_skin_px': total_skin_px,
        'top_nonpart_mm': top_nonpart_mm,
        'top_skin_mm': top_skin_mm,
        'core_mm': core_mm,
        'bot_skin_mm': bot_skin_mm,
        'bot_nonpart_mm': bot_nonpart_mm,
        'total_skin_mm': total_skin_mm,
        'top_skin_pct': top_skin_pct,
        'core_pct': core_pct,
        'bot_skin_pct': bot_skin_pct,
        'total_skin_pct': total_skin_pct,
        'global_fiber_pct': global_fiber_pct,
        'global_a11': global_a11,
        'global_a22': global_a22,
        'full_fiber_mask': full_fiber_mask,
        'a11_map': a11_map,
        'a22_map': a22_map,
        'theta_deg': theta_deg,
        'df_profile': df_profile,
        'image_gray': img_gray
    }


def draw_segmentation_overlay(img_gray, res, alpha=0.35):
    h, w = img_gray.shape
    overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    color_mask = overlay.copy()
    
    xl = res['x_part_left']
    xr = res['x_part_right']
    y_pt = res['y_part_top']
    y_ct = res['y_core_top']
    y_cb = res['y_core_bot']
    y_pb = res['y_part_bot']
    
    # 1. Shading for 5 zones
    # Top & bottom non-part
    if y_pt > 0:
        color_mask[0:y_pt, :] = [30, 30, 30]
    if y_pb < h - 1:
        color_mask[y_pb:h, :] = [30, 30, 30]
        
    # Left & right non-part trims
    if xl > 0:
        color_mask[:, 0:xl] = [20, 20, 20]
    if xr < w - 1:
        color_mask[:, xr:w] = [20, 20, 20]
        
    # 5-zone color shading on physical sample
    if y_ct > y_pt:
        color_mask[y_pt:y_ct, xl:xr+1] = [0, 200, 80]
    if y_pb > y_cb:
        color_mask[y_cb:y_pb, xl:xr+1] = [0, 200, 80]
    if y_cb > y_ct:
        color_mask[y_ct:y_cb, xl:xr+1] = [0, 180, 255]
        
    cv2.addWeighted(color_mask, alpha, overlay, 1 - alpha, 0, overlay)
    
    # Draw crisp boundary lines across the CT scan image
    line_w = max(2, int(h / 450))
    if y_pt > 0:
        cv2.line(overlay, (xl, y_pt), (w, y_pt), (220, 220, 220), line_w)
    if y_ct > y_pt:
        cv2.line(overlay, (xl, y_ct), (w, y_ct), (0, 255, 255), line_w)
    if y_cb < y_pb:
        cv2.line(overlay, (xl, y_cb), (w, y_cb), (0, 255, 255), line_w)
    if y_pb < h - 1:
        cv2.line(overlay, (xl, y_pb), (w, y_pb), (220, 220, 220), line_w)
        
    if xl > 0:
        cv2.line(overlay, (xl, 0), (xl, h), (160, 160, 160), line_w)
    if xr < w - 1:
        cv2.line(overlay, (xr, 0), (xr, h), (160, 160, 160), line_w)
        
    # Create clean canvas with dedicated right-hand annotation panel (Zero text overlapping CT scan)
    gutter_w = max(560, int(w * 0.18))
    canvas = np.full((h, w + gutter_w, 3), (18, 20, 24), dtype=np.uint8)
    canvas[:, 0:w] = overlay
    cv2.line(canvas, (w, 0), (w, h), (60, 65, 75), 2)
    
    # Scale font size dynamically based on image height
    f_scale = max(0.85, min(1.2, h / 750.0))
    font = cv2.FONT_HERSHEY_SIMPLEX
    thick = 2
    
    # 1. Total Sample Thickness Header Card at top of annotation panel
    card_h = max(75, int(h * 0.13))
    cv2.rectangle(canvas, (w + 15, 15), (w + gutter_w - 15, card_h), (28, 32, 38), -1)
    cv2.rectangle(canvas, (w + 15, 15), (w + gutter_w - 15, card_h), (0, 200, 240), 2)
    cv2.putText(canvas, "TOTAL SAMPLE THICKNESS", (w + 30, int(card_h * 0.42)), font, f_scale * 0.75, (160, 175, 190), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"{res['actual_thickness_mm']:.2f} mm  ({res['part_h_px']} px)", (w + 30, int(card_h * 0.85)), font, f_scale * 1.15, (0, 225, 255), 2, cv2.LINE_AA)
    
    # 2. Boundary Pointer Lines & Clear Labels in the side panel
    ptr_len = 35
    for y_pos, label, color in [
        (y_pt, f"Top Surface (y={y_pt}px)", (240, 240, 240)),
        (y_ct, f"Top Skin / Core (y={y_ct}px)", (0, 240, 255)),
        (y_cb, f"Core / Bot Skin (y={y_cb}px)", (0, 240, 255)),
        (y_pb, f"Bottom Surface (y={y_pb}px)", (240, 240, 240))
    ]:
        if 0 <= y_pos < h:
            cv2.line(canvas, (w, y_pos), (w + ptr_len, y_pos), color, 2)
            cv2.circle(canvas, (w + ptr_len, y_pos), 4, color, -1)
            cv2.putText(canvas, label, (w + ptr_len + 15, y_pos + int(6 * f_scale)), font, f_scale * 0.85, color, thick, cv2.LINE_AA)
            
    # 3. Layer Measurement Callout Cards in the side panel
    y_mid_top = (y_pt + y_ct) // 2
    if y_mid_top > card_h + 30 and y_mid_top < y_ct - 25:
        cv2.rectangle(canvas, (w + ptr_len + 15, y_mid_top - 22), (w + gutter_w - 20, y_mid_top + 22), (20, 38, 28), -1)
        cv2.rectangle(canvas, (w + ptr_len + 15, y_mid_top - 22), (w + gutter_w - 20, y_mid_top + 22), (0, 180, 80), 1)
        cv2.putText(canvas, f"Top Skin: {res['top_skin_mm']:.2f} mm ({res['top_skin_px']} px)", (w + ptr_len + 25, y_mid_top + 6), font, f_scale * 0.88, (80, 240, 140), 2, cv2.LINE_AA)
        
    y_mid_bot = (y_cb + y_pb) // 2
    if y_mid_bot > y_cb + 25 and y_mid_bot < h - 25:
        cv2.rectangle(canvas, (w + ptr_len + 15, y_mid_bot - 22), (w + gutter_w - 20, y_mid_bot + 22), (20, 38, 28), -1)
        cv2.rectangle(canvas, (w + ptr_len + 15, y_mid_bot - 22), (w + gutter_w - 20, y_mid_bot + 22), (0, 180, 80), 1)
        cv2.putText(canvas, f"Bot Skin: {res['bot_skin_mm']:.2f} mm ({res['bot_skin_px']} px)", (w + ptr_len + 25, y_mid_bot + 6), font, f_scale * 0.88, (80, 240, 140), 2, cv2.LINE_AA)
        
    return canvas


# ---------------------------------------------------------
# MAIN UPLOAD SECTION (Clean, Minimal, Single Upload Box)
# ---------------------------------------------------------
st.title("CT Scan Analyzer")

uploaded_file = st.file_uploader(
    "Upload CT scan", 
    type=["png", "jpg", "jpeg", "tif", "tiff"]
)

# Optional demo dropdown inside a subtle expander
with st.expander("Or select from demo CT scans", expanded=False):
    demo_selection = st.selectbox(
        "Choose demo scan:", 
        ["(None)"] + list(PRESET_DICT.keys()), 
        index=0
    )

selected_image_gray = None
source_title = ""

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    selected_image_gray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    source_title = uploaded_file.name
elif demo_selection != "(None)":
    preset_path = PRESET_DICT[demo_selection]
    if os.path.exists(preset_path):
        img_bgr = cv2.imread(preset_path)
        selected_image_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        source_title = demo_selection

# ---------------------------------------------------------
# RUN ANALYSIS & DISPLAY ONLY AFTER IMAGE IS UPLOADED
# ---------------------------------------------------------
if selected_image_gray is not None:
    h, w = selected_image_gray.shape
    
    # Auto-detect boundaries (X and Y)
    auto_xl, auto_xr, auto_pt, auto_ct, auto_cb, auto_pb = detect_sample_boundaries(selected_image_gray)
    
    # Optional fine-tuning expander
    with st.expander(":material/tune: Fine-Tune Boundaries (Optional)", expanded=False):
        t_c1, t_c2 = st.columns(2)
        with t_c1:
            sel_xl = st.number_input("Left Trim (x px)", min_value=0, max_value=w//2, value=auto_xl, step=2)
        with t_c2:
            sel_xr = st.number_input("Right Trim (x px)", min_value=w//2, max_value=w-1, value=auto_xr, step=2)
            
        b_c1, b_c2, b_c3, b_c4 = st.columns(4)
        with b_c1:
            sel_pt = st.number_input("Top Surface (y px)", min_value=0, max_value=h-4, value=auto_pt, step=2)
        with b_c2:
            sel_ct = st.number_input("Top Skin / Core (y px)", min_value=sel_pt+1, max_value=h-3, value=max(sel_pt+1, auto_ct), step=2)
        with b_c3:
            sel_cb = st.number_input("Core / Bot Skin (y px)", min_value=sel_ct+1, max_value=h-2, value=max(sel_ct+1, auto_cb), step=2)
        with b_c4:
            sel_pb = st.number_input("Bottom Surface (y px)", min_value=sel_cb+1, max_value=h-1, value=max(sel_cb+1, auto_pb), step=2)
            
    res = analyze_sample(
        selected_image_gray, 
        sel_xl,
        sel_xr,
        sel_pt, 
        sel_ct, 
        sel_cb, 
        sel_pb, 
        scale_mode=scale_mode,
        user_thickness_mm=user_thickness_mm if user_thickness_mm is not None else 2.0,
        user_pixel_size_um=user_pixel_size_um if user_pixel_size_um is not None else 5.21,
        user_fov_width_mm=user_fov_width_mm if user_fov_width_mm is not None else 19.13,
        blur_ksize=21
    )
    
    # Top KPI Metric Cards
    kpi_c1, kpi_c2, kpi_c3, kpi_c4, kpi_c5 = st.columns(5)
    with kpi_c1:
        st.metric(
            "Total Sample Thickness", 
            f"{res['actual_thickness_mm']:.2f} mm", 
            f"{res['part_h_px']} px ({res['um_per_px']:.2f} µm/px)",
            help="Total through-thickness of the physical sample measured in vertical direction",
            border=True
        )
    with kpi_c2:
        st.metric(
            "Fiber Vol Fraction (Vf)", 
            f"{res['global_fiber_pct']:.1f}%", 
            help="Estimated directly from imagery (fibers = white lines & dots, matrix = background; nominal 30-40% range)",
            border=True
        )
    with kpi_c3:
        st.metric(
            "Total Skin Thickness", 
            f"{res['total_skin_mm']:.2f} mm", 
            help=f"Top Skin: {res['top_skin_mm']:.2f}mm ({res['top_skin_px']} px) | Bot Skin: {res['bot_skin_mm']:.2f}mm ({res['bot_skin_px']} px)",
            border=True
        )
    with kpi_c4:
        st.metric(
            "Core Layer Thickness", 
            f"{res['core_mm']:.2f} mm", 
            help=f"Thin lighter section centered in sample ({res['core_px']} pixels)",
            border=True
        )
    with kpi_c5:
        st.metric(
            "Skin-to-Core Ratio", 
            f"{res['total_skin_pct']:.1f}% / {res['core_pct']:.1f}%", 
            f"{res['total_skin_px']}px Skin / {res['core_px']}px Core", 
            help="Volumetric ratio of outer skin layers to central core layer",
            border=True
        )
        
    st.write("")
    
    # ---------------------------------------------------------
    # MAIN TABS (In exact requested order; Background is the last tab)
    # ---------------------------------------------------------
    tab_layers, tab_fiber, tab_orientation, tab_guide = st.tabs([
        ":material/straighten: Layer Thickness & Segmentation",
        ":material/percent: Fiber Volume Fraction (Vf)",
        ":material/show_chart: Orientation Profile (A11 vs A22)",
        ":material/menu_book: Background & Reference Guide"
    ])
    
    # TAB 1: LAYER THICKNESS & SEGMENTATION
    with tab_layers:
        st.subheader("Layer Thickness Measurement & Segmentation")
        st.caption(f"Specimen: `{source_title}` | Sample: {res['part_h_px']} px high × {res['sample_w_px']} px wide ({res['actual_thickness_mm']:.2f} mm thick | {res['um_per_px']:.2f} µm/px)")
        
        overlay_img = draw_segmentation_overlay(res['image_gray'], res)
        st.image(overlay_img, width="stretch", caption="5-Zone Physical Layer Overlay with Vertical Dimension Indicator (Green = Skin, Yellow/Amber = Core, Dark Gray = Non-Part)")
        
        t_col1, t_col2 = st.columns([3, 2])
        with t_col1:
            with st.container(border=True):
                st.markdown("**Layer Thickness Breakdown Table:**")
                table_data = [
                    {"Zone": "1. Top Non-Part (Smooth Dark Grey)", "Thickness (px)": res['top_nonpart_px'], "Thickness (mm)": f"{res['top_nonpart_mm']:.3f}", "% of Physical Part": "N/A (Background)"},
                    {"Zone": "2. Top Skin Layer (Dark Grainy)", "Thickness (px)": res['top_skin_px'], "Thickness (mm)": f"{res['top_skin_mm']:.3f}", "% of Physical Part": f"{res['top_skin_pct']:.1f}%"},
                    {"Zone": "3. Core Layer (Thin Lighter Center)", "Thickness (px)": res['core_px'], "Thickness (mm)": f"{res['core_mm']:.3f}", "% of Physical Part": f"{res['core_pct']:.1f}%"},
                    {"Zone": "4. Bottom Skin Layer (Dark Grainy)", "Thickness (px)": res['bot_skin_px'], "Thickness (mm)": f"{res['bot_skin_mm']:.3f}", "% of Physical Part": f"{res['bot_skin_pct']:.1f}%"},
                    {"Zone": "5. Bottom Non-Part (Smooth Dark Grey)", "Thickness (px)": res['bot_nonpart_px'], "Thickness (mm)": f"{res['bot_nonpart_mm']:.3f}", "% of Physical Part": "N/A (Background)"},
                    {"Zone": "TOTAL PHYSICAL SAMPLE", "Thickness (px)": res['part_h_px'], "Thickness (mm)": f"{res['actual_thickness_mm']:.3f}", "% of Physical Part": "100.0%"}
                ]
                st.dataframe(pd.DataFrame(table_data), width="stretch", hide_index=True)
        with t_col2:
            with st.container(border=True):
                fig_pie = px.pie(
                    values=[res['top_skin_pct'], res['core_pct'], res['bot_skin_pct']],
                    names=['Top Skin (Grainy)', 'Core (Light Center)', 'Bottom Skin (Grainy)'],
                    color=['Top Skin (Grainy)', 'Core (Light Center)', 'Bottom Skin (Grainy)'],
                    color_discrete_map={'Top Skin (Grainy)': '#10B981', 'Core (Light Center)': '#F59E0B', 'Bottom Skin (Grainy)': '#059669'},
                    hole=0.4,
                    title="Skin vs. Core Volume Ratio"
                )
                st.plotly_chart(fig_pie, width="stretch")

    # TAB 2: FIBER VOLUME FRACTION (Vf)
    with tab_fiber:
        st.subheader("Direct Fiber Volume Fraction ($V_f$) Estimation")
        st.markdown(
            f"Estimated Global Fiber Volume Fraction: **{res['global_fiber_pct']:.1f}%** "
            f"(Directly estimated from imagery: fibers = white lines & dots, resin matrix = dark background)."
        )
        
        # Fiber mask visualization
        fib_vis = cv2.cvtColor(res['image_gray'], cv2.COLOR_GRAY2BGR)
        fib_vis[res['full_fiber_mask'] > 0] = [0, 240, 255] # bright cyan/lime fibers
        
        f_c1, f_c2 = st.columns(2)
        with f_c1:
            with st.container(border=True):
                st.markdown("**Original CT Grayscale Scan**")
                st.image(res['image_gray'], width="stretch")
        with f_c2:
            with st.container(border=True):
                st.markdown("**Segmented Fiber Pixels (Lines & Dots Highlighted in Cyan)**")
                st.image(fib_vis, width="stretch")
                
        with st.container(border=True):
            fig_fib = px.line(
                res['df_profile'],
                x='Normalized_Z',
                y='Fiber_Pct',
                markers=True,
                title="Through-Thickness Fiber Density Profile (%)",
                labels={'Normalized_Z': 'Normalized Thickness z_norm', 'Fiber_Pct': 'Fiber Percentage (%)'}
            )
            fig_fib.update_traces(line_color='#3B82F6', line_width=2.5)
            st.plotly_chart(fig_fib, width="stretch")

    # TAB 3: ORIENTATION PROFILE
    with tab_orientation:
        st.subheader("Through-Thickness Fiber Orientation Tensor Profile")
        st.markdown("Quantifies flow-direction orientation ($A_{11}$) vs transverse orientation ($A_{22}$) from top mold skin to center core.")
        
        with st.container(border=True):
            fig_prof = go.Figure()
            fig_prof.add_trace(go.Scatter(
                x=res['df_profile']['Normalized_Z'],
                y=res['df_profile']['A11'],
                mode='lines+markers',
                name='A11 (Parallel to Flow Axis)',
                line=dict(color='#10B981', width=3),
                marker=dict(size=6)
            ))
            fig_prof.add_trace(go.Scatter(
                x=res['df_profile']['Normalized_Z'],
                y=res['df_profile']['A22'],
                mode='lines+markers',
                name='A22 (Perpendicular Transverse)',
                line=dict(color='#EF4444', width=2, dash='dot'),
                marker=dict(size=5)
            ))
            fig_prof.add_vline(x=0.0, line_dash="solid", line_color="gray", opacity=0.4, annotation_text="Center Core (z=0)")
            fig_prof.update_layout(
                title="Through-Thickness Fiber Orientation Tensor Profile ($A_{11}$ vs $A_{22}$)",
                xaxis_title="Normalized Thickness z_norm [ +1.0 = Top Mold Skin, 0.0 = Center Core, -1.0 = Bottom Mold Skin ]",
                yaxis_title="Orientation Tensor Component Value (0.0 to 1.0)",
                yaxis=dict(range=[0.0, 1.0]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_prof, width="stretch")

    # TAB 4: BACKGROUND & REFERENCE GUIDE (Last tab)
    with tab_guide:
        st.subheader("Microstructure & Cross-Section Reference Guide")
        st.info(
            "**FEA & Tensile Context:** In injection molded tensile bars, mold wall shear aligns fibers parallel to the flow (creating the skin layers), "
            "while slower center flow leaves fibers transverse (creating the core). Capturing this fiber variance is critical for tensile and FEA correlation."
        )
        st.markdown(
            "**Layer Definitions in CT Scan:**\n"
            "- **Top & Bottom Non-Part:** Smooth dark grey background (air/mounting void; 0mm if cropped to surface).\n"
            "- **Left & Right Non-Part:** Solid untextured borders (e.g. solid black air margin).\n"
            "- **Top & Bottom Skin:** Dark grainy section containing aligned fibers.\n"
            "- **Center Core:** Thin lighter section centered in the sample containing transverse fibers.\n"
            "- **Fiber Volume Fraction ($V_f$):** Directly extracted by segmenting high-density fiber lines and dots from the darker polymer matrix."
        )
