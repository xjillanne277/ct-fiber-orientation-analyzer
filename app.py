import streamlit as st
import cv2
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os

st.set_page_config(
    page_title="CT Fiber Orientation & Layer Thickness Analyzer", 
    page_icon=":material/layers:", 
    layout="wide"
)

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CT_IMAGES_DIR = os.path.join(BASE_DIR, "CT Scan Images")

PRESET_IMAGES = {
    "XZ Plane: Cross-Section with Skin-Core-Skin (black, 90deg, xz plane with core.png)": os.path.join(CT_IMAGES_DIR, "black, 90deg, xz plane with core.png"),
    "XZ Plane: Cross-Section #2 (black, 90deg, xz plane with core2.png)": os.path.join(CT_IMAGES_DIR, "black, 90deg, xz plane with core2.png"),
    "XY Plane: Surface Slice - Only Skin (black, 90deg, xy plane, only skin.png)": os.path.join(CT_IMAGES_DIR, "black, 90deg, xy plane, only skin.png"),
    "XY Plane: Surface Slice - Only Skin #2 (black, 90deg, xy plane, only skin2.png)": os.path.join(CT_IMAGES_DIR, "black, 90deg, xy plane, only skin2.png"),
    "0° Cut Reference Slice (sample_ct_0deg.png)": os.path.join(BASE_DIR, "sample_ct_0deg.png"),
    "45° Cut Reference Slice (sample_ct_45deg.png)": os.path.join(BASE_DIR, "sample_ct_45deg.png"),
    "90° Cut Reference Slice (sample_ct_90deg.png)": os.path.join(BASE_DIR, "sample_ct_90deg.png")
}

st.title(":material/layers: CT Scan Fiber Orientation & Layer Thickness Analyzer")
st.markdown(
    "Automated research tool enforcing the physical cross-section model: "
    "**`Non-Part (Smooth Dark Grey) ➔ Top Skin (Dark Grainy) ➔ Core (Thin Lighter Center) ➔ Bottom Skin (Dark Grainy) ➔ Non-Part (Smooth Dark Grey)`**."
)

# ---------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------
with st.sidebar:
    st.header(":material/settings: Physical Specimen Dimensions")
    specimen_thickness_mm = st.number_input(
        "Specimen Part Thickness (mm)", 
        min_value=0.1, 
        max_value=20.0, 
        value=2.0, 
        step=0.1, 
        help="Physical thickness of the molded part to scale pixel measurements into millimeters"
    )
    
    st.markdown("**Micromechanics Inputs (Halpin-Tsai / Rule of Mixtures):**")
    e_fiber_gpa = st.number_input("Fiber Modulus Ef (GPa)", min_value=1.0, max_value=500.0, value=72.0, step=1.0, help="E.g., E-glass fibers ~72 GPa")
    e_matrix_gpa = st.number_input("Matrix Modulus Em (GPa)", min_value=0.1, max_value=50.0, value=3.0, step=0.5, help="E.g., Polyamide/Polypropylene ~3.0 GPa")
    
    st.divider()
    st.header(":material/tune: Processing Parameters")
    blur_kernel_size = st.slider(
        "Structure Tensor Kernel Size", 
        min_value=3, 
        max_value=41, 
        value=21, 
        step=2, 
        help="Spatial smoothing window for fiber orientation gradient computation"
    )
    fiber_threshold = st.slider(
        "Fiber Intensity Threshold", 
        min_value=10, 
        max_value=200, 
        value=55, 
        help="Grayscale threshold to segment fiber pixels from resin matrix"
    )


# ---------------------------------------------------------
# AUTOMATIC SCAN RECOGNITION ALGORITHM
# Rule:
# 1. Non-Part (Top): Smooth dark grey (background/air/mount)
# 2. Top Skin: Dark grainy layer
# 3. Core: Thin lighter section centered in the sample
# 4. Bottom Skin: Dark grainy layer
# 5. Non-Part (Bottom): Smooth dark grey (background/air/mount)
# ---------------------------------------------------------
def auto_detect_5zone_boundaries(img_gray):
    h, w = img_gray.shape
    row_means = np.mean(img_gray, axis=1)
    row_stds = np.std(img_gray, axis=1)
    
    mid_y = h // 2
    
    # 1. Detect Top Sample Surface (where std and mean rise above smooth non-part background)
    y_part_top = 0
    for y in range(mid_y, 0, -1):
        if row_stds[y] < 9.0 or row_means[y] < 62.0:
            y_part_top = y + 1
            break
            
    # 2. Detect Bottom Sample Surface (where std and mean drop below smooth non-part background)
    y_part_bot = h - 1
    for y in range(mid_y, h):
        if row_stds[y] < 8.5 or row_means[y] < 58.0:
            y_part_bot = y - 1
            break
            
    part_h = max(y_part_bot - y_part_top + 1, 10)
    center_part = (y_part_top + y_part_bot) // 2
    
    # 3. Detect Thin Lighter Core Section Centered in the Sample
    # Search around the center of the part
    search_radius = max(int(0.20 * part_h), 10)
    s_min = max(0, center_part - search_radius)
    s_max = min(h, center_part + search_radius)
    
    core_search = row_means[s_min:s_max]
    if len(core_search) > 0:
        core_peak_y = s_min + int(np.argmax(core_search))
        peak_val = row_means[core_peak_y]
        
        # Skin baseline intensity
        skin_val_top = row_means[max(y_part_top, core_peak_y - search_radius)]
        skin_val_bot = row_means[min(y_part_bot, core_peak_y + search_radius)]
        baseline_skin = (skin_val_top + skin_val_bot) / 2.0
        
        core_thresh = baseline_skin + 0.35 * (peak_val - baseline_skin)
        
        y_core_top = core_peak_y
        while y_core_top > y_part_top and row_means[y_core_top] > core_thresh:
            y_core_top -= 1
            
        y_core_bot = core_peak_y
        while y_core_bot < y_part_bot and row_means[y_core_bot] > core_thresh:
            y_core_bot += 1
            
        # Ensure minimum core thickness if peak exists
        if y_core_bot - y_core_top < 6:
            y_core_top = max(y_part_top + 1, core_peak_y - 6)
            y_core_bot = min(y_part_bot - 1, core_peak_y + 6)
    else:
        y_core_top = center_part - 5
        y_core_bot = center_part + 5
        
    # Clamp bounds
    y_part_top = max(0, min(y_part_top, h - 4))
    y_core_top = max(y_part_top + 1, min(y_core_top, h - 3))
    y_core_end = max(y_core_top + 1, min(y_core_bot, h - 2))
    y_part_end = max(y_core_end + 1, min(y_part_bot, h - 1))
    
    return y_part_top, y_core_top, y_core_end, y_part_end


def compute_layer_metrics(img_gray, y_part_top, y_core_top, y_core_bot, y_part_bot, total_thick_mm=2.0, blur_ksize=21, fiber_thresh=55):
    h, w = img_gray.shape
    
    # Structure tensor computation for orientation analysis
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
    
    # Layer pixel counts
    top_nonpart_px = y_part_top
    top_skin_px = y_core_top - y_part_top
    core_px = y_core_bot - y_core_top
    bot_skin_px = y_part_bot - y_core_bot
    bot_nonpart_px = h - 1 - y_part_bot
    
    part_h_px = max(y_part_bot - y_part_top + 1, 1)
    total_skin_px = top_skin_px + bot_skin_px
    
    mm_per_px = total_thick_mm / part_h_px
    
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
    
    # Fiber % and Global Orientation on Physical Sample
    part_mask = (img_gray[y_part_top:y_part_bot+1, :] > fiber_thresh)
    global_fiber_pct = float(np.mean(part_mask) * 100.0)
    
    global_a11 = float(np.mean(a11_map[y_part_top:y_part_bot+1, :]))
    global_a22 = 1.0 - global_a11
    
    v_f = global_fiber_pct / 100.0
    e_predicted = e_matrix_gpa + global_a11 * v_f * (e_fiber_gpa - e_matrix_gpa)
    
    # Through-thickness profile dataframe
    num_slices = 50
    step = part_h_px / num_slices
    profile_rows = []
    
    for i in range(num_slices):
        ys = y_part_top + int(round(i * step))
        ye = y_part_top + int(round((i + 1) * step))
        if ye <= ys:
            ye = ys + 1
        ye = min(ye, h)
        
        slice_img = img_gray[ys:ye, :]
        slice_a11 = a11_map[ys:ye, :]
        slice_a22 = a22_map[ys:ye, :]
        
        m = slice_img > fiber_thresh
        a11_v = float(np.mean(slice_a11[m])) if m.sum() > 0 else float(np.mean(slice_a11))
        a22_v = float(np.mean(slice_a22[m])) if m.sum() > 0 else float(np.mean(slice_a22))
        f_pct = float(np.mean(m) * 100.0)
        
        y_mid = (ys + ye) / 2.0
        if y_mid < y_core_top:
            zone_name = "Top Skin (Dark Grainy)"
        elif y_mid < y_core_bot:
            zone_name = "Core (Thin Lighter Center)"
        else:
            zone_name = "Bottom Skin (Dark Grainy)"
            
        z_norm = 1.0 - 2.0 * (i + 0.5) / num_slices
        depth_mm = (i + 0.5) * (total_thick_mm / num_slices)
        
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
        'y_part_top': y_part_top,
        'y_core_top': y_core_top,
        'y_core_bot': y_core_bot,
        'y_part_bot': y_part_bot,
        'part_h_px': part_h_px,
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
        'e_predicted': e_predicted,
        'a11_map': a11_map,
        'a22_map': a22_map,
        'theta_deg': theta_deg,
        'df_profile': df_profile,
        'image_gray': img_gray
    }


def draw_5zone_overlay(img_gray, res, alpha=0.35):
    h, w = img_gray.shape
    overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    color_mask = overlay.copy()
    
    y_pt = res['y_part_top']
    y_ct = res['y_core_top']
    y_cb = res['y_core_bot']
    y_pb = res['y_part_bot']
    
    # 1. Top Non-Part (Dark Grey / Shaded)
    if y_pt > 0:
        color_mask[0:y_pt, :] = [35, 35, 35]
        
    # 2. Top Skin (Translucent Green)
    if y_ct > y_pt:
        color_mask[y_pt:y_ct, :] = [0, 200, 80]
        
    # 3. Core (Thin Lighter Section - Bright Amber/Yellow)
    if y_cb > y_ct:
        color_mask[y_ct:y_cb, :] = [0, 180, 255]
        
    # 4. Bottom Skin (Translucent Green)
    if y_pb > y_cb:
        color_mask[y_cb:y_pb, :] = [0, 200, 80]
        
    # 5. Bottom Non-Part (Dark Grey / Shaded)
    if y_pb < h - 1:
        color_mask[y_pb:h, :] = [35, 35, 35]
        
    cv2.addWeighted(color_mask, alpha, overlay, 1 - alpha, 0, overlay)
    
    # Draw boundary transition lines
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    
    # Top Non-Part Line
    if y_pt > 0:
        cv2.line(overlay, (0, y_pt), (w, y_pt), (200, 200, 200), 2)
        cv2.putText(overlay, f"Top Sample Surface (y={y_pt}px)", (15, max(y_pt - 8, 20)), font, font_scale, (200, 200, 200), 2)
        
    # Core Top Boundary Line
    if y_ct > y_pt:
        cv2.line(overlay, (0, y_ct), (w, y_ct), (0, 255, 255), 2)
        cv2.putText(overlay, f"Top Skin / Core Boundary (y={y_ct}px | Top Skin = {res['top_skin_mm']:.2f}mm / {res['top_skin_pct']:.1f}%)", (15, y_ct - 8), font, font_scale, (0, 255, 255), 2)
        
    # Core Bottom Boundary Line
    if y_cb < y_pb:
        cv2.line(overlay, (0, y_cb), (w, y_cb), (0, 255, 255), 2)
        cv2.putText(overlay, f"Core / Bottom Skin Boundary (y={y_cb}px | Core = {res['core_mm']:.2f}mm / {res['core_pct']:.1f}%)", (15, y_cb + 20), font, font_scale, (0, 255, 255), 2)
        
    # Bottom Non-Part Line
    if y_pb < h - 1:
        cv2.line(overlay, (0, y_pb), (w, y_pb), (200, 200, 200), 2)
        cv2.putText(overlay, f"Bottom Sample Surface (y={y_pb}px | Bot Skin = {res['bot_skin_mm']:.2f}mm / {res['bot_skin_pct']:.1f}%)", (15, min(y_pb + 20, h - 5)), font, font_scale, (200, 200, 200), 2)
        
    return overlay


# ---------------------------------------------------------
# INPUT SOURCE SELECTION
# ---------------------------------------------------------
col_src1, col_src2 = st.columns([1, 2])
with col_src1:
    input_mode = st.radio(
        "Select CT Scan Source:",
        ["📁 Preset CT Scan Images (Folder)", "📤 Upload Custom Image / CT Scan", "🧪 Synthetic Demo"],
        index=0
    )

selected_image_gray = None
source_title = ""

if input_mode == "📁 Preset CT Scan Images (Folder)":
    with col_src2:
        preset_choice = st.selectbox("Choose Specimen Image:", list(PRESET_IMAGES.keys()), index=0)
        preset_path = PRESET_IMAGES[preset_choice]
        if os.path.exists(preset_path):
            img_bgr = cv2.imread(preset_path)
            selected_image_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            source_title = preset_choice
        else:
            st.error(f"Image not found: {preset_path}")

elif input_mode == "📤 Upload Custom Image / CT Scan":
    with col_src2:
        uploaded_file = st.file_uploader("Upload 2D Cross-Section Image (PNG, JPG, TIFF):", type=["png", "jpg", "jpeg", "tif", "tiff"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            selected_image_gray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
            source_title = uploaded_file.name

elif input_mode == "🧪 Synthetic Demo":
    with col_src2:
        st.info("Synthetic cross-section: Non-Part ➔ Top Skin (Grainy) ➔ Core (Thin Light Center) ➔ Bottom Skin (Grainy) ➔ Non-Part.")
        syn_h, syn_w = 400, 800
        syn_img = np.full((syn_h, syn_w), 50, dtype=np.uint8)
        # Sample area: y=60 to y=340
        for i in range(60, 340):
            # Center core is at y=200
            dist_from_core = abs(i - 200)
            if dist_from_core < 10:
                base_val = 90 # thin lighter core
            else:
                base_val = 74 # grainy skin
            syn_img[i, :] = base_val
            for _ in range(6):
                cx = np.random.randint(10, syn_w - 10)
                length = np.random.randint(15, 30)
                ang = 0 if dist_from_core > 15 else np.pi/2
                dx = int(length * np.cos(ang))
                dy = int(length * np.sin(ang))
                cv2.line(syn_img, (cx - dx//2, i - dy//2), (cx + dx//2, i + dy//2), int(np.random.randint(160, 230)), 2)
        noise = np.random.normal(0, 4, (syn_h, syn_w)).astype(np.uint8)
        selected_image_gray = cv2.add(syn_img, noise)
        source_title = "Synthetic 5-Zone Cross-Section"

# ---------------------------------------------------------
# LAYER DETECTION & BOUNDARY TUNING
# ---------------------------------------------------------
res = None
if selected_image_gray is not None:
    h, w = selected_image_gray.shape
    
    # Auto-detect boundaries
    auto_pt, auto_ct, auto_cb, auto_pb = auto_detect_5zone_boundaries(selected_image_gray)
    
    with st.expander(":material/tune: Fine-Tune Layer Boundaries (Interactive Adjustment)", expanded=False):
        st.caption("Adjust the 4 transition boundaries across the image height if needed:")
        tb_c1, tb_c2, tb_c3, tb_c4 = st.columns(4)
        with tb_c1:
            sel_pt = st.number_input("Top Sample Surface (y px)", min_value=0, max_value=h-4, value=auto_pt, step=2)
        with tb_c2:
            sel_ct = st.number_input("Top Skin / Core Boundary (y px)", min_value=sel_pt+1, max_value=h-3, value=max(sel_pt+1, auto_ct), step=2)
        with tb_c3:
            sel_cb = st.number_input("Core / Bottom Skin Boundary (y px)", min_value=sel_ct+1, max_value=h-2, value=max(sel_ct+1, auto_cb), step=2)
        with tb_c4:
            sel_pb = st.number_input("Bottom Sample Surface (y px)", min_value=sel_cb+1, max_value=h-1, value=max(sel_cb+1, auto_pb), step=2)
            
    res = compute_layer_metrics(
        selected_image_gray, 
        sel_pt, 
        sel_ct, 
        sel_cb, 
        sel_pb, 
        total_thick_mm=specimen_thickness_mm,
        blur_ksize=blur_kernel_size,
        fiber_thresh=fiber_threshold
    )

# ---------------------------------------------------------
# TOP KPI ROW
# ---------------------------------------------------------
if res is not None:
    st.divider()
    kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
    with kpi_c1:
        st.metric(
            "Top & Bottom Skin Thickness", 
            f"{res['total_skin_mm']:.2f} mm", 
            f"{res['total_skin_pct']:.1f}% of sample", 
            help=f"Top Skin: {res['top_skin_mm']:.2f}mm ({res['top_skin_pct']:.1f}%) | Bottom Skin: {res['bot_skin_mm']:.2f}mm ({res['bot_skin_pct']:.1f}%)",
            border=True
        )
    with kpi_c2:
        st.metric(
            "Core Layer Thickness (Center)", 
            f"{res['core_mm']:.2f} mm", 
            f"{res['core_pct']:.1f}% of sample", 
            help=f"Thin lighter section centered in sample ({res['core_px']} pixels)",
            border=True
        )
    with kpi_c3:
        st.metric(
            "Skin-to-Core Ratio", 
            f"{res['total_skin_pct']:.1f}% / {res['core_pct']:.1f}%", 
            f"{res['total_skin_px']}px Skin / {res['core_px']}px Core", 
            help="Volumetric ratio of outer skin layers to central core layer",
            border=True
        )
    with kpi_c4:
        st.metric(
            "Estimated Fiber % (Vf)", 
            f"{res['global_fiber_pct']:.1f}%", 
            f"Predicted E: {res['e_predicted']:.2f} GPa", 
            help="Volume-averaged fiber percentage and predicted tensile stiffness for FEA",
            border=True
        )

st.write("")

# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
tab_bg, tab_layers, tab_profile, tab_vector_tab, tab_export = st.tabs([
    ":material/menu_book: Background & Layer Definitions",
    ":material/straighten: Layer Thickness Measurement",
    ":material/show_chart: Through-Thickness Orientation Profile",
    ":material/explore: Vector Field Overlay",
    ":material/table_chart: Layer Data & FEA Export"
])

# ---------------------------------------------------------
# TAB 1: BACKGROUND & LAYER DEFINITIONS
# ---------------------------------------------------------
with tab_bg:
    st.subheader("CT Scan Cross-Section Layer Definitions & Physical Model")
    
    st.info(
        "**Finite Element Analysis (FEA) Simulation Context:**\n\n"
        "In injection-molded short-glass-fiber reinforced polymers (SFRTPs), high mold wall shear rates freeze fibers parallel to the flow direction (forming the **Skin Layers**), while slower extensional flow at the center causes fibers to orient transversely (forming the **Core Layer**).\n\n"
        "Capturing this through-thickness **fiber variance** is critical for establishing realistic anisotropic material cards in FEA solvers (such as Abaqus, ANSYS, Moldflow, and Digimat). Assuming uniform isotropic stiffness leads to significant discrepancies in structural deflection, stress concentration, and failure predictions."
    )
    
    st.markdown("### 🔬 CT Scan Layer Identification Guide")
    st.markdown(
        "The cross-section image follows the exact physical progression from top to bottom:\n\n"
        "1. ⬛ **Top Non-Part (Smooth Dark Grey):** Air / mounting medium above the sample ($0\\text{ px}$ if image is cropped to sample surface).\n"
        "2. 🟢 **Top Skin (Dark Grainy Section):** Outer skin region where fibers are frozen parallel to mold flow.\n"
        "3. 🟡 **Core (Thin Lighter Section Centered in the Sample):** Central transverse core region appearing as a distinct lighter band.\n"
        "4. 🟢 **Bottom Skin (Dark Grainy Section):** Outer skin region where fibers are frozen parallel to bottom mold wall.\n"
        "5. ⬛ **Bottom Non-Part (Smooth Dark Grey):** Air / mounting medium below the sample ($0\\text{ px}$ if image is cropped to sample surface)."
    )
    
    with st.container(border=True):
        st.markdown(
            "#### ⚠️ Why Inputting Full Cross-Sections (Skin + Core) is Required for FEA\n\n"
            "- **Top-Layer Only (XY Surface Slice):** Capturing only the surface of the mold part reflects *only* the skin layer where fibers are frozen parallel to the mold wall. This produces an artificially high stiffness estimate and completely misses the transverse core.\n\n"
            "- **Full Cross-Section (XZ / YZ Plane):** Capturing the complete thickness reveals both the outer skin layers ($A_{11} \\ge 0.5$) and the transverse core layer ($A_{11} < 0.5$). This enables the analyzer to quantify true layer thicknesses, locate transition inflection points, calculate the skin-to-core ratio, and compute volume-averaged tensor properties for FEA."
        )
        
    st.markdown("### 🖼️ Example Images from CT Scan Dataset")
    st.caption("Visual comparison of valid through-thickness cross-section vs. top-layer surface slice:")
    
    ex_c1, ex_c2 = st.columns(2)
    with ex_c1:
        with st.container(border=True):
            st.markdown("**:material/check_circle: Valid Input: Full Cross-Section (XZ Plane with Skin & Core)**")
            xz_img_path = PRESET_IMAGES["XZ Plane: Cross-Section with Skin-Core-Skin (black, 90deg, xz plane with core.png)"]
            if os.path.exists(xz_img_path):
                st.image(xz_img_path, width="stretch", caption="Example: XZ Plane Cross-Section showing distinct Skin & Core Layers")
            st.markdown(
                "- **Contains:** Non-Part ➔ Top Skin (Dark Grainy) ➔ Core (Thin Light Center) ➔ Bottom Skin (Dark Grainy) ➔ Non-Part.\n"
                "- **Result:** Accurately measures individual skin/core thicknesses and through-thickness orientation tensors for FEA."
            )
            
    with ex_c2:
        with st.container(border=True):
            st.markdown("**:material/cancel: Insufficient Input: Top Surface Layer Only (XY Plane)**")
            xy_img_path = PRESET_IMAGES["XY Plane: Surface Slice - Only Skin (black, 90deg, xy plane, only skin.png)"]
            if os.path.exists(xy_img_path):
                st.image(xy_img_path, width="stretch", caption="Example: XY Plane Slice showing Only Skin (Surface Layer)")
            st.markdown(
                "- **Limitation:** Captures only the superficial skin layer without through-thickness core context.\n"
                "- **Risk:** Severe sampling bias; leads to over-estimated stiffness and inaccurate FEA predictions."
            )

# ---------------------------------------------------------
# TAB 2: LAYER THICKNESS MEASUREMENT
# ---------------------------------------------------------
with tab_layers:
    if res is not None:
        st.subheader("Physical Layer Thickness Measurement & Segmentation")
        st.markdown(
            f"**Specimen:** `{source_title}` | Total Image Height: **{res['h']} px** | "
            f"Physical Part Thickness: **{res['part_h_px']} px** (**{specimen_thickness_mm:.2f} mm**)"
        )
        
        # 5-Zone Metric Cards
        z1, z2, z3, z4, z5 = st.columns(5)
        with z1:
            val_txt = f"{res['top_nonpart_mm']:.2f} mm" if res['top_nonpart_px'] > 0 else "0.0 mm (Cropped)"
            st.metric("1. Top Non-Part (Smooth)", val_txt, f"{res['top_nonpart_px']} px", border=True)
        with z2:
            st.metric("2. Top Skin (Dark Grainy)", f"{res['top_skin_mm']:.2f} mm", f"{res['top_skin_pct']:.1f}% ({res['top_skin_px']} px)", border=True)
        with z3:
            st.metric("3. Core (Thin Light Center)", f"{res['core_mm']:.2f} mm", f"{res['core_pct']:.1f}% ({res['core_px']} px)", border=True)
        with z4:
            st.metric("4. Bottom Skin (Dark Grainy)", f"{res['bot_skin_mm']:.2f} mm", f"{res['bot_skin_pct']:.1f}% ({res['bot_skin_px']} px)", border=True)
        with z5:
            val_txt = f"{res['bot_nonpart_mm']:.2f} mm" if res['bot_nonpart_px'] > 0 else "0.0 mm (Cropped)"
            st.metric("5. Bottom Non-Part (Smooth)", val_txt, f"{res['bot_nonpart_px']} px", border=True)
            
        st.write("")
        st.markdown("### Visual Layer Segmentation Overlay on CT Scan Image")
        st.caption(
            "⬛ Shaded: Non-Part Background | 🟢 Green: Top & Bottom Skin Layers (Dark Grainy) | "
            "🟡 Yellow/Amber: Core Layer (Thin Lighter Center Band)"
        )
        
        overlay_img = draw_5zone_overlay(res['image_gray'], res)
        st.image(overlay_img, width="stretch", caption=f"5-Zone Layer Segmentation Overlay for {source_title}")
        
        st.write("")
        # Thickness Breakdown Table & Pie Chart
        bk_c1, bk_c2 = st.columns([1, 1])
        with bk_c1:
            with st.container(border=True):
                st.markdown("**Layer Thickness Breakdown Table:**")
                table_data = [
                    {"Zone": "1. Top Non-Part (Smooth Dark Grey)", "Thickness (px)": res['top_nonpart_px'], "Thickness (mm)": f"{res['top_nonpart_mm']:.3f}", "% of Physical Part": "N/A (Background)"},
                    {"Zone": "2. Top Skin Layer (Dark Grainy)", "Thickness (px)": res['top_skin_px'], "Thickness (mm)": f"{res['top_skin_mm']:.3f}", "% of Physical Part": f"{res['top_skin_pct']:.1f}%"},
                    {"Zone": "3. Core Layer (Thin Lighter Center)", "Thickness (px)": res['core_px'], "Thickness (mm)": f"{res['core_mm']:.3f}", "% of Physical Part": f"{res['core_pct']:.1f}%"},
                    {"Zone": "4. Bottom Skin Layer (Dark Grainy)", "Thickness (px)": res['bot_skin_px'], "Thickness (mm)": f"{res['bot_skin_mm']:.3f}", "% of Physical Part": f"{res['bot_skin_pct']:.1f}%"},
                    {"Zone": "5. Bottom Non-Part (Smooth Dark Grey)", "Thickness (px)": res['bot_nonpart_px'], "Thickness (mm)": f"{res['bot_nonpart_mm']:.3f}", "% of Physical Part": "N/A (Background)"},
                    {"Zone": "TOTAL PHYSICAL PART", "Thickness (px)": res['part_h_px'], "Thickness (mm)": f"{specimen_thickness_mm:.3f}", "% of Physical Part": "100.0%"}
                ]
                st.dataframe(pd.DataFrame(table_data), width="stretch", hide_index=True)
                
        with bk_c2:
            with st.container(border=True):
                st.markdown("**Skin vs. Core Volume Ratio Pie Chart:**")
                fig_pie = px.pie(
                    values=[res['top_skin_pct'], res['core_pct'], res['bot_skin_pct']],
                    names=['Top Skin (Grainy)', 'Core (Light Center)', 'Bottom Skin (Grainy)'],
                    color=['Top Skin (Grainy)', 'Core (Light Center)', 'Bottom Skin (Grainy)'],
                    color_discrete_map={'Top Skin (Grainy)': '#10B981', 'Core (Light Center)': '#F59E0B', 'Bottom Skin (Grainy)': '#059669'},
                    hole=0.4
                )
                st.plotly_chart(fig_pie, width="stretch")
    else:
        st.info("Please select or upload a CT scan image to measure layer thicknesses.")

# ---------------------------------------------------------
# TAB 3: THROUGH-THICKNESS ORIENTATION PROFILE
# ---------------------------------------------------------
with tab_profile:
    if res is not None:
        st.subheader("Through-Thickness Orientation ($A_{11}$ vs $A_{22}$) and Fiber % Profiles")
        st.markdown(
            "Profile mapped across normalized thickness $z_{\\text{norm}} \\in [-1.0, 1.0]$ "
            "($+1.0 = \\text{Top Skin Surface}$, $0.0 = \\text{Center Core}$, $-1.0 = \\text{Bottom Skin Surface}$)."
        )
        
        with st.container(border=True):
            fig_prof = go.Figure()
            
            # A11 Trace
            fig_prof.add_trace(go.Scatter(
                x=res['df_profile']['Normalized_Z'],
                y=res['df_profile']['A11'],
                mode='lines+markers',
                name='A11 (Parallel to Tensile Flow Axis)',
                line=dict(color='#10B981', width=3),
                marker=dict(size=6)
            ))
            
            # A22 Trace
            fig_prof.add_trace(go.Scatter(
                x=res['df_profile']['Normalized_Z'],
                y=res['df_profile']['A22'],
                mode='lines+markers',
                name='A22 (Perpendicular Transverse)',
                line=dict(color='#EF4444', width=2, dash='dot'),
                marker=dict(size=5)
            ))
            
            # Center Core Line
            fig_prof.add_vline(x=0.0, line_dash="solid", line_color="gray", opacity=0.4, annotation_text="Center Core (z=0)")
            
            fig_prof.update_layout(
                title=f"Through-Thickness Orientation Tensor Profile — {source_title}",
                xaxis_title="Normalized Thickness z_norm [ +1.0 = Top Mold Skin, 0.0 = Center Core, -1.0 = Bottom Mold Skin ]",
                yaxis_title="Orientation Tensor Component Value (0.0 to 1.0)",
                yaxis=dict(range=[0.0, 1.0]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_prof, width="stretch")
            
        with st.container(border=True):
            fig_fib = px.line(
                res['df_profile'],
                x='Normalized_Z',
                y='Fiber_Pct',
                markers=True,
                title="Through-Thickness Fiber Density / Volume Fraction Profile (%)",
                labels={'Normalized_Z': 'Normalized Thickness z_norm', 'Fiber_Pct': 'Fiber Percentage (%)'}
            )
            fig_fib.update_traces(line_color='#3B82F6', line_width=2.5)
            st.plotly_chart(fig_fib, width="stretch")
    else:
        st.info("Please select or upload a CT scan image to view through-thickness profile curves.")

# ---------------------------------------------------------
# TAB 4: VECTOR FIELD OVERLAY
# ---------------------------------------------------------
with tab_vector_tab:
    if res is not None:
        st.subheader("Local Fiber Orientation Vector Field Overlay")
        st.markdown("Visualizes local fiber orientation vectors computed via Structure Tensor gradients across the 2D cross-section.")
        
        vec_step = st.slider("Vector Grid Density", min_value=12, max_value=48, value=24, step=4)
        
        h, w = res['image_gray'].shape
        vec_overlay = cv2.cvtColor(res['image_gray'], cv2.COLOR_GRAY2BGR)
        
        for y in range(vec_step // 2, h, vec_step):
            for x in range(vec_step // 2, w, vec_step):
                ang = res['theta_deg'][y, x]
                a11_val = res['a11_map'][y, x]
                
                rad = np.radians(ang)
                if a11_val >= 0.65:
                    color = (0, 230, 100)   # Green (aligned with flow)
                elif a11_val >= 0.45:
                    color = (0, 180, 255)   # Amber (intermediate)
                else:
                    color = (30, 40, 240)   # Red (transverse)
                    
                dx = int(18 * np.cos(rad))
                dy = int(18 * np.sin(rad))
                
                pt1 = (x - dx, y + dy)
                pt2 = (x + dx, y - dy)
                cv2.arrowedLine(vec_overlay, pt1, pt2, color, 1, tipLength=0.3)
                
        st.image(vec_overlay, width="stretch", caption=f"Vector Field Overlay for {source_title}")
        
        v_col1, v_col2 = st.columns(2)
        with v_col1:
            with st.container(border=True):
                st.markdown("**Original Raw CT Grayscale Scan**")
                st.image(res['image_gray'], width="stretch")
        with v_col2:
            with st.container(border=True):
                st.markdown("**5-Zone Layer Segmentation Overlay**")
                st.image(overlay_img, width="stretch")
    else:
        st.info("Please select or upload a CT scan image to view vector field overlays.")

# ---------------------------------------------------------
# TAB 5: DATA & FEA EXPORT
# ---------------------------------------------------------
with tab_data:
    if res is not None:
        st.subheader("Through-Thickness Orientation Tensor Data Table")
        st.markdown("Detailed tabular orientation and fiber percentage distribution formatted for FEA material calibration (Abaqus, ANSYS, Moldflow).")
        
        st.dataframe(res['df_profile'], width="stretch", hide_index=True)
        
        csv_export = res['df_profile'].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Through-Thickness Tensor CSV for FEA",
            data=csv_export,
            file_name=f"ct_orientation_tensor_{source_title.replace(' ', '_')}.csv",
            mime='text/csv'
        )
    else:
        st.info("Please select or upload a CT scan image to export data.")

st.markdown("<br><br><p style='text-align: center; font-size: 11px; color: gray;'>CT Fiber Orientation & Layer Thickness Analyzer | AA/MSE Product Design</p>", unsafe_allow_html=True)
