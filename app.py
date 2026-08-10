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
    "XZ Plane: Through-Thickness with Core & Skin (black, 90deg, xz plane with core.png)": os.path.join(CT_IMAGES_DIR, "black, 90deg, xz plane with core.png"),
    "XZ Plane: Through-Thickness with Core #2 (black, 90deg, xz plane with core2.png)": os.path.join(CT_IMAGES_DIR, "black, 90deg, xz plane with core2.png"),
    "XY Plane: Surface Slice - Only Skin (black, 90deg, xy plane, only skin.png)": os.path.join(CT_IMAGES_DIR, "black, 90deg, xy plane, only skin.png"),
    "XY Plane: Surface Slice - Only Skin #2 (black, 90deg, xy plane, only skin2.png)": os.path.join(CT_IMAGES_DIR, "black, 90deg, xy plane, only skin2.png"),
    "0° Cut Reference Slice (sample_ct_0deg.png)": os.path.join(BASE_DIR, "sample_ct_0deg.png"),
    "45° Cut Reference Slice (sample_ct_45deg.png)": os.path.join(BASE_DIR, "sample_ct_45deg.png"),
    "90° Cut Reference Slice (sample_ct_90deg.png)": os.path.join(BASE_DIR, "sample_ct_90deg.png")
}

st.title(":material/layers: CT Scan Fiber Orientation & Layer Thickness Analyzer")
st.markdown(
    "Automated research tool enforcing the physical cross-section model: "
    "**`Non-Part ➔ Top Skin ➔ Core ➔ Bottom Skin ➔ Non-Part`**. "
    "Accurately measures layer thicknesses, calculates fiber volume fraction ($V_f$), and maps orientation tensors for FEA simulation."
)

# ---------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------
with st.sidebar:
    st.header(":material/settings: Physical & Material Settings")
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
    st.header(":material/tune: 5-Zone Model Parameters")
    skin_a11_threshold = st.slider(
        "Skin/Core A11 Threshold", 
        min_value=0.30, 
        max_value=0.70, 
        value=0.50, 
        step=0.02, 
        help="A11 >= threshold classified as Skin (aligned with flow); A11 < threshold classified as Core"
    )
    bg_intensity_threshold = st.slider(
        "Non-Part Background Threshold", 
        min_value=5, 
        max_value=100, 
        value=25, 
        help="Grayscale intensity below which rows are classified as Non-Part (Air / Mounting void)"
    )
    blur_kernel_size = st.slider(
        "Gaussian Blur Kernel Size", 
        min_value=3, 
        max_value=41, 
        value=21, 
        step=2, 
        help="Structure tensor spatial smoothing window"
    )
    fiber_threshold = st.slider(
        "Fiber Intensity Threshold", 
        min_value=10, 
        max_value=200, 
        value=55, 
        help="Grayscale threshold to segment fiber pixels from resin matrix"
    )


# ---------------------------------------------------------
# 5-ZONE PHYSICAL LAYER RECOGNITION ALGORITHM
# Enforces: Non-Part -> Top Skin -> Core -> Bottom Skin -> Non-Part
# ---------------------------------------------------------
def analyze_5zone_cross_section(
    img_gray, 
    blur_ksize=21, 
    fiber_thresh=55, 
    skin_thresh=0.5, 
    bg_thresh=25, 
    total_thick_mm=2.0,
    manual_bounds=None
):
    h, w = img_gray.shape
    
    # Structure tensor computation
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
    
    if manual_bounds is not None:
        y_part_start, y_core_start, y_core_end, y_part_end = manual_bounds
    else:
        # STEP 1: Detect Part vs Non-Part (top and bottom background)
        row_means = np.mean(img_gray, axis=1)
        row_smooth = cv2.GaussianBlur(row_means.reshape(-1, 1), (1, 15), 0).flatten()
        
        valid_rows = np.where(row_smooth > bg_thresh)[0]
        if len(valid_rows) > 0:
            y_part_start = int(valid_rows[0])
            y_part_end = int(valid_rows[-1])
            # If within 10px of border, treat as tightly cropped to sample surface
            if y_part_start < 10:
                y_part_start = 0
            if (h - 1 - y_part_end) < 10:
                y_part_end = h - 1
        else:
            y_part_start = 0
            y_part_end = h - 1
            
        part_h = max(y_part_end - y_part_start + 1, 1)
        
        # STEP 2: Enforce Top Skin -> Core -> Bottom Skin inside Part
        a11_prof = np.mean(a11_map[y_part_start:y_part_end+1, :], axis=1)
        a11_smooth = cv2.GaussianBlur(a11_prof.reshape(-1, 1), (1, 31), 0).flatten()
        
        mid_rel = part_h // 2
        
        # Search downward from top part surface for core transition (A11 < skin_thresh)
        core_start_rel = 0
        for y in range(mid_rel):
            if a11_smooth[y] < skin_thresh:
                core_start_rel = y
                break
        if core_start_rel == 0 and a11_smooth[0] >= skin_thresh:
            core_start_rel = int(np.argmin(a11_smooth[:mid_rel]))
            
        # Search upward from bottom part surface for core transition
        core_end_rel = part_h - 1
        for y in range(part_h - 1, mid_rel, -1):
            if a11_smooth[y] < skin_thresh:
                core_end_rel = y
                break
        if core_end_rel == part_h - 1 and a11_smooth[-1] >= skin_thresh:
            core_end_rel = mid_rel + int(np.argmin(a11_smooth[mid_rel:]))
            
        # Enforce ordering y_part_start <= y_core_start <= y_core_end <= y_part_end
        if core_start_rel > core_end_rel:
            core_start_rel = mid_rel
            core_end_rel = mid_rel
            
        y_core_start = y_part_start + core_start_rel
        y_core_end = y_part_start + core_end_rel
        
    # Enforce bounds clamping
    y_part_start = max(0, min(y_part_start, h - 1))
    y_core_start = max(y_part_start, min(y_core_start, h - 1))
    y_core_end = max(y_core_start, min(y_core_end, h - 1))
    y_part_end = max(y_core_end, min(y_part_end, h - 1))
    
    part_h_px = max(y_part_end - y_part_start + 1, 1)
    
    # 5-Zone Layer Thicknesses
    top_nonpart_px = y_part_start
    top_skin_px = y_core_start - y_part_start
    core_px = y_core_end - y_core_start
    bot_skin_px = y_part_end - y_core_end
    bot_nonpart_px = h - 1 - y_part_end
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
    
    # Compute Fiber % and Tensor Metrics on Part
    part_mask = (img_gray[y_part_start:y_part_end+1, :] > fiber_thresh)
    global_fiber_pct = float(np.mean(part_mask) * 100.0)
    
    global_a11 = float(np.mean(a11_map[y_part_start:y_part_end+1, :]))
    global_a22 = 1.0 - global_a11
    
    v_f = global_fiber_pct / 100.0
    e_predicted = e_matrix_gpa + global_a11 * v_f * (e_fiber_gpa - e_matrix_gpa)
    
    # Build detailed discrete slice table for plotting through-thickness curve
    num_sub_slices = 50
    step = part_h_px / num_sub_slices
    sub_data = []
    for i in range(num_sub_slices):
        ys = y_part_start + int(round(i * step))
        ye = y_part_start + int(round((i + 1) * step))
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
        
        # Determine 5-zone classification
        y_mid = (ys + ye) / 2.0
        if y_mid < y_part_start:
            zone_name = "Top Non-Part"
        elif y_mid < y_core_start:
            zone_name = "Top Skin Layer"
        elif y_mid < y_core_end:
            zone_name = "Core Layer"
        elif y_mid <= y_part_end:
            zone_name = "Bottom Skin Layer"
        else:
            zone_name = "Bottom Non-Part"
            
        z_norm = 1.0 - 2.0 * (i + 0.5) / num_sub_slices
        depth_mm = (i + 0.5) * (total_thick_mm / num_sub_slices)
        
        sub_data.append({
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
        
    df_profile = pd.DataFrame(sub_data)
    
    return {
        'height': h,
        'width': w,
        'y_part_start': y_part_start,
        'y_core_start': y_core_start,
        'y_core_end': y_core_end,
        'y_part_end': y_part_end,
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


def draw_5zone_segmentation_overlay(img_gray, res, alpha=0.35):
    h, w = img_gray.shape
    overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    color_mask = overlay.copy()
    
    y_ps = res['y_part_start']
    y_cs = res['y_core_start']
    y_ce = res['y_core_end']
    y_pe = res['y_part_end']
    
    # 1. Top Non-Part (Dark gray/black shading)
    if y_ps > 0:
        color_mask[0:y_ps, :] = [40, 40, 40]
        
    # 2. Top Skin (Translucent Green)
    if y_cs > y_ps:
        color_mask[y_ps:y_cs, :] = [0, 220, 100]
        
    # 3. Core Layer (Translucent Red/Orange)
    if y_ce > y_cs:
        color_mask[y_cs:y_ce, :] = [30, 50, 230]
        
    # 4. Bottom Skin (Translucent Green)
    if y_pe > y_ce:
        color_mask[y_ce:y_pe, :] = [0, 220, 100]
        
    # 5. Bottom Non-Part (Dark gray/black shading)
    if y_pe < h - 1:
        color_mask[y_pe:h, :] = [40, 40, 40]
        
    cv2.addWeighted(color_mask, alpha, overlay, 1 - alpha, 0, overlay)
    
    # Draw demarcation transition lines & text badges
    line_thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    
    if y_ps > 0:
        cv2.line(overlay, (0, y_ps), (w, y_ps), (200, 200, 200), line_thickness)
        cv2.putText(overlay, f"Top Sample Surface (y={y_ps}px)", (15, max(y_ps - 8, 20)), font, font_scale, (200, 200, 200), 2)
        
    if y_cs > y_ps:
        cv2.line(overlay, (0, y_cs), (w, y_cs), (0, 255, 255), line_thickness)
        cv2.putText(overlay, f"Top Skin / Core Boundary (y={y_cs}px | {res['top_skin_mm']:.2f}mm)", (15, y_cs - 8), font, font_scale, (0, 255, 255), 2)
        
    if y_ce < y_pe:
        cv2.line(overlay, (0, y_ce), (w, y_ce), (0, 255, 255), line_thickness)
        cv2.putText(overlay, f"Core / Bottom Skin Boundary (y={y_ce}px | {res['core_mm']:.2f}mm)", (15, y_ce + 20), font, font_scale, (0, 255, 255), 2)
        
    if y_pe < h - 1:
        cv2.line(overlay, (0, y_pe), (w, y_pe), (200, 200, 200), line_thickness)
        cv2.putText(overlay, f"Bottom Sample Surface (y={y_pe}px)", (15, min(y_pe + 20, h - 5)), font, font_scale, (200, 200, 200), 2)
        
    return overlay


# ---------------------------------------------------------
# INPUT SELECTION
# ---------------------------------------------------------
col_src1, col_src2 = st.columns([1, 2])
with col_src1:
    input_mode = st.radio(
        "Select CT Scan Source:",
        ["📁 Preset CT Scan Images (Folder)", "📤 Upload Custom Image / CT Scan", "🧪 Synthetic 3D Demo"],
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

elif input_mode == "🧪 Synthetic 3D Demo":
    with col_src2:
        st.info("Synthetic cross-section with Top Skin (aligned) -> Center Core (transverse) -> Bottom Skin (aligned).")
        syn_h, syn_w = 400, 800
        syn_img = np.full((syn_h, syn_w), 45, dtype=np.uint8)
        for i in range(syn_h):
            z_n = 1.0 - 2.0 * i / (syn_h - 1)
            target_ang = 90.0 * (1.0 - abs(z_n)**1.5)
            rad = np.radians(target_ang)
            for _ in range(8):
                cx = np.random.randint(10, syn_w - 10)
                length = np.random.randint(15, 30)
                dx = int(length * np.cos(rad))
                dy = int(length * np.sin(rad))
                cv2.line(syn_img, (cx - dx//2, i - dy//2), (cx + dx//2, i + dy//2), int(np.random.randint(180, 240)), 2)
        noise = np.random.normal(0, 5, (syn_h, syn_w)).astype(np.uint8)
        selected_image_gray = cv2.add(syn_img, noise)
        source_title = "Synthetic Skin-Core Cross-Section"

# Run 5-Zone Analysis
res = None
if selected_image_gray is not None:
    res = analyze_5zone_cross_section(
        selected_image_gray,
        blur_ksize=blur_kernel_size,
        fiber_thresh=fiber_threshold,
        skin_thresh=skin_a11_threshold,
        bg_thresh=bg_intensity_threshold,
        total_thick_mm=specimen_thickness_mm
    )

# ---------------------------------------------------------
# TOP KPI ROW
# ---------------------------------------------------------
if res is not None:
    st.divider()
    kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
    with kpi_c1:
        st.metric(
            "Total Skin Layer Thickness", 
            f"{res['total_skin_mm']:.2f} mm", 
            f"{res['total_skin_pct']:.1f}% of sample", 
            help=f"Top Skin: {res['top_skin_mm']:.2f}mm ({res['top_skin_pct']:.1f}%) | Bottom Skin: {res['bot_skin_mm']:.2f}mm ({res['bot_skin_pct']:.1f}%)",
            border=True
        )
    with kpi_c2:
        st.metric(
            "Core Layer Thickness", 
            f"{res['core_mm']:.2f} mm", 
            f"{res['core_pct']:.1f}% of sample", 
            help="Internal core region where fibers are transverse to tensile flow direction",
            border=True
        )
    with kpi_c3:
        st.metric(
            "Skin-to-Core Ratio", 
            f"{res['total_skin_pct']:.1f}% / {res['core_pct']:.1f}%", 
            f"{res['total_skin_px']}px Skin / {res['core_px']}px Core", 
            help="Ratio of aligned skin layers to transverse core layer",
            border=True
        )
    with kpi_c4:
        st.metric(
            "Estimated Fiber % (Vf)", 
            f"{res['global_fiber_pct']:.1f}%", 
            f"E_pred: {res['e_predicted']:.2f} GPa", 
            help="Fiber percentage from grayscale thresholding & Halpin-Tsai predicted tensile modulus",
            border=True
        )

st.write("")

# ---------------------------------------------------------
# APPLICATION TABS
# ---------------------------------------------------------
tab_bg, tab_layers, tab_profile, tab_vector_tab, tab_export = st.tabs([
    ":material/menu_book: Background & Physical Rule",
    ":material/straighten: 5-Zone Layer Thickness Measurement",
    ":material/show_chart: Through-Thickness Orientation Profile",
    ":material/explore: Vector Field Overlay",
    ":material/table_chart: Layer Data & FEA Export"
])

# ---------------------------------------------------------
# TAB 1: BACKGROUND & PHYSICAL RULE
# ---------------------------------------------------------
with tab_bg:
    st.subheader("Physical Cross-Section Rule & Microstructure Background")
    
    st.info(
        "**Finite Element Analysis (FEA) Simulation Context:**\n\n"
        "In injection-molded short-glass-fiber reinforced polymers (SFRTPs), high mold wall shear rates freeze fibers parallel to the flow direction (forming the **Skin Layer**), while slower extensional flow at the center causes fibers to orient transversely (forming the **Core Layer**).\n\n"
        "Capturing this through-thickness **fiber variance** is critical for establishing realistic anisotropic material cards in FEA solvers (such as Abaqus, ANSYS, Moldflow, and Digimat). Assuming uniform isotropic stiffness leads to significant discrepancies in structural deflection, stress concentration, and failure predictions."
    )
    
    st.markdown("### 📏 Enforced Physical Cross-Section Model")
    st.markdown(
        "The analyzer enforces the strict physical sequence across the specimen height:\n\n"
        "$$\\Large \\text{Non-Part (Top)} \\longrightarrow \\text{Top Skin} \\longrightarrow \\text{Core} \\longrightarrow \\text{Bottom Skin} \\longrightarrow \\text{Non-Part (Bottom)}$$\n\n"
        "- **Tightly Cropped Images:** When the image is cropped directly to the physical sample boundary, the **Non-Part** regions will automatically have $0\\text{ px}$ ($0.0\\text{ mm}$) thickness.\n"
        "- **Uncropped / Mounted Images:** Any air, mounting medium, or dark background above or below the sample is segmented out as Non-Part, ensuring thickness measurements are computed strictly on the physical part."
    )
    
    with st.container(border=True):
        st.markdown(
            "#### ⚠️ Why Analyzing Only the Top Layer is Insufficient for Determining Orientation\n\n"
            "- **Top-Layer Only (XY Surface Slice):** Capturing only the surface of the mold part reflects *only* the highly aligned skin layer ($A_{11} \\approx 0.7 - 0.9$). Analyzing this alone introduces severe sampling bias, falsely indicating that the entire cross-section is aligned in the flow direction. This artificially inflates predicted tensile stiffness and conceals the compliant transverse core.\n\n"
            "- **Full Cross-Section (XZ / YZ Plane):** Capturing the complete thickness reveals both the outer skin layers ($A_{11} \\ge 0.5$) and the transverse core layer ($A_{11} < 0.5$). This enables the analyzer to quantify true layer thicknesses, locate transition inflection points, calculate the skin-to-core ratio, and compute volume-averaged tensor properties for FEA."
        )
        
    st.markdown("### 🖼️ Example Images from CT Scan Dataset")
    st.caption("Comparison of valid through-thickness cross-section vs. insufficient surface-only slice:")
    
    ex_c1, ex_c2 = st.columns(2)
    with ex_c1:
        with st.container(border=True):
            st.markdown("**:material/check_circle: Valid Input: Full Cross-Section (XZ Plane with Skin & Core)**")
            xz_img_path = PRESET_IMAGES["XZ Plane: Through-Thickness with Core & Skin (black, 90deg, xz plane with core.png)"]
            if os.path.exists(xz_img_path):
                st.image(xz_img_path, width="stretch", caption="Example: XZ Plane Cross-Section showing distinct Skin & Core Layers")
            st.markdown(
                "- **Contains:** Top Skin, Center Transverse Core, and Bottom Skin.\n"
                "- **Result:** Allows measuring individual skin/core thicknesses and true orientation tensors for FEA."
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
# TAB 2: 5-ZONE LAYER THICKNESS MEASUREMENT
# ---------------------------------------------------------
with tab_layers:
    if res is not None:
        st.subheader("5-Zone Layer Thickness Segmentation & Measurement")
        st.markdown(
            f"**Specimen:** `{source_title}` | Total Image Height: **{res['height']} px** | "
            f"Part Thickness: **{res['part_h_px']} px** (**{specimen_thickness_mm:.2f} mm**)"
        )
        
        # 5-Zone Metric Cards
        z1, z2, z3, z4, z5 = st.columns(5)
        with z1:
            val_txt = f"{res['top_nonpart_mm']:.2f} mm" if res['top_nonpart_px'] > 0 else "0.0 mm (Cropped)"
            st.metric("1. Top Non-Part", val_txt, f"{res['top_nonpart_px']} px", border=True)
        with z2:
            st.metric("2. Top Skin Layer", f"{res['top_skin_mm']:.2f} mm", f"{res['top_skin_pct']:.1f}% ({res['top_skin_px']} px)", border=True)
        with z3:
            st.metric("3. Core Layer", f"{res['core_mm']:.2f} mm", f"{res['core_pct']:.1f}% ({res['core_px']} px)", border=True)
        with z4:
            st.metric("4. Bottom Skin Layer", f"{res['bot_skin_mm']:.2f} mm", f"{res['bot_skin_pct']:.1f}% ({res['bot_skin_px']} px)", border=True)
        with z5:
            val_txt = f"{res['bot_nonpart_mm']:.2f} mm" if res['bot_nonpart_px'] > 0 else "0.0 mm (Cropped)"
            st.metric("5. Bottom Non-Part", val_txt, f"{res['bot_nonpart_px']} px", border=True)
            
        st.write("")
        st.markdown("### Visual 5-Zone Segmentation Overlay on CT Scan Image")
        st.caption(
            "⬛ Gray: Non-Part Background | 🟢 Green: Top/Bottom Skin Layers (A11 ≥ threshold) | "
            "🔴 Red: Core Layer (A11 < threshold) | 🟡 Yellow: Transition Boundaries"
        )
        
        overlay_img = draw_5zone_segmentation_overlay(res['image_gray'], res)
        st.image(overlay_img, width="stretch", caption=f"5-Zone Layer Segmentation Overlay for {source_title}")
        
        st.write("")
        # Thickness Breakdown Table & Pie Chart
        bk_c1, bk_c2 = st.columns([1, 1])
        with bk_c1:
            with st.container(border=True):
                st.markdown("**5-Zone Layer Thickness Breakdown Table:**")
                table_data = [
                    {"Zone": "1. Top Non-Part (Air / Void)", "Thickness (px)": res['top_nonpart_px'], "Thickness (mm)": f"{res['top_nonpart_mm']:.3f}", "% of Physical Part": "N/A (Background)"},
                    {"Zone": "2. Top Skin Layer (Flow Aligned)", "Thickness (px)": res['top_skin_px'], "Thickness (mm)": f"{res['top_skin_mm']:.3f}", "% of Physical Part": f"{res['top_skin_pct']:.1f}%"},
                    {"Zone": "3. Core Layer (Transverse/Random)", "Thickness (px)": res['core_px'], "Thickness (mm)": f"{res['core_mm']:.3f}", "% of Physical Part": f"{res['core_pct']:.1f}%"},
                    {"Zone": "4. Bottom Skin Layer (Flow Aligned)", "Thickness (px)": res['bot_skin_px'], "Thickness (mm)": f"{res['bot_skin_mm']:.3f}", "% of Physical Part": f"{res['bot_skin_pct']:.1f}%"},
                    {"Zone": "5. Bottom Non-Part (Air / Void)", "Thickness (px)": res['bot_nonpart_px'], "Thickness (mm)": f"{res['bot_nonpart_mm']:.3f}", "% of Physical Part": "N/A (Background)"},
                    {"Zone": "TOTAL PHYSICAL PART", "Thickness (px)": res['part_h_px'], "Thickness (mm)": f"{specimen_thickness_mm:.3f}", "% of Physical Part": "100.0%"}
                ]
                st.dataframe(pd.DataFrame(table_data), width="stretch", hide_index=True)
                
        with bk_c2:
            with st.container(border=True):
                st.markdown("**Skin vs. Core Volume Ratio Pie Chart:**")
                fig_pie = px.pie(
                    values=[res['top_skin_pct'], res['core_pct'], res['bot_skin_pct']],
                    names=['Top Skin Layer', 'Core Layer', 'Bottom Skin Layer'],
                    color=['Top Skin Layer', 'Core Layer', 'Bottom Skin Layer'],
                    color_discrete_map={'Top Skin Layer': '#10B981', 'Core Layer': '#EF4444', 'Bottom Skin Layer': '#059669'},
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
            
            # Threshold Line
            fig_prof.add_hline(
                y=skin_a11_threshold, 
                line_dash="dash", 
                line_color="#F59E0B", 
                annotation_text=f"Skin/Core Threshold (A11 = {skin_a11_threshold:.2f})"
            )
            
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
        
        # Draw vector field overlay
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
                
        st.image(vec_overlay, width="stretch", caption=f"Vector Field Overlay for {source_title} (Green: Flow Aligned, Amber: Intermediate, Red: Transverse)")
        
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
