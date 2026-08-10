import streamlit as st
import cv2
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os

st.set_page_config(
    page_title="CT Fiber & Layer Thickness Analyzer", 
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
# SIDEBAR CONTROLS
# ---------------------------------------------------------
with st.sidebar:
    st.header(":material/tune: Test & Material Inputs")
    specimen_thickness_mm = st.number_input(
        "Specimen Part Thickness (mm)", 
        min_value=0.1, 
        max_value=20.0, 
        value=2.0, 
        step=0.1, 
        help="Physical thickness of molded specimen to convert pixel measurements into millimeters"
    )
    
    st.markdown("**Micromechanics Inputs (Halpin-Tsai / FEA):**")
    e_fiber_gpa = st.number_input("Fiber Modulus Ef (GPa)", min_value=1.0, max_value=500.0, value=72.0, step=1.0, help="E-glass fibers ~72 GPa")
    e_matrix_gpa = st.number_input("Matrix Modulus Em (GPa)", min_value=0.1, max_value=50.0, value=3.0, step=0.5, help="Polyamide/Polypropylene ~3.0 GPa")
    
    st.divider()
    with st.expander(":material/menu_book: Quick Reference & Guide", expanded=False):
        st.markdown(
            "**Cross-Section Layer Model:**\n"
            "1. **Non-Part (Top):** Smooth dark grey (background/air/mount; 0mm if cropped).\n"
            "2. **Top Skin:** Dark grainy section (flow-aligned fibers).\n"
            "3. **Core:** Thin lighter section centered in sample (transverse fibers).\n"
            "4. **Bottom Skin:** Dark grainy section (flow-aligned fibers).\n"
            "5. **Non-Part (Bottom):** Smooth dark grey (background/air/mount; 0mm if cropped).\n\n"
            "**Fiber Volume Fraction ($V_f$):**\n"
            "Directly estimated from imagery: fibers = white lines & dots; matrix = dark background. Typical short-glass resins are 30–40% glass filled."
        )


# ---------------------------------------------------------
# CORE IMAGE ANALYSIS & RECOGNITION PIPELINE
# ---------------------------------------------------------
def detect_5zone_layers(img_gray):
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
    
    # 3. Detect Thin Lighter Core Centered in Sample
    search_radius = max(int(0.20 * part_h), 10)
    s_min = max(0, center_part - search_radius)
    s_max = min(h, center_part + search_radius)
    
    core_search = row_means[s_min:s_max]
    if len(core_search) > 0:
        core_peak_y = s_min + int(np.argmax(core_search))
        peak_val = row_means[core_peak_y]
        
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
            
        if y_core_bot - y_core_top < 6:
            y_core_top = max(y_part_top + 1, core_peak_y - 6)
            y_core_bot = min(y_part_bot - 1, core_peak_y + 6)
    else:
        y_core_top = center_part - 5
        y_core_bot = center_part + 5
        
    y_part_top = max(0, min(y_part_top, h - 4))
    y_core_top = max(y_part_top + 1, min(y_core_top, h - 3))
    y_core_bot = max(y_core_top + 1, min(y_core_bot, h - 2))
    y_part_bot = max(y_core_bot + 1, min(y_part_bot, h - 1))
    
    return y_part_top, y_core_top, y_core_bot, y_part_bot


def analyze_sample(img_gray, y_part_top, y_core_top, y_core_bot, y_part_bot, total_thick_mm=2.0, blur_ksize=21):
    h, w = img_gray.shape
    
    # 1. Structure tensor for fiber orientation mapping
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
    
    # 2. Direct Fiber Volume Fraction (Vf) Estimation (Adaptive local thresholding)
    part_img = img_gray[y_part_top:y_part_bot+1, :]
    fiber_mask = cv2.adaptiveThreshold(part_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 0)
    global_fiber_pct = float(np.mean(fiber_mask > 0) * 100.0)
    
    # Full image fiber mask for visualization
    full_fiber_mask = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 0)
    full_fiber_mask[0:y_part_top, :] = 0
    full_fiber_mask[y_part_bot+1:h, :] = 0
    
    # 3. Layer Dimensions
    part_h_px = max(y_part_bot - y_part_top + 1, 1)
    top_nonpart_px = y_part_top
    top_skin_px = y_core_top - y_part_top
    core_px = y_core_bot - y_core_top
    bot_skin_px = y_part_bot - y_core_bot
    bot_nonpart_px = h - 1 - y_part_bot
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
    
    # Global orientation & Predicted Tensile Modulus for FEA
    global_a11 = float(np.mean(a11_map[y_part_top:y_part_bot+1, :]))
    global_a22 = 1.0 - global_a11
    
    v_f = global_fiber_pct / 100.0
    e_predicted = e_matrix_gpa + global_a11 * v_f * (e_fiber_gpa - e_matrix_gpa)
    
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
        
        slice_fib = full_fiber_mask[ys:ye, :]
        slice_a11 = a11_map[ys:ye, :]
        slice_a22 = a22_map[ys:ye, :]
        
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
    
    y_pt = res['y_part_top']
    y_ct = res['y_core_top']
    y_cb = res['y_core_bot']
    y_pb = res['y_part_bot']
    
    if y_pt > 0:
        color_mask[0:y_pt, :] = [35, 35, 35]
    if y_ct > y_pt:
        color_mask[y_pt:y_ct, :] = [0, 200, 80]
    if y_cb > y_ct:
        color_mask[y_ct:y_cb, :] = [0, 180, 255]
    if y_pb > y_cb:
        color_mask[y_cb:y_pb, :] = [0, 200, 80]
    if y_pb < h - 1:
        color_mask[y_pb:h, :] = [35, 35, 35]
        
    cv2.addWeighted(color_mask, alpha, overlay, 1 - alpha, 0, overlay)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    
    if y_pt > 0:
        cv2.line(overlay, (0, y_pt), (w, y_pt), (200, 200, 200), 2)
        cv2.putText(overlay, f"Top Surface (y={y_pt}px)", (15, max(y_pt - 8, 20)), font, font_scale, (200, 200, 200), 2)
    if y_ct > y_pt:
        cv2.line(overlay, (0, y_ct), (w, y_ct), (0, 255, 255), 2)
        cv2.putText(overlay, f"Top Skin / Core (y={y_ct}px | Top Skin = {res['top_skin_mm']:.2f}mm / {res['top_skin_pct']:.1f}%)", (15, y_ct - 8), font, font_scale, (0, 255, 255), 2)
    if y_cb < y_pb:
        cv2.line(overlay, (0, y_cb), (w, y_cb), (0, 255, 255), 2)
        cv2.putText(overlay, f"Core / Bot Skin (y={y_cb}px | Core = {res['core_mm']:.2f}mm / {res['core_pct']:.1f}%)", (15, y_cb + 20), font, font_scale, (0, 255, 255), 2)
    if y_pb < h - 1:
        cv2.line(overlay, (0, y_pb), (w, y_pb), (200, 200, 200), 2)
        cv2.putText(overlay, f"Bottom Surface (y={y_pb}px | Bot Skin = {res['bot_skin_mm']:.2f}mm / {res['bot_skin_pct']:.1f}%)", (15, min(y_pb + 20, h - 5)), font, font_scale, (200, 200, 200), 2)
        
    return overlay


# ---------------------------------------------------------
# MAIN UPLOAD & SPECIMEN SELECTION AREA (Clean & Prominent)
# ---------------------------------------------------------
st.title(":material/layers: CT Scan Fiber & Layer Thickness Analyzer")
st.caption("Automated skin-effect thickness measurement, core detection, and direct fiber volume fraction ($V_f$) estimation.")

# Clean prominent upload container
with st.container(border=True):
    up_col1, up_col2 = st.columns([3, 2])
    with up_col1:
        uploaded_file = st.file_uploader(
            "Upload CT Scan Image (PNG, JPG, TIFF)",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            help="Drop your 2D cross-section micro-CT scan image here"
        )
    with up_col2:
        preset_selection = st.selectbox(
            "Or Choose a Demo Specimen Scan:",
            ["(None - Use Uploaded File)"] + list(PRESET_DICT.keys()),
            index=1 if uploaded_file is None else 0
        )

selected_image_gray = None
source_title = ""

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    selected_image_gray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    source_title = uploaded_file.name
elif preset_selection != "(None - Use Uploaded File)":
    preset_path = PRESET_DICT[preset_selection]
    if os.path.exists(preset_path):
        img_bgr = cv2.imread(preset_path)
        selected_image_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        source_title = preset_selection

# ---------------------------------------------------------
# RUN ANALYSIS & DISPLAY DASHBOARD
# ---------------------------------------------------------
if selected_image_gray is not None:
    h, w = selected_image_gray.shape
    
    # Auto-detect boundaries
    auto_pt, auto_ct, auto_cb, auto_pb = detect_5zone_layers(selected_image_gray)
    
    # Optional fine-tuning expander
    with st.expander(":material/tune: Fine-Tune Boundaries (Optional Adjustment)", expanded=False):
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
        sel_pt, 
        sel_ct, 
        sel_cb, 
        sel_pb, 
        total_thick_mm=specimen_thickness_mm,
        blur_ksize=blur_kernel_size
    )
    
    # KPI Metric Cards
    kpi_c1, kpi_c2, kpi_c3, kpi_c4, kpi_c5 = st.columns(5)
    with kpi_c1:
        st.metric(
            "Fiber Vol Fraction (Vf)", 
            f"{res['global_fiber_pct']:.1f}%", 
            help="Estimated directly from imagery (glass fibers vs resin matrix; nominal 30-40% range)",
            border=True
        )
    with kpi_c2:
        st.metric(
            "Total Skin Thickness", 
            f"{res['total_skin_mm']:.2f} mm", 
            f"{res['total_skin_pct']:.1f}% of sample", 
            help=f"Top Skin: {res['top_skin_mm']:.2f}mm ({res['top_skin_pct']:.1f}%) | Bot Skin: {res['bot_skin_mm']:.2f}mm ({res['bot_skin_pct']:.1f}%)",
            border=True
        )
    with kpi_c3:
        st.metric(
            "Core Layer Thickness", 
            f"{res['core_mm']:.2f} mm", 
            f"{res['core_pct']:.1f}% of sample", 
            help=f"Thin lighter section centered in sample ({res['core_px']} pixels)",
            border=True
        )
    with kpi_c4:
        st.metric(
            "Skin-to-Core Ratio", 
            f"{res['total_skin_pct']:.1f}% / {res['core_pct']:.1f}%", 
            f"{res['total_skin_px']}px Skin / {res['core_px']}px Core", 
            help="Volumetric ratio of outer skin layers to central core layer",
            border=True
        )
    with kpi_c5:
        st.metric(
            "Predicted Tensile Modulus", 
            f"{res['e_predicted']:.2f} GPa", 
            f"A11 = {res['global_a11']:.3f}", 
            help="Halpin-Tsai / Rule of Mixtures predicted stiffness for FEA tensile correlation",
            border=True
        )
        
    st.write("")
    
    # ---------------------------------------------------------
    # MAIN TABS (Focused on Analysis & Output)
    # ---------------------------------------------------------
    tab_layers, tab_fiber, tab_orientation, tab_guide = st.tabs([
        ":material/straighten: Layer Thickness & Segmentation",
        ":material/percent: Fiber Volume Fraction (Vf)",
        ":material/show_chart: Orientation Profile & FEA Data",
        ":material/menu_book: Reference Guide & Background"
    ])
    
    # TAB 1: LAYER THICKNESS & SEGMENTATION
    with tab_layers:
        st.subheader("Physical Layer Thickness Measurement & Segmentation")
        st.caption(f"Specimen: `{source_title}` | Part Thickness: {res['part_h_px']} px ({specimen_thickness_mm:.2f} mm)")
        
        overlay_img = draw_segmentation_overlay(res['image_gray'], res)
        st.image(overlay_img, width="stretch", caption="5-Zone Physical Layer Overlay (Green = Skin, Yellow/Amber = Core, Dark Gray = Non-Part)")
        
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
                    {"Zone": "TOTAL PHYSICAL PART", "Thickness (px)": res['part_h_px'], "Thickness (mm)": f"{specimen_thickness_mm:.3f}", "% of Physical Part": "100.0%"}
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
            f"(Typically between 30%–40% glass filled for standard injection molded resins)."
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

    # TAB 3: ORIENTATION PROFILE & FEA DATA
    with tab_orientation:
        st.subheader("Through-Thickness Orientation Profile & FEA Calibration")
        st.markdown("Standardizes input parameters for finite element modeling and tensile test correlation.")
        
        with st.container(border=True):
            fig_prof = go.Figure()
            fig_prof.add_trace(go.Scatter(
                x=res['df_profile']['Normalized_Z'],
                y=res['df_profile']['A11'],
                mode='lines+markers',
                name='A11 (Parallel to Flow / Tensile Axis)',
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
            
        st.markdown("### Export Metrics for FEA Material Cards")
        st.dataframe(res['df_profile'], width="stretch", hide_index=True)
        
        csv_export = res['df_profile'].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Tensor Profile CSV for FEA",
            data=csv_export,
            file_name=f"ct_tensor_profile_{source_title.replace(' ', '_')}.csv",
            mime='text/csv'
        )

    # TAB 4: REFERENCE & GUIDE (Side reference)
    with tab_guide:
        st.subheader("Microstructure & Cross-Section Reference Guide")
        st.info(
            "**FEA Context:** In injection molded parts, mold wall shear aligns fibers parallel to the flow (creating the skin layers), "
            "while slower center flow leaves fibers transverse (creating the core). Capturing this fiber variance is critical for FEA accuracy."
        )
        st.markdown(
            "**Layer Definitions in CT Scan:**\n"
            "- **Top & Bottom Non-Part:** Smooth dark grey background (air/mounting void).\n"
            "- **Top & Bottom Skin:** Dark grainy section containing aligned fibers.\n"
            "- **Center Core:** Thin lighter section centered in the sample containing transverse fibers.\n"
            "- **Fiber Volume Fraction ($V_f$):** Directly extracted by segmenting high-density fiber lines and dots from the darker polymer matrix."
        )
else:
    st.info("Please upload a CT scan image or choose a demo specimen from the dropdown above to begin analysis.")

st.markdown("<br><br><p style='text-align: center; font-size: 11px; color: gray;'>CT Fiber Orientation & Layer Thickness Analyzer | AA/MSE Product Design</p>", unsafe_allow_html=True)
