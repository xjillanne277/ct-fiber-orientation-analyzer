import streamlit as st
import cv2
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import io
import os
import zipfile
from PIL import Image, ImageSequence

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
st.markdown("Research tool for estimating fiber percentage, measuring layer thicknesses (skin vs. core effect), and determining orientation tensors to improve Finite Element Analysis (FEA) accuracy.")

# ---------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------
with st.sidebar:
    st.header(":material/settings: Material & Physical Parameters")
    specimen_thickness_mm = st.number_input("Specimen Thickness (mm)", min_value=0.1, max_value=20.0, value=2.0, step=0.1, help="Total physical thickness of specimen to convert pixels into mm")
    
    st.markdown("**Micromechanics Inputs (Halpin-Tsai / Rule of Mixtures):**")
    e_fiber_gpa = st.number_input("Fiber Modulus Ef (GPa)", min_value=1.0, max_value=500.0, value=72.0, step=1.0, help="E-glass fibers ~72 GPa")
    e_matrix_gpa = st.number_input("Matrix Modulus Em (GPa)", min_value=0.1, max_value=50.0, value=3.0, step=0.5, help="Polyamide/Polypropylene ~3.0 GPa")
    
    st.divider()
    st.header(":material/tune: Thickness & Tensor Parameters")
    num_thickness_layers = st.slider("Through-Thickness Layers", min_value=10, max_value=80, value=40, step=5, help="Number of horizontal bands across height to evaluate through-thickness gradient")
    skin_a11_threshold = st.slider("Skin/Core A11 Threshold", min_value=0.30, max_value=0.70, value=0.50, step=0.02, help="A11 >= threshold classified as Skin (aligned with flow); A11 < threshold classified as Core")
    blur_kernel_size = st.slider("Gaussian Blur Kernel Size", min_value=3, max_value=41, step=2, value=21, help="Smoothing window for Structure Tensor computation")
    fiber_threshold = st.slider("Fiber Intensity Threshold", min_value=10, max_value=200, value=55, help="Grayscale threshold to isolate high-density fibers from matrix background")


# ---------------------------------------------------------
# IMAGE PROCESSING FUNCTIONS
# ---------------------------------------------------------
def compute_structure_tensor(img_gray, blur_ksize=21):
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
    
    return a11_map, a22_map, theta_deg


def analyze_cross_section_image(img_gray, num_layers=40, blur_ksize=21, fiber_thresh=55, skin_thresh=0.5, total_thick_mm=2.0):
    h, w = img_gray.shape
    
    # Structure tensor maps
    a11_map, a22_map, theta_deg = compute_structure_tensor(img_gray, blur_ksize)
    
    layer_h = h / num_layers
    layers_data = []
    
    for i in range(num_layers):
        y0 = int(round(i * layer_h))
        y1 = int(round((i + 1) * layer_h))
        if y1 <= y0:
            y1 = y0 + 1
            
        sub_img = img_gray[y0:y1, :]
        sub_a11 = a11_map[y0:y1, :]
        sub_a22 = a22_map[y0:y1, :]
        
        mask = sub_img > fiber_thresh
        fib_pct = float(np.mean(mask) * 100.0)
        
        if mask.sum() > 0:
            a11_v = float(np.mean(sub_a11[mask]))
            a22_v = float(np.mean(sub_a22[mask]))
        else:
            a11_v = float(np.mean(sub_a11))
            a22_v = float(np.mean(sub_a22))
            
        # z_norm: +1.0 at top mold surface (y=0), 0.0 at center, -1.0 at bottom mold surface (y=h)
        z_norm = 1.0 - 2.0 * (i + 0.5) / num_layers
        depth_mm = (i + 0.5) * (total_thick_mm / num_layers)
        is_skin = (a11_v >= skin_thresh)
        
        layers_data.append({
            'Layer': i + 1,
            'Y_Start': y0,
            'Y_End': y1,
            'Normalized_Z': z_norm,
            'Depth_mm': depth_mm,
            'A11': np.clip(a11_v, 0.0, 1.0),
            'A22': np.clip(a22_v, 0.0, 1.0),
            'Fiber_Pct': fib_pct,
            'Region': 'Skin (Flow Aligned)' if is_skin else 'Core (Transverse)'
        })
        
    df = pd.DataFrame(layers_data)
    
    # Compute Thickness metrics
    num_skin = int((df['Region'] == 'Skin (Flow Aligned)').sum())
    num_core = num_layers - num_skin
    
    skin_pct = (num_skin / num_layers) * 100.0
    core_pct = (num_core / num_layers) * 100.0
    
    skin_mm = (skin_pct / 100.0) * total_thick_mm
    core_mm = (core_pct / 100.0) * total_thick_mm
    
    global_a11 = float(df['A11'].mean())
    global_a22 = 1.0 - global_a11
    global_fiber_pct = float(df['Fiber_Pct'].mean())
    
    v_f = global_fiber_pct / 100.0
    e_predicted = e_matrix_gpa + global_a11 * v_f * (e_fiber_gpa - e_matrix_gpa)
    
    return {
        'df': df,
        'a11_map': a11_map,
        'a22_map': a22_map,
        'theta_deg': theta_deg,
        'num_skin': num_skin,
        'num_core': num_core,
        'skin_pct': skin_pct,
        'core_pct': core_pct,
        'skin_mm': skin_mm,
        'core_mm': core_mm,
        'global_a11': global_a11,
        'global_a22': global_a22,
        'global_fiber_pct': global_fiber_pct,
        'e_predicted': e_predicted,
        'image_gray': img_gray,
        'height': h,
        'width': w
    }


def draw_layer_segmentation_overlay(img_gray, df, alpha=0.35):
    h, w = img_gray.shape
    overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    color_mask = overlay.copy()
    
    for _, row in df.iterrows():
        y0 = int(row['Y_Start'])
        y1 = int(row['Y_End'])
        if row['Region'] == 'Skin (Flow Aligned)':
            # Green for skin
            color_mask[y0:y1, :] = [0, 220, 100]
        else:
            # Red/amber for core
            color_mask[y0:y1, :] = [40, 50, 230]
            
    cv2.addWeighted(color_mask, alpha, overlay, 1 - alpha, 0, overlay)
    
    # Draw boundary transition lines
    for i in range(len(df) - 1):
        if df.iloc[i]['Region'] != df.iloc[i+1]['Region']:
            y_trans = int(df.iloc[i]['Y_End'])
            cv2.line(overlay, (0, y_trans), (w, y_trans), (255, 255, 0), 2)
            cv2.putText(overlay, f"Skin/Core Transition (y={y_trans}px)", (15, y_trans - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
    return overlay


def draw_vector_field_overlay(img_gray, theta_deg, a11_map, grid_step=24, line_len=18):
    h, w = img_gray.shape
    overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    
    for y in range(grid_step // 2, h, grid_step):
        for x in range(grid_step // 2, w, grid_step):
            ang = theta_deg[y, x]
            a11_val = a11_map[y, x]
            
            rad = np.radians(ang)
            if a11_val >= 0.65:
                color = (0, 230, 100)   # Green (aligned with flow)
            elif a11_val >= 0.45:
                color = (0, 180, 255)   # Amber (intermediate)
            else:
                color = (30, 40, 240)   # Red (transverse)
                
            dx = int(line_len * np.cos(rad))
            dy = int(line_len * np.sin(rad))
            
            pt1 = (x - dx, y + dy)
            pt2 = (x + dx, y - dy)
            cv2.arrowedLine(overlay, pt1, pt2, color, 1, tipLength=0.3)
            
    return overlay


# ---------------------------------------------------------
# INPUT SOURCE SELECTION
# ---------------------------------------------------------
col_mode, col_sel = st.columns([1, 2])
with col_mode:
    input_mode = st.radio(
        "Select Input Source:",
        ["📁 Preset CT Scan Images (Folder)", "📤 Upload Custom Image / CT Scan", "🧪 Synthetic 3D Demo"],
        index=0
    )

selected_image_gray = None
source_title = ""

if input_mode == "📁 Preset CT Scan Images (Folder)":
    with col_sel:
        preset_choice = st.selectbox("Choose CT Scan Specimen Image:", list(PRESET_IMAGES.keys()), index=0)
        preset_path = PRESET_IMAGES[preset_choice]
        if os.path.exists(preset_path):
            img_bgr = cv2.imread(preset_path)
            selected_image_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            source_title = preset_choice
        else:
            st.error(f"Image not found at path: {preset_path}")

elif input_mode == "📤 Upload Custom Image / CT Scan":
    with col_sel:
        uploaded_file = st.file_uploader("Upload 2D Cross-Section Image (PNG, JPG, TIFF) or 3D Stack:", type=["png", "jpg", "jpeg", "tif", "tiff", "zip"])
        if uploaded_file is not None:
            fname = uploaded_file.name.lower()
            if fname.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                selected_image_gray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
                source_title = uploaded_file.name

elif input_mode == "🧪 Synthetic 3D Demo":
    with col_sel:
        st.info("Synthetic 3D Stack simulating Skin-Core-Skin fiber orientation across 40 layers.")
        # Generate synthetic 2D cross-section
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
        source_title = "Synthetic 2D Cross-Section (Skin-Core Simulation)"

# Run Analysis
res = None
if selected_image_gray is not None:
    res = analyze_cross_section_image(
        selected_image_gray,
        num_layers=num_thickness_layers,
        blur_ksize=blur_kernel_size,
        fiber_thresh=fiber_threshold,
        skin_thresh=skin_a11_threshold,
        total_thick_mm=specimen_thickness_mm
    )

# ---------------------------------------------------------
# TOP KPI METRIC ROW
# ---------------------------------------------------------
if res is not None:
    st.divider()
    kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
    with kpi_c1:
        st.metric(
            "Estimated Fiber % (Vf)", 
            f"{res['global_fiber_pct']:.1f}%", 
            help="Volume-averaged fiber percentage extracted from grayscale intensity thresholding", 
            border=True
        )
    with kpi_c2:
        st.metric(
            "Skin-to-Core Ratio", 
            f"{res['skin_pct']:.1f}% Skin / {res['core_pct']:.1f}% Core", 
            help=f"Skin Thickness: {res['skin_mm']:.2f} mm | Core Thickness: {res['core_mm']:.2f} mm", 
            border=True
        )
    with kpi_c3:
        st.metric(
            "Global Tensor A11 (Orientation)", 
            f"{res['global_a11']:.3f}", 
            help="1.0 = Perfect longitudinal alignment (flow direction), 0.0 = Pure transverse orientation", 
            border=True
        )
    with kpi_c4:
        st.metric(
            "Predicted Tensile Modulus", 
            f"{res['e_predicted']:.2f} GPa", 
            help="Halpin-Tsai / Rule of Mixtures stiffness prediction for FEA", 
            border=True
        )

st.write("")

# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
tab_bg, tab_thickness, tab_profile, tab_vectors, tab_data = st.tabs([
    ":material/menu_book: Background",
    ":material/straighten: Layer Thickness & Skin Effect",
    ":material/show_chart: Through-Thickness Profile",
    ":material/explore: Visual Overlays & Vector Field",
    ":material/table_chart: Data & FEA Export"
])

# ---------------------------------------------------------
# TAB 1: BACKGROUND & METHODOLOGY
# ---------------------------------------------------------
with tab_bg:
    st.subheader("Microstructure Anisotropy & CT Scan Cross-Section Guide")
    
    # FEA Context Setting Callout
    st.info(
        "**Finite Element Analysis (FEA) Simulation Context:**\n\n"
        "In injection-molded short-glass-fiber reinforced polymers (SFRTPs), high shear rates along the mold walls freeze fibers parallel to the flow direction (forming the **Skin Layer**), while slower extensional flow at the center causes fibers to orient transversely or randomly (forming the **Core Layer**).\n\n"
        "Capturing this through-thickness **fiber variance** is critical for establishing accurate anisotropic material cards in FEA simulations (such as Abaqus, ANSYS, and Moldflow). Assuming uniform isotropic stiffness leads to significant errors in stiffness prediction, stress concentration calculations, and failure analysis."
    )
    
    st.markdown("### 🔬 How to Input Cross-Sections for Accurate Orientation")
    st.markdown(
        "To determine true fiber orientation and measure layer thicknesses, **users must input cross-sections that include both the core and the skin effect.**"
    )
    
    with st.container(border=True):
        st.markdown(
            "#### ⚠️ Why Analyzing Only the Top Layer is Insufficient\n\n"
            "- **Top-Layer Only (XY Plane):** Analyzing only a top-surface planar slice captures *only* the skin layer where fibers are frozen parallel to the mold wall. This produces an artificially high stiffness estimate and completely misses the transverse core.\n\n"
            "- **Full Cross-Section (XZ / YZ Plane):** Analyzing the complete thickness reveals both outer skin layers and the internal core, capturing the true fiber variance necessary for accurate FEA simulations."
        )
        
    st.markdown("### 🖼️ Example Images from CT Scan Dataset")
    st.caption("Visual comparison of valid through-thickness cross-section vs. top-layer surface slice:")
    
    ex_c1, ex_c2 = st.columns(2)
    with ex_c1:
        with st.container(border=True):
            st.markdown("**:material/check_circle: Valid Input: Cross-Section with Core & Skin (XZ Plane)**")
            xz_img_path = PRESET_IMAGES["XZ Plane: Through-Thickness with Core & Skin (black, 90deg, xz plane with core.png)"]
            if os.path.exists(xz_img_path):
                st.image(xz_img_path, width="stretch", caption="Example: XZ Plane Cross-Section showing distinct Skin & Core Layers")
            st.markdown(
                "- **Includes:** Top mold skin layer, center core layer, bottom mold skin layer.\n"
                "- **Result:** Allows measuring individual skin/core thicknesses and true orientation tensors for FEA."
            )
            
    with ex_c2:
        with st.container(border=True):
            st.markdown("**:material/cancel: Insufficient Input: Top Surface Layer Only (XY Plane)**")
            xy_img_path = PRESET_IMAGES["XY Plane: Surface Slice - Only Skin (black, 90deg, xy plane, only skin.png)"]
            if os.path.exists(xy_img_path):
                st.image(xy_img_path, width="stretch", caption="Example: XY Plane Slice showing Only Skin (Surface Layer)")
            st.markdown(
                "- **Limitation:** Captures only the surface skin layer without any through-thickness core context.\n"
                "- **Risk:** Severe sampling bias; over-estimates global longitudinal alignment."
            )

# ---------------------------------------------------------
# TAB 2: LAYER THICKNESS & SKIN EFFECT
# ---------------------------------------------------------
with tab_thickness:
    if res is not None:
        st.subheader("Layer Thickness Measurement & Skin-to-Core Segmentation")
        st.markdown(f"Analyzed specimen **`{source_title}`** across **{num_thickness_layers} thickness layers** (Total Thickness: **{specimen_thickness_mm:.2f} mm** / **{res['height']} pixels**).")
        
        thick_c1, thick_c2, thick_c3 = st.columns(3)
        with thick_c1:
            st.metric("Total Skin Layer Thickness", f"{res['skin_mm']:.2f} mm", f"{res['skin_pct']:.1f}% of total thickness", border=True)
        with thick_c2:
            st.metric("Total Core Layer Thickness", f"{res['core_mm']:.2f} mm", f"{res['core_pct']:.1f}% of total thickness", border=True)
        with thick_c3:
            st.metric("Skin-to-Core Ratio", f"{res['skin_pct']:.1f}% / {res['core_pct']:.1f}%", f"{res['num_skin']} skin / {res['num_core']} core layers", border=True)
            
        st.write("")
        st.markdown("### Visual Skin vs. Core Layer Segmentation on CT Scan Image")
        st.caption("🟢 Green bands = Skin layer (A11 ≥ threshold) | 🔴 Red bands = Core layer (A11 < threshold) | 🟡 Yellow lines = Boundary transitions")
        
        seg_overlay = draw_layer_segmentation_overlay(res['image_gray'], res['df'])
        st.image(seg_overlay, width="stretch", caption=f"Segmentation Overlay for {source_title}")
        
        st.markdown("### Morphological Volume Ratio")
        pie_col1, pie_col2 = st.columns([1, 1])
        with pie_col1:
            with st.container(border=True):
                fig_pie = px.pie(
                    values=[res['skin_pct'], res['core_pct']],
                    names=['Skin Layers (Aligned)', 'Core Layers (Transverse)'],
                    color=['Skin Layers (Aligned)', 'Core Layers (Transverse)'],
                    color_discrete_map={'Skin Layers (Aligned)': '#10B981', 'Core Layers (Transverse)': '#EF4444'},
                    hole=0.4,
                    title="Skin-to-Core Volumetric Distribution"
                )
                st.plotly_chart(fig_pie, width="stretch")
        with pie_col2:
            with st.container(border=True):
                st.markdown("**Layer Thickness Breakdown Summary:**")
                st.write(f"- **Image Dimensions:** `{res['width']} px (W) × {res['height']} px (H)`")
                st.write(f"- **Physical Thickness Scale:** `{(specimen_thickness_mm / res['height']) * 1000:.2f} µm / pixel`")
                st.write(f"- **Skin Thickness:** `{res['skin_mm']:.3f} mm` (`{res['skin_pct']:.1f}%`)")
                st.write(f"- **Core Thickness:** `{res['core_mm']:.3f} mm` (`{res['core_pct']:.1f}%`)")
                st.write(f"- **Estimated Fiber Volume Fraction Vf:** `{res['global_fiber_pct']:.1f}%`")
                st.write(f"- **Global Volume-Averaged Tensor A11:** `{res['global_a11']:.3f}`")
    else:
        st.info("Please select or upload a CT scan image to measure layer thicknesses.")

# ---------------------------------------------------------
# TAB 3: THROUGH-THICKNESS PROFILE
# ---------------------------------------------------------
with tab_profile:
    if res is not None:
        st.subheader("Through-Thickness Orientation ($A_{11}$ vs $A_{22}$) and Fiber % Profiles")
        st.markdown("Quantifies fiber orientation tensor values and fiber density from top surface ($z_{\\text{norm}} = +1.0$) to bottom surface ($z_{\\text{norm}} = -1.0$).")
        
        with st.container(border=True):
            fig_prof = go.Figure()
            
            # A11 trace
            fig_prof.add_trace(go.Scatter(
                x=res['df']['Normalized_Z'],
                y=res['df']['A11'],
                mode='lines+markers',
                name='A11 (Parallel / Flow Axis)',
                line=dict(color='#10B981', width=3),
                marker=dict(size=6)
            ))
            
            # A22 trace
            fig_prof.add_trace(go.Scatter(
                x=res['df']['Normalized_Z'],
                y=res['df']['A22'],
                mode='lines+markers',
                name='A22 (Perpendicular / Transverse)',
                line=dict(color='#EF4444', width=2, dash='dot'),
                marker=dict(size=5)
            ))
            
            # Threshold line
            fig_prof.add_hline(
                y=skin_a11_threshold, 
                line_dash="dash", 
                line_color="#F59E0B", 
                annotation_text=f"Skin/Core Threshold (A11 = {skin_a11_threshold:.2f})"
            )
            
            # Center Core line
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
                res['df'],
                x='Normalized_Z',
                y='Fiber_Pct',
                markers=True,
                title="Through-Thickness Fiber Density / Volume Fraction Profile (%)",
                labels={'Normalized_Z': 'Normalized Thickness z_norm', 'Fiber_Pct': 'Fiber Percentage (%)'}
            )
            fig_fib.update_traces(line_color='#3B82F6', line_width=2.5)
            st.plotly_chart(fig_fib, width="stretch")
    else:
        st.info("Please select or upload a CT scan image to view through-thickness profiles.")

# ---------------------------------------------------------
# TAB 4: VISUAL OVERLAYS & VECTOR FIELD
# ---------------------------------------------------------
with tab_vectors:
    if res is not None:
        st.subheader("Fiber Orientation Vector Field Overlay")
        st.markdown("Visualizes local fiber orientation vectors computed via Structure Tensor gradients directly over the CT scan image.")
        
        vec_step = st.slider("Vector Grid Density", min_value=12, max_value=48, value=24, step=4)
        vec_overlay = draw_vector_field_overlay(res['image_gray'], res['theta_deg'], res['a11_map'], grid_step=vec_step)
        
        st.image(vec_overlay, width="stretch", caption=f"Vector Field Overlay for {source_title} (Green: Flow Aligned, Amber: Intermediate, Red: Transverse)")
        
        v_c1, v_c2 = st.columns(2)
        with v_c1:
            with st.container(border=True):
                st.markdown("**Original Raw CT Grayscale Scan**")
                st.image(res['image_gray'], width="stretch")
        with v_c2:
            with st.container(border=True):
                st.markdown("**Layer Segmentation Overlay (Skin vs Core)**")
                st.image(seg_overlay, width="stretch")
    else:
        st.info("Please select or upload a CT scan image to view vector field overlays.")

# ---------------------------------------------------------
# TAB 5: DATA & FEA EXPORT
# ---------------------------------------------------------
with tab_data:
    if res is not None:
        st.subheader("Orientation Tensor & Layer Thickness Data Table")
        st.markdown("Through-thickness tensor distribution formatted for FEA material calibration (Abaqus, ANSYS, Moldflow).")
        
        st.dataframe(res['df'], width="stretch", hide_index=True)
        
        csv_export = res['df'].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Layer Tensor CSV for FEA",
            data=csv_export,
            file_name=f"ct_orientation_tensor_{source_title.replace(' ', '_')}.csv",
            mime='text/csv'
        )
    else:
        st.info("Please select or upload a CT scan image to view and export data.")

st.markdown("<br><br><p style='text-align: center; font-size: 11px; color: gray;'>CT Fiber Orientation & Layer Thickness Analyzer | AA/MSE Product Design</p>", unsafe_allow_html=True)
