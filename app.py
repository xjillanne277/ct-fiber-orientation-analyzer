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

st.set_page_config(page_title="CT Fiber Orientation Analyzer (Through-Thickness)", page_icon=":material/layers:", layout="wide")

st.title(":material/layers: CT Scan Through-Thickness Fiber Orientation Analyzer")

st.markdown(r"""
Quantitative **Through-Thickness ($Z$-axis)** fiber orientation tensor analysis for injection-molded tensile dogbones and plaques.
Iterates through 3D CT scan slice stacks along the thickness direction, extracts 2D orientation tensors ($A_{11}$ and $A_{22}$), maps the **Skin-Core effect**, and predicts volume-averaged tensile properties.
""")

# Sidebar Controls
with st.sidebar:
    st.header(":material/settings: Test & Material Settings")
    st.caption("Tensile Loading Axis: X-axis (longitudinal)")
    
    st.markdown("**Micromechanics Inputs (Halpin-Tsai / Rule of Mixtures):**")
    e_fiber_gpa = st.number_input("Fiber Modulus Ef (GPa)", min_value=1.0, max_value=500.0, value=72.0, step=1.0, help="E.g., E-glass fibers ~72 GPa")
    e_matrix_gpa = st.number_input("Matrix Modulus Em (GPa)", min_value=0.1, max_value=50.0, value=3.0, step=0.5, help="E.g., Polyamide/Polypropylene ~3.0 GPa")
    v_fiber_pct = st.slider("Fiber Volume Fraction Vf (%)", min_value=1.0, max_value=60.0, value=20.0, step=1.0)
    v_f = v_fiber_pct / 100.0

    st.divider()
    st.header(":material/tune: Image Processing Parameters")
    blur_kernel_size = st.slider("Gaussian Blur Kernel Size", min_value=3, max_value=31, step=2, value=11)
    intensity_threshold = st.slider("Fiber Intensity Threshold", min_value=0, max_value=255, value=40)
    slice_binning = st.slider("Z-Slice Binning / Downsampling", min_value=1, max_value=10, value=1, help="Bin adjacent slices to speed up processing")


def analyze_slice_orientation_tensor(img, blur_ksize=11, threshold=40):
    """
    Computes 2D Orientation Tensor components (A11, A22, A12) for a single Z-slice.
    A11: Parallel to X-axis (tensile pull direction).
    A22: Perpendicular to X-axis (transverse Y-direction).
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()
        
    gray_smooth = cv2.GaussianBlur(gray, (3, 3), 0)
    _, mask = cv2.threshold(gray_smooth, threshold, 255, cv2.THRESH_BINARY)
    
    # Image gradients using Sobel
    Ix = cv2.Sobel(gray_smooth, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray_smooth, cv2.CV_64F, 0, 1, ksize=3)
    
    # Structure tensor components
    Jxx = cv2.GaussianBlur(Ix**2, (blur_ksize, blur_ksize), 0)
    Jyy = cv2.GaussianBlur(Iy**2, (blur_ksize, blur_ksize), 0)
    Jxy = cv2.GaussianBlur(Ix * Iy, (blur_ksize, blur_ksize), 0)
    
    # Fiber orientation is perpendicular to intensity gradient direction
    # A11 (parallel to X-axis) = Jyy / (Jxx + Jyy)
    denom = Jxx + Jyy + 1e-8
    a11_map = Jyy / denom
    a22_map = Jxx / denom
    
    # Compute orientation angle theta relative to X-axis for vector overlays
    theta_rad = 0.5 * np.arctan2(2 * Jxy, Jyy - Jxx) + (np.pi / 2.0)
    theta_deg = (np.degrees(theta_rad)) % 180.0
    
    valid_mask = mask > 0
    if valid_mask.any():
        a11_val = float(np.mean(a11_map[valid_mask]))
        a22_val = float(np.mean(a22_map[valid_mask]))
    else:
        a11_val = float(np.mean(a11_map))
        a22_val = float(np.mean(a22_map))
        
    return {
        'A11': np.clip(a11_val, 0.0, 1.0),
        'A22': np.clip(a22_val, 0.0, 1.0),
        'A12': float(1.0 - a11_val - a22_val),
        'a11_map': a11_map,
        'theta_deg': theta_deg,
        'mask': mask,
        'image': gray
    }


def draw_slice_vectors(image, theta_deg, grid_size=10, line_len=18):
    overlay = image.copy()
    if len(overlay.shape) == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)
        
    h, w = theta_deg.shape
    cell_h, cell_w = h // grid_size, w // grid_size
    
    for i in range(grid_size):
        for j in range(grid_size):
            cy = i * cell_h + cell_h // 2
            cx = j * cell_w + cell_w // 2
            
            ang = theta_deg[cy, cx]
            # A11 > 0.5 -> Green (parallel), A11 < 0.5 -> Red (perpendicular)
            rad = np.radians(ang)
            a11_local = np.cos(rad)**2
            
            if a11_local >= 0.7:
                color = (0, 230, 118)   # Bright Green
            elif a11_local >= 0.4:
                color = (255, 171, 0)  # Amber
            else:
                color = (255, 23, 68)   # Red
                
            dx = int(line_len * np.cos(rad))
            dy = int(line_len * np.sin(rad))
            
            pt1 = (cx - dx, cy + dy)
            pt2 = (cx + dx, cy - dy)
            cv2.arrowedLine(overlay, pt1, pt2, color, 2, tipLength=0.3)
            
    return overlay


def generate_synthetic_3d_stack(num_slices=40, width=280, height=280):
    """
    Generates a synthetic 3D CT scan stack simulating the Skin-Core-Skin fiber effect
    in an injection-molded tensile dogbone specimen.
    """
    slices = []
    z_norms = [2.0 * z / (num_slices - 1.0) - 1.0 for z in range(num_slices)]
    
    for z_idx, z_norm in enumerate(z_norms):
        img = np.full((height, width), 40, dtype=np.uint8)
        
        # Skin-core angle transition: z_norm near +/-1 -> 0 deg (parallel to X), z_norm near 0 -> 90 deg (transverse)
        target_angle_deg = 90.0 * (1.0 - abs(z_norm)**1.5)
        rad_target = np.radians(target_angle_deg)
        
        num_fibers = 260
        for _ in range(num_fibers):
            cx = np.random.randint(15, width - 15)
            cy = np.random.randint(15, height - 15)
            length = np.random.randint(14, 30)
            
            ang = rad_target + np.radians(np.random.normal(0, 6.0))
            dx = int(length * np.cos(ang))
            dy = int(length * np.sin(ang))
            
            pt1 = (int(cx - dx/2), int(cy - dy/2))
            pt2 = (int(cx + dx/2), int(cy + dy/2))
            cv2.line(img, pt1, pt2, int(np.random.randint(180, 240)), thickness=2)
            
        noise = np.random.normal(0, 6, (height, width)).astype(np.uint8)
        slices.append(cv2.add(img, noise))
        
    return slices


# Ingestion UI
col_ing1, col_ing2 = st.columns([3, 1])
with col_ing1:
    uploaded_files = st.file_uploader(
        "Upload 3D CT Scan Stack (Multi-page TIFF, ZIP of slices, or multiple PNG/JPG/TIFF slices)", 
        type=["tif", "tiff", "zip", "png", "jpg", "jpeg"], 
        accept_multiple_files=True
    )
with col_ing2:
    st.write("")
    st.write("")
    use_synthetic_demo = st.toggle("Use synthetic 3D Skin-Core CT stack", value=True if not uploaded_files else False)

stack_slices = []
stack_source_name = ""

if use_synthetic_demo and not uploaded_files:
    stack_slices = generate_synthetic_3d_stack(num_slices=40)
    stack_source_name = "Synthetic 3D Injection Molded Dogbone Stack (40 Slices)"
    st.success("Loaded synthetic 3D CT stack simulating injection-molded Skin-Core fiber distribution!")
elif uploaded_files:
    # Single Multi-page TIFF or ZIP or Multiple Slice Images
    if len(uploaded_files) == 1:
        f = uploaded_files[0]
        fname = f.name.lower()
        if fname.endswith((".tif", ".tiff")):
            try:
                pil_img = Image.open(f)
                for page in ImageSequence.Iterator(pil_img):
                    stack_slices.append(np.array(page.convert("L")))
                stack_source_name = f.name
            except Exception as e:
                st.error(f"Error reading multi-page TIFF: {e}")
        elif fname.endswith(".zip"):
            try:
                with zipfile.ZipFile(f) as z:
                    for zfile in sorted(z.namelist()):
                        if zfile.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                            with z.open(zfile) as zf:
                                img_b = np.frombuffer(zf.read(), np.uint8)
                                img = cv2.imdecode(img_b, cv2.IMREAD_GRAYSCALE)
                                if img is not None:
                                    stack_slices.append(img)
                stack_source_name = f.name
            except Exception as e:
                st.error(f"Error extracting ZIP archive: {e}")
        else:
            # Single image slice
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                stack_slices.append(img)
                stack_source_name = f.name
    else:
        # Multiple slice files
        sorted_files = sorted(uploaded_files, key=lambda x: x.name)
        for f in sorted_files:
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                stack_slices.append(img)
        stack_source_name = f"{len(stack_slices)} Uploaded Image Slices"

if stack_slices:
    # Downsample / Bin Slices if requested
    if slice_binning > 1:
        binned_slices = stack_slices[::slice_binning]
    else:
        binned_slices = stack_slices
        
    num_z = len(binned_slices)
    
    # Iterate through Z-Axis Slices and Calculate Orientation Tensors
    slice_results = []
    z_norms = [2.0 * z / (num_z - 1.0) - 1.0 for z in range(num_z)] if num_z > 1 else [0.0]
    
    for z_idx, img in enumerate(binned_slices):
        res = analyze_slice_orientation_tensor(img, blur_ksize=blur_kernel_size, threshold=intensity_threshold)
        res['z_idx'] = z_idx
        res['z_norm'] = z_norms[z_idx]
        slice_results.append(res)
        
    # Build DataFrame of Through-Thickness Orientation Tensor
    df_tensor = pd.DataFrame({
        'Z Slice': [r['z_idx'] for r in slice_results],
        'Normalized Thickness': [r['z_norm'] for r in slice_results],
        'A11 (Parallel to Tensile Pull)': [r['A11'] for r in slice_results],
        'A22 (Perpendicular Transverse)': [r['A22'] for r in slice_results]
    })
    
    # 1. Weighted Global Volume Average Tensor A11_global
    global_a11 = float(df_tensor['A11 (Parallel to Tensile Pull)'].mean())
    global_a22 = 1.0 - global_a11
    
    # Predictive Tensile Modulus (Halpin-Tsai estimate)
    e_predicted_gpa = e_matrix_gpa + global_a11 * v_f * (e_fiber_gpa - e_matrix_gpa)
    
    # 2. Inflection Point & Skin-to-Core Ratio Calculation
    skin_mask = df_tensor['A11 (Parallel to Tensile Pull)'] >= 0.5
    num_skin = int(skin_mask.sum())
    num_core = num_z - num_skin
    
    skin_pct = (num_skin / num_z) * 100.0
    core_pct = (num_core / num_z) * 100.0
    ratio_str = f"{skin_pct:.1f}% Skin / {core_pct:.1f}% Core" if core_pct > 0 else "100% Skin"
    
    st.divider()
    
    # KPI Metric Cards Row
    kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
    with kpi_c1:
        st.metric("Total Z Slices Analyzed", f"{num_z} Slices", border=True)
    with kpi_c2:
        st.metric("Weighted Global Tensor A11", f"{global_a11:.3f}", help="Volume-averaged A11 alignment parallel to tensile pull direction", border=True)
    with kpi_c3:
        st.metric("Skin-to-Core Ratio", ratio_str, help="Skin (A11 ≥ 0.5) vs Core (A11 < 0.5) percentage", border=True)
    with kpi_c4:
        st.metric("Predicted Tensile Modulus", f"{e_predicted_gpa:.2f} GPa", help="Micro-mechanical prediction based on A11_global, Ef, Em, and Vf", border=True)
        
    st.write("")
    
    tab_profile, tab_explorer, tab_table, tab_export = st.tabs([
        "📈 Through-Thickness Profile", 
        "🔍 Interactive Slice Explorer", 
        "📊 Tensor Component Data", 
        "💾 Export Metrics"
    ])
    
    # ---------------------------------------------------------
    # TAB 1: THROUGH-THICKNESS PROFILE GRAPH
    # ---------------------------------------------------------
    with tab_profile:
        st.subheader("Through-Thickness Orientation Tensor Profile ($A_{11}$ vs $A_{22}$)")
        st.markdown(r"""
        Line chart displaying the **Skin-Core Effect** along normalized thickness $z_{\text{norm}} \in [-1.0, 1.0]$. 
        - $A_{11} > 0.5$: **Skin Region** (Fibers aligned parallel to X-axis tensile loading direction).
        - $A_{11} < 0.5$: **Core Region** (Fibers aligned perpendicular to X-axis in transverse direction).
        """)
        
        with st.container(border=True):
            fig_profile = go.Figure()
            
            # A11 (Parallel) Line
            fig_profile.add_trace(go.Scatter(
                x=df_tensor['Normalized Thickness'],
                y=df_tensor['A11 (Parallel to Tensile Pull)'],
                mode='lines+markers',
                name='A11 (Parallel to X-Axis / Tensile Pull)',
                line=dict(color='#10B981', width=3),
                marker=dict(size=6)
            ))
            
            # A22 (Perpendicular) Line
            fig_profile.add_trace(go.Scatter(
                x=df_tensor['Normalized Thickness'],
                y=df_tensor['A22 (Perpendicular Transverse)'],
                mode='lines+markers',
                name='A22 (Perpendicular to X-Axis / Transverse Y)',
                line=dict(color='#EF4444', width=2, dash='dot'),
                marker=dict(size=5)
            ))
            
            # A11 = 0.5 Skin/Core Threshold Line
            fig_profile.add_hline(
                y=0.5, line_dash="dash", line_color="#F59E0B", 
                annotation_text="Skin-Core Transition Threshold (A11 = 0.5)", annotation_position="top right"
            )
            
            # Dead Center Core Annotation (z_norm = 0)
            fig_profile.add_vline(x=0.0, line_dash="solid", line_color="gray", opacity=0.4, annotation_text="Center Core (z=0)")
            
            fig_profile.update_layout(
                title=f"Through-Thickness Fiber Orientation Profile — {stack_source_name}",
                xaxis_title="Normalized Thickness (z / (h/2)) [ -1.0 = Bottom Mold Skin, 0.0 = Core, +1.0 = Top Mold Skin ]",
                yaxis_title="Orientation Tensor Component Value (0.0 to 1.0)",
                yaxis=dict(range=[0.0, 1.0]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig_profile, width="stretch")
            
        st.markdown("### Skin-to-Core Morphological Breakdown")
        break_col1, break_col2 = st.columns(2)
        with break_col1:
            with st.container(border=True):
                st.markdown("**Skin-to-Core Volume Ratio Pie Chart**")
                fig_pie = px.pie(
                    values=[skin_pct, core_pct], 
                    names=['Skin Region (A11 ≥ 0.5)', 'Core Region (A11 < 0.5)'],
                    color=['Skin Region (A11 ≥ 0.5)', 'Core Region (A11 < 0.5)'],
                    color_discrete_map={'Skin Region (A11 ≥ 0.5)': '#10B981', 'Core Region (A11 < 0.5)': '#EF4444'},
                    hole=0.4
                )
                st.plotly_chart(fig_pie, width="stretch")
                
        with break_col2:
            with st.container(border=True):
                st.markdown("**Inflection Points & Transition Metrics**")
                st.write(f"- **Total Slices**: `{num_z}`")
                st.write(f"- **Skin Layer Thickness**: `{skin_pct:.1f}%` ({num_skin} slices)")
                st.write(f"- **Core Layer Thickness**: `{core_pct:.1f}%` ({num_core} slices)")
                st.write(f"- **Skin-to-Core Ratio**: `{ratio_str}`")
                st.write(f"- **Volume-Averaged Tensor $A_{{11,\\text{{global}}}}$**: `{global_a11:.3f}`")
                st.write(f"- **Estimated Tensile Modulus $E_{{\\text{{tensile}}}}$**: `{e_predicted_gpa:.2f} GPa`")

    # ---------------------------------------------------------
    # TAB 2: INTERACTIVE SLICE EXPLORER
    # ---------------------------------------------------------
    with tab_explorer:
        st.subheader("Interactive Through-Thickness Slice Explorer")
        st.markdown("Scrub through depth slices along the Z-axis to inspect local fiber structure tensor vector overlays and slice metrics.")
        
        sel_z_idx = st.slider("Select Depth Slice Index Z", min_value=0, max_value=num_z - 1, value=num_z // 2)
        target_res = slice_results[sel_z_idx]
        
        vec_overlay = draw_slice_vectors(target_res['image'], target_res['theta_deg'], grid_size=12)
        
        ex_col1, ex_col2 = st.columns(2)
        with ex_col1:
            with st.container(border=True):
                st.markdown(f"**Original CT Slice Z={sel_z_idx}** (z_norm = {target_res['z_norm']:.2f})")
                st.image(target_res['image'], width="stretch")
                
        with ex_col2:
            with st.container(border=True):
                st.markdown(f"**Vector Tensor Overlay Z={sel_z_idx}** (Green: Parallel $A_{{11}}\\ge 0.7$, Amber: $0.4\\le A_{{11}}<0.7$, Red: Perpendicular $A_{{11}}<0.4$)")
                st.image(vec_overlay, width="stretch")
                
        # Slice KPI Breakdown
        sl_m1, sl_m2, sl_m3 = st.columns(3)
        with sl_m1:
            st.metric("Slice A11 (Parallel)", f"{target_res['A11']:.3f}", border=True)
        with sl_m2:
            st.metric("Slice A22 (Perpendicular)", f"{target_res['A22']:.3f}", border=True)
        with sl_m3:
            region_label = "Skin Region (Parallel Flow)" if target_res['A11'] >= 0.5 else "Core Region (Transverse Flow)"
            st.metric("Layer Morphology", region_label, border=True)

    # ---------------------------------------------------------
    # TAB 3: TENSOR COMPONENT DATA TABLE
    # ---------------------------------------------------------
    with tab_table:
        st.subheader("Through-Thickness Orientation Tensor Data Table")
        st.dataframe(df_tensor, width="stretch", hide_index=True)

    # ---------------------------------------------------------
    # TAB 4: EXPORT METRICS
    # ---------------------------------------------------------
    with tab_export:
        st.subheader("Export Orientation Tensor Profile CSV")
        st.markdown("Download full through-thickness $A_{11}(z)$ and $A_{22}(z)$ orientation tensor components formatted for FEA modeling (e.g. Abaqus, ANSYS, Moldflow).")
        
        csv_export = df_tensor.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Through-Thickness Tensor CSV",
            data=csv_export,
            file_name='ct_through_thickness_fiber_orientation_tensor.csv',
            mime='text/csv',
        )

else:
    st.info("Please upload a 3D CT scan stack or toggle the synthetic 3D stack from the sidebar to begin through-thickness analysis.")

st.markdown("<br><br><p style='text-align: center; font-size: 11px; color: gray;'>Created by Product Design Engineering Intern, Advanced Architecture</p>", unsafe_allow_html=True)
