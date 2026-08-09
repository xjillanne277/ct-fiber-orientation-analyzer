import streamlit as st
import cv2
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
from PIL import Image

st.set_page_config(page_title="CT Fiber Orientation Analyzer", page_icon=":material/center_focus_strong:", layout="wide")

st.title(":material/center_focus_strong: CT Fiber Orientation Analyzer")

st.markdown("""
Analyze short-glass-fiber orientation from **CT scan screenshots & image slices**. 
Computes local structure tensor matrices ($J_{xx}, J_{yy}, J_{xy}$), generates directional vector overlays, calculates circular angular statistics, and compares orientation angles against nominal flow directions ($0^\circ, 45^\circ, 90^\circ$).
""")

# Sidebar Controls
with st.sidebar:
    st.header(":material/tune: Analysis Controls")
    
    nominal_orientation_str = st.selectbox("Expected Nominal Orientation", ["0° (Parallel)", "45° (Transverse)", "90° (Perpendicular)", "Custom Angle"])
    if nominal_orientation_str == "Custom Angle":
        nominal_angle = st.slider("Custom Nominal Angle (°)", min_value=0.0, max_value=180.0, value=0.0, step=1.0)
    else:
        nominal_angle = float(nominal_orientation_str.split("°")[0])
        
    st.divider()
    st.markdown("**Image Processing Parameters:**")
    grid_size = st.slider("Grid Resolution (Cells)", min_value=3, max_value=15, value=6)
    blur_kernel_size = st.slider("Gaussian Blur Kernel Size", min_value=3, max_value=31, step=2, value=11)
    intensity_threshold = st.slider("Fiber Intensity Threshold", min_value=0, max_value=255, value=40)
    arrow_length = st.slider("Overlay Vector Arrow Length", min_value=10, max_value=50, value=22)


def analyze_fiber_orientation(image, grid_size, threshold, blur_ksize):
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
        
    # Thresholding to isolate fibers
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    gray_filtered = cv2.bitwise_and(gray, gray, mask=mask)
        
    # Image gradients using Sobel
    Ix = cv2.Sobel(gray_filtered, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray_filtered, cv2.CV_64F, 0, 1, ksize=3)
    
    # Structure tensor components
    Jxx = Ix**2
    Jyy = Iy**2
    Jxy = Ix * Iy
    
    # Gaussian smoothing on tensor components
    Jxx_smooth = cv2.GaussianBlur(Jxx, (blur_ksize, blur_ksize), 0)
    Jyy_smooth = cv2.GaussianBlur(Jyy, (blur_ksize, blur_ksize), 0)
    Jxy_smooth = cv2.GaussianBlur(Jxy, (blur_ksize, blur_ksize), 0)
    
    h, w = gray.shape
    cell_h = h // grid_size
    cell_w = w // grid_size
    
    angles = []
    centers = []
    grid_indices = []
    
    for i in range(grid_size):
        for j in range(grid_size):
            y_start, y_end = i * cell_h, (i + 1) * cell_h
            x_start, x_end = j * cell_w, (j + 1) * cell_w
            
            sum_Jxx = np.sum(Jxx_smooth[y_start:y_end, x_start:x_end])
            sum_Jyy = np.sum(Jyy_smooth[y_start:y_end, x_start:x_end])
            sum_Jxy = np.sum(Jxy_smooth[y_start:y_end, x_start:x_end])
            
            theta_rad = 0.5 * np.arctan2(2 * sum_Jxy, sum_Jyy - sum_Jxx)
            theta_deg = (np.degrees(theta_rad) + 90.0) % 180.0
                
            cy = y_start + cell_h // 2
            cx = x_start + cell_w // 2
            
            angles.append(theta_deg)
            centers.append((cx, cy))
            grid_indices.append((i, j))
            
    return angles, centers, grid_indices


def draw_overlay(image, angles, centers, nominal_angle, line_length=20):
    overlay = image.copy()
    if len(overlay.shape) == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)
        
    for angle, center in zip(angles, centers):
        diff = min(abs(angle - nominal_angle), 180 - abs(angle - nominal_angle))
        
        if diff <= 5:
            color = (0, 230, 118)   # Green
        elif diff <= 15:
            color = (255, 171, 0)  # Amber
        else:
            color = (255, 23, 68)   # Red
            
        cx, cy = center
        rad = np.radians(angle)
        
        dx = int(line_length * np.cos(rad))
        dy = int(line_length * np.sin(rad))
        
        pt1 = (cx - dx, cy + dy) 
        pt2 = (cx + dx, cy - dy)
        
        cv2.arrowedLine(overlay, pt1, pt2, color, 2, tipLength=0.3)
        
    return overlay


# Upload Controls
col_up1, col_up2 = st.columns([3, 1])
with col_up1:
    uploaded_files = st.file_uploader("Upload CT Scan Screenshot(s)", type=["png", "jpg", "jpeg", "tiff"], accept_multiple_files=True)
with col_up2:
    st.write("")
    st.write("")
    use_demo = st.toggle("Use sample CT screenshots")

images_dict = {}

if use_demo and not uploaded_files:
    sample_files = [
        ("0° Nominal Sample (ct_sample_0deg.png)", os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_ct_0deg.png")),
        ("45° Nominal Sample (ct_sample_45deg.png)", os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_ct_45deg.png")),
        ("90° Nominal Sample (ct_sample_90deg.png)", os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_ct_90deg.png"))
    ]
    for label, spath in sample_files:
        if os.path.exists(spath):
            img = cv2.imread(spath)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images_dict[label] = img_rgb
    if images_dict:
        st.success("Loaded 3 sample CT screenshots (0°, 45°, 90° orientation)!")
elif uploaded_files:
    for f in uploaded_files:
        try:
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images_dict[f.name] = img_rgb
        except Exception as e:
            st.error(f"Error loading {f.name}: {e}")

if images_dict:
    processed_results = []
    
    for name, img in images_dict.items():
        angles, centers, grid_indices = analyze_fiber_orientation(img, grid_size, intensity_threshold, blur_kernel_size)
        overlay_img = draw_overlay(img, angles, centers, nominal_angle, line_length=arrow_length)
        
        # Circular angular statistics
        angles_arr = np.array(angles)
        rads_2x = np.radians(angles_arr * 2)
        mean_sin = np.mean(np.sin(rads_2x))
        mean_cos = np.mean(np.cos(rads_2x))
        mean_angle_2x = np.arctan2(mean_sin, mean_cos)
        mean_angle_deg = (np.degrees(mean_angle_2x) / 2) % 180.0
        
        R = np.sqrt(mean_sin**2 + mean_cos**2)
        std_deg = np.degrees(np.sqrt(max(0, -np.log(R))) / 2) if R > 0 else 0.0
        angular_error = min(abs(mean_angle_deg - nominal_angle), 180 - abs(mean_angle_deg - nominal_angle))
        
        df_grid = pd.DataFrame({
            'Angle': angles,
            'Grid Row': [idx[0] for idx in grid_indices],
            'Grid Col': [idx[1] for idx in grid_indices]
        })
        
        processed_results.append({
            'name': name,
            'image': img,
            'overlay': overlay_img,
            'mean_angle': mean_angle_deg,
            'std_deg': std_deg,
            'angular_error': angular_error,
            'df_grid': df_grid,
            'angles': angles
        })
        
    st.divider()
    
    tab_view, tab_stats, tab_comp, tab_export = st.tabs([
        "📸 Screenshots & Vector Overlays", 
        "📊 Orientation Analytics", 
        "📋 Multi-Screenshot Comparison", 
        "💾 Export Data"
    ])
    
    # ---------------------------------------------------------
    # TAB 1: SCREENSHOTS & VECTOR OVERLAYS
    # ---------------------------------------------------------
    with tab_view:
        st.subheader("CT Scan Vector Overlays")
        
        names_list = [r['name'] for r in processed_results]
        selected_name = st.selectbox("Select Screenshot to Inspect", options=names_list)
        selected_res = next(r for r in processed_results if r['name'] == selected_name)
        
        # Metric Cards for Selected Image
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Nominal Angle", f"{nominal_angle:.1f}°", border=True)
        with m2:
            st.metric("Measured Mean Angle", f"{selected_res['mean_angle']:.1f}°", border=True)
        with m3:
            st.metric("Circular Std Dev", f"{selected_res['std_deg']:.1f}°", border=True)
        with m4:
            st.metric("Angular Error", f"{selected_res['angular_error']:.1f}°", border=True)
            
        st.write("")
        c_img1, c_img2 = st.columns(2)
        with c_img1:
            with st.container(border=True):
                st.markdown("**Original CT Screenshot**")
                st.image(selected_res['image'], width="stretch")
        with c_img2:
            with st.container(border=True):
                st.markdown("**Fiber Vector Overlay** (Green: $\\le 5^\circ$, Amber: $\\le 15^\circ$, Red: $>15^\circ$)")
                st.image(selected_res['overlay'], width="stretch")

    # ---------------------------------------------------------
    # TAB 2: ORIENTATION ANALYTICS
    # ---------------------------------------------------------
    with tab_stats:
        st.subheader(f"Orientation Distribution & Heatmap — {selected_name}")
        
        an_col1, an_col2 = st.columns(2)
        with an_col1:
            with st.container(border=True):
                st.markdown("**Angle Distribution Histogram**")
                fig_hist = px.histogram(
                    selected_res['df_grid'], x="Angle", nbins=20, 
                    title="Fiber Orientation Angle Distribution",
                    labels={'Angle': 'Fiber Angle (°)'},
                    color_discrete_sequence=['#10B981']
                )
                fig_hist.add_vline(x=nominal_angle, line_dash="dash", line_color="#EF4444", annotation_text="Nominal Target")
                st.plotly_chart(fig_hist, width="stretch")
                
        with an_col2:
            with st.container(border=True):
                st.markdown("**Spatial Orientation Heatmap**")
                heatmap_data = selected_res['df_grid'].pivot(index='Grid Row', columns='Grid Col', values='Angle')
                fig_heat = px.imshow(
                    heatmap_data, 
                    labels=dict(x="Grid Column", y="Grid Row", color="Angle (°)"),
                    title="Spatial Grid Fiber Angles",
                    color_continuous_scale="Viridis"
                )
                fig_heat.update_yaxes(autorange="reversed")
                st.plotly_chart(fig_heat, width="stretch")

    # ---------------------------------------------------------
    # TAB 3: MULTI-SCREENSHOT COMPARISON
    # ---------------------------------------------------------
    with tab_comp:
        st.subheader("Summary Comparison Across All Uploaded Screenshots")
        
        comp_data = []
        for r in processed_results:
            comp_data.append({
                'Screenshot Name': r['name'],
                'Nominal Target (°)': round(nominal_angle, 1),
                'Measured Mean (°)': round(r['mean_angle'], 1),
                'Circular Std Dev (°)': round(r['std_deg'], 1),
                'Angular Error (°)': round(r['angular_error'], 1)
            })
            
        comp_df = pd.DataFrame(comp_data)
        st.dataframe(comp_df, width="stretch", hide_index=True)
        
        st.subheader("Angular Error Comparison Chart")
        with st.container(border=True):
            fig_comp_bar = px.bar(
                comp_df, x="Screenshot Name", y="Angular Error (°)", 
                color="Angular Error (°)", 
                title="Angular Alignment Error by Screenshot",
                color_continuous_scale="Reds"
            )
            st.plotly_chart(fig_comp_bar, width="stretch")

    # ---------------------------------------------------------
    # TAB 4: EXPORT DATA
    # ---------------------------------------------------------
    with tab_export:
        st.subheader("Export Orientation CSV Data")
        st.markdown("Download full grid cell orientation tensors for downstream analysis or FEA mapping.")
        
        all_export_dfs = []
        for r in processed_results:
            d_export = r['df_grid'].copy()
            d_export.insert(0, 'Screenshot Name', r['name'])
            all_export_dfs.append(d_export)
            
        combined_export_df = pd.concat(all_export_dfs, ignore_index=True)
        st.dataframe(combined_export_df, width="stretch", hide_index=True)
        
        csv_export = combined_export_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Grid Orientation CSV",
            data=csv_export,
            file_name='ct_fiber_orientation_summary.csv',
            mime='text/csv',
        )

else:
    st.info("Please upload CT scan screenshot image(s) from the sidebar or toggle sample CT screenshots to begin analysis.")

st.markdown("<br><br><p style='text-align: center; font-size: 11px; color: gray;'>Created by Product Design Engineering Intern, Advanced Architecture</p>", unsafe_allow_html=True)
