import streamlit as st
import cv2
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="CT Fiber Orientation Analyzer", layout="wide")

# Sidebar
st.sidebar.title("Controls")

uploaded_file = st.sidebar.file_uploader("Upload CT Scan", type=["mp4", "avi", "png", "jpg", "jpeg"])

nominal_orientation_str = st.sidebar.selectbox("Expected Nominal Orientation", ["0°", "45°", "90°"])
nominal_angle = float(nominal_orientation_str.replace("°", ""))

grid_size = st.sidebar.slider("Grid Resolution", min_value=3, max_value=10, value=5)

blur_kernel_size = st.sidebar.slider("Gaussian Blur Kernel Size", min_value=3, max_value=31, step=2, value=11)
intensity_threshold = st.sidebar.slider("Intensity Threshold", min_value=0, max_value=255, value=50)

# Helper for video frame extraction
@st.cache_resource
def save_uploaded_video(uploaded_file):
    import tempfile
    import shutil
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        uploaded_file.seek(0)
        shutil.copyfileobj(uploaded_file, tmp)
        return tmp.name

@st.cache_data
def get_video_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count

def get_single_frame(video_path, frame_idx):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None

@st.cache_data
def analyze_fiber_orientation(image, grid_size, threshold, blur_ksize):
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
        
    # Thresholding to isolate fibers
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    # Apply mask to gray
    gray = cv2.bitwise_and(gray, gray, mask=mask)
        
    # Image gradients using Sobel
    Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Structure tensor components
    Jxx = Ix**2
    Jyy = Iy**2
    Jxy = Ix * Iy
    
    # Gaussian smoothing on tensor components
    Jxx_smooth = cv2.GaussianBlur(Jxx, (blur_ksize, blur_ksize), 0)
    Jyy_smooth = cv2.GaussianBlur(Jyy, (blur_ksize, blur_ksize), 0)
    Jxy_smooth = cv2.GaussianBlur(Jxy, (blur_ksize, blur_ksize), 0)
    
    # Compute angles for each grid cell
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
            
            # Aggregate tensor in the cell
            sum_Jxx = np.sum(Jxx_smooth[y_start:y_end, x_start:x_end])
            sum_Jyy = np.sum(Jyy_smooth[y_start:y_end, x_start:x_end])
            sum_Jxy = np.sum(Jxy_smooth[y_start:y_end, x_start:x_end])
            
            theta_rad = 0.5 * np.arctan2(2 * sum_Jxy, sum_Jyy - sum_Jxx)
            theta_deg = np.degrees(theta_rad) + 90.0 # adding 90 degrees as requested
            
            # map to [0, 180] relative to longitudinal axis
            theta_deg = theta_deg % 180
                
            cy = y_start + cell_h // 2
            cx = x_start + cell_w // 2
            
            angles.append(theta_deg)
            centers.append((cx, cy))
            grid_indices.append((i, j))
            
    return angles, centers, grid_indices

@st.cache_data
def draw_overlay(image, angles, centers, nominal_angle):
    overlay = image.copy()
    if len(overlay.shape) == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)
        
    line_length = 20
    
    for angle, center in zip(angles, centers):
        # calculate deviation
        diff = min(abs(angle - nominal_angle), 180 - abs(angle - nominal_angle))
        
        if diff <= 5:
            color = (0, 255, 0) # green
        elif diff <= 15:
            color = (255, 165, 0) # orange
        else:
            color = (255, 0, 0) # red
            
        cx, cy = center
        rad = np.radians(angle)
        
        dx = int(line_length * np.cos(rad))
        dy = int(line_length * np.sin(rad))
        
        # in image coords, y goes down
        pt1 = (cx - dx, cy + dy) 
        pt2 = (cx + dx, cy - dy)
        
        cv2.arrowedLine(overlay, pt1, pt2, color, 2, tipLength=0.3)
        
    return overlay

st.title("CT Fiber Orientation Analyzer")

if uploaded_file is not None:
    filename = uploaded_file.name
    if filename.endswith(("mp4", "avi")):
        video_path = save_uploaded_video(uploaded_file)
        frame_count = get_video_frame_count(video_path)
        if frame_count > 0:
            frame_idx = st.sidebar.slider("Select Frame", 0, frame_count - 1, 0)
            current_frame = get_single_frame(video_path, frame_idx)
            if current_frame is None:
                st.error("Failed to read the selected frame.")
                st.stop()
        else:
            st.error("Failed to extract frames from video.")
            st.stop()
    else:
        # read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        current_frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if current_frame is not None:
            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
        else:
            st.error("Failed to read the image.")
            st.stop()
        
    angles, centers, grid_indices = analyze_fiber_orientation(current_frame, grid_size, intensity_threshold, blur_kernel_size)
    overlay_img = draw_overlay(current_frame, angles, centers, nominal_angle)
    
    # Calculate stats
    angles_arr = np.array(angles)
    
    # Circular mean/std since angles are [0, 180)
    rads_2x = np.radians(angles_arr * 2)
    mean_sin = np.mean(np.sin(rads_2x))
    mean_cos = np.mean(np.cos(rads_2x))
    mean_angle_2x = np.arctan2(mean_sin, mean_cos)
    mean_angle_deg = (np.degrees(mean_angle_2x) / 2) % 180
    
    R = np.sqrt(mean_sin**2 + mean_cos**2)
    std_deg = np.degrees(np.sqrt(max(0, -np.log(R))) / 2) if R > 0 else 0
    
    angular_error = min(abs(mean_angle_deg - nominal_angle), 180 - abs(mean_angle_deg - nominal_angle))
    
    # Top Metric Cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nominal (Expected)", f"{nominal_angle:.1f}°")
    col2.metric("Measured Mean Angle", f"{mean_angle_deg:.1f}°")
    col3.metric("Standard Deviation", f"{std_deg:.1f}°")
    col4.metric("Angular Error", f"{angular_error:.1f}°")
    
    st.divider()
    
    main_col1, main_col2 = st.columns(2)
    
    with main_col1:
        st.subheader("Original vs Overlay CT Slice")
        st.image(current_frame, caption="Original CT Slice")
        st.image(overlay_img, caption="Overlay CT Slice (Green: ≤5°, Orange: ≤15°, Red: >15°)")
        
    with main_col2:
        st.subheader("Sectional Analysis")
        
        # 1. Histogram
        df = pd.DataFrame({
            'Angle': angles,
            'Grid Row': [idx[0] for idx in grid_indices],
            'Grid Col': [idx[1] for idx in grid_indices]
        })
        
        fig_hist = px.histogram(df, x="Angle", nbins=20, title="Fiber Orientation Distribution",
                                labels={'Angle': 'Angle (Degrees)'})
        st.plotly_chart(fig_hist)
        
        # 2. Heatmap
        heatmap_data = df.pivot(index='Grid Row', columns='Grid Col', values='Angle')
        fig_heat = px.imshow(heatmap_data, 
                             labels=dict(x="Grid Col", y="Grid Row", color="Angle"),
                             title="Spatial Breakdown of Orientation Angles",
                             color_continuous_scale="Viridis")
        fig_heat.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_heat)

    # Export Section
    st.subheader("Export")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Summary CSV",
        data=csv,
        file_name='fiber_orientation_summary.csv',
        mime='text/csv',
    )
    
else:
    st.info("Please upload a CT scan video or image from the sidebar to begin analysis.")
