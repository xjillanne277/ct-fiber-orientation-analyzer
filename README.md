# CT Scan Through-Thickness Fiber Orientation Analyzer

An advanced research and engineering tool built with Python and Streamlit to analyze 3D micro-CT scan data of short-fiber-reinforced injection-molded composite materials. 

The application enables researchers and finite element analysts to non-destructively quantify fiber microstructure, determine directional orientation tensors, measure through-thickness layer morphology (the **skin-core effect**), and export calibrated anisotropic material inputs to improve **Finite Element Analysis (FEA)** simulation accuracy.

---

## 🎯 Application Purpose & Engineering Objectives

Standard structural FEA simulations often assume injection-molded plastics are isotropic, applying a single uniform Young's modulus ($E$) and yield strength across entire components. In reality, short-glass-fiber reinforced polymers (SFRTPs) exhibit substantial spatial and through-thickness **anisotropy** induced by injection molding fluid dynamics and mold wall shear.

This application serves as a dedicated research tool to:

1. **Estimate Fiber Percentage / Volume Fraction ($V_f$):**  
   Quantify the local and global fiber concentration to scale composite micromechanics equations (Halpin-Tsai, Mori-Tanaka, and Rule of Mixtures).
2. **Measure Layer Thicknesses (The Skin Effect):**  
   Identify transition inflection points, quantify distinct **skin layer** and **core layer** thicknesses, and compute the volume-averaged **skin-to-core ratio**.
3. **Determine Local & Through-Thickness Fiber Orientation:**  
   Extract 2D/3D structure tensor components and calculate second-order orientation tensor components ($A_{11}$ along the tensile/fill axis, $A_{22}$ across the transverse axis) slice-by-slice along the normalized thickness ($z_{\text{norm}} \in [-1.0, 1.0]$).
4. **Improve Finite Element Analysis (FEA) Accuracy:**  
   Provide high-fidelity, through-thickness orientation tensor distributions and volume-averaged material stiffness cards ($E_{\text{predicted}}$) directly formatted for multi-scale FEA software (e.g., Abaqus, ANSYS, Moldflow, Digimat).

---

## 🔬 Scientific Background: The Skin-Core Effect

During injection molding of fiber-filled thermoplastics, complex polymer melt rheology creates distinct morphological zones across the part thickness:

```
+-------------------------------------------------------------+  z = +1.0 (Top Mold Wall)
|  TOP SKIN LAYER: High shear freezes fibers parallel to flow |  A11 ≥ 0.5 (Highly Aligned)
+-------------------------------------------------------------+  z = +z_inflection
|                                                             |
|  CORE LAYER: Slower extensional flow causes transverse      |  A11 < 0.5 (Transverse/Random)
|              or out-of-plane fiber orientation              |
|                                                             |
+-------------------------------------------------------------+  z = -z_inflection
| BOTTOM SKIN LAYER: High shear freezes fibers parallel       |  A11 ≥ 0.5 (Highly Aligned)
+-------------------------------------------------------------+  z = -1.0 (Bottom Mold Wall)
```

- **Skin Layer ($A_{11} \ge 0.5$):** Near mold walls, high shear rates align fibers strongly parallel to the primary melt flow axis ($X$-axis), yielding high longitudinal stiffness ($E_0^\circ$).
- **Core Layer ($A_{11} < 0.5$):** Towards the center mid-plane, low shear and transverse extensional flow cause fibers to orient perpendicular to the flow axis ($Y$-axis) or remain dispersed, resulting in lower longitudinal stiffness ($E_{90}^\circ$).

### ⚠️ Critical Input Requirement: Full Cross-Sections
> **Important:** Analyzing only a top-surface planar slice (XY plane) is **insufficient** because it only captures the highly aligned skin layer. This produces an artificially inflated stiffness estimate and completely misses the compliant transverse core.  
> **Users must input cross-sections spanning the full thickness (XZ or YZ planes, or sequential Z-slice stacks)** containing both the outer skin and inner core to capture the true fiber variance required for accurate FEA simulations.

---

## 🚀 Step-by-Step Setup and Execution Guide

Follow these step-by-step instructions to set up and run the application on your local workstation.

### Prerequisites
- **Python 3.10 or higher** (Python 3.11/3.12/3.13 supported)
- **pip** package manager
- Recommended: 8 GB+ RAM for handling high-resolution 3D CT scan stacks

---

### Step 1: Clone or Navigate to the Project Directory

Open a terminal and navigate to the project directory:

```bash
cd /path/to/ct-fiber-orientation-analyzer
```

---

### Step 2: Set Up a Python Virtual Environment

Create and activate an isolated virtual environment to manage dependencies:

**On Linux / macOS:**
```bash
# Create virtual environment named .venv
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate
```

**On Windows (Command Prompt / PowerShell):**
```cmd
# Create virtual environment
python -m venv .venv

# Activate (Command Prompt)
.venv\Scripts\activate.bat

# Activate (PowerShell)
.venv\Scripts\Activate.ps1
```

---

### Step 3: Install Required Dependencies

Install the required Python libraries using the included `requirements.txt`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Dependencies Overview:
- `streamlit` — Modern interactive web UI framework
- `opencv-python-headless` — Computer vision and image processing (Sobel filters, structure tensors)
- `numpy` & `scipy` — Numerical matrix operations, Gaussian smoothing, tensor algebra
- `pandas` — Structured data manipulation and tabular analysis
- `plotly` — Interactive, publication-ready data visualizations
- `matplotlib` — Scientific image colormapping
- `Pillow` — Multi-page TIFF stack ingestion and image sequence decoding

---

### Step 4: Launch the Streamlit Application

Run the Streamlit app:

```bash
streamlit run app.py
```

Upon launching, Streamlit will start a local web server (default: `http://localhost:8501`) and automatically open the application in your default web browser.

To specify a custom port or run headless on a remote server:
```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

---

## 💻 How to Use the Application

### 1. Ingesting Data
The application provides multiple flexible input methods via the main screen:
- **Synthetic 3D Benchmark Mode (Default):** Toggle `"Use synthetic 3D Skin-Core CT stack"` to instantly test and explore the analyzer with a mathematically modeled 40-slice injection-molded specimen.
- **Multi-Page 3D TIFF Files (`.tif`, `.tiff`):** Upload a single volumetric file containing ordered Z-slices through the specimen thickness.
- **ZIP Archives (`.zip`):** Upload a compressed archive containing sequential slice image files (`.png`, `.jpg`, `.tif`).
- **Batch Slice Upload:** Multi-select individual slice files directly in the file uploader.

### 2. Configuring Analysis Parameters (Sidebar)
- **Fiber Modulus $E_f$ (GPa):** Elastic modulus of the reinforcement fiber (e.g., $E$-glass $\approx 72\text{ GPa}$, Carbon fiber $\approx 230\text{ GPa}$).
- **Matrix Modulus $E_m$ (GPa):** Modulus of the polymer resin matrix (e.g., Polyamide/PA66 $\approx 3.0\text{ GPa}$, Polypropylene $\approx 1.5\text{ GPa}$).
- **Fiber Volume Fraction $V_f$ (%):** Target fiber volume loading (e.g., $20\% - 50\%$).
- **Gaussian Blur Kernel Size:** Spatial window size ($3 \times 3$ to $31 \times 31$) for local structure tensor averaging.
- **Fiber Intensity Threshold:** Grayscale threshold ($0 - 255$) to isolate high-density fiber pixels from matrix background.
- **Z-Slice Binning / Downsampling:** Step size to downsample thick stacks for faster iterative tuning.

---

## 📊 Application Tabs & Features

| Tab | Feature & Description |
|---|---|
| 📖 **Background & Specimen Guide** | Theoretical foundation, microstructural anisotropy explanation, FEA simulation significance, visual reference guide comparing valid through-thickness cross-sections (skin + core) vs. invalid surface-only slices, and specimen preparation protocol. |
| 📈 **Through-Thickness Profile** | Interactive Plotly graph plotting orientation tensor components $A_{11}(z)$ and $A_{22}(z)$ across normalized thickness $z_{\text{norm}} \in [-1.0, 1.0]$, transition threshold line ($A_{11} = 0.5$), skin-to-core volumetric pie chart, and inflection layer thicknesses. |
| 🔍 **Interactive Slice Explorer** | Interactive depth slider to scrub through each individual Z-slice, displaying the raw grayscale CT slice side-by-side with an orientation vector field overlay (Green = parallel flow, Amber = intermediate, Red = transverse core) and slice-specific tensor metrics. |
| 📊 **Tensor Component Data** | Complete, interactive data table detailing slice index, normalized thickness, $A_{11}$, and $A_{22}$ values for auditing and validation. |
| 💾 **Export Metrics** | One-click CSV download (`ct_through_thickness_fiber_orientation_tensor.csv`) formatted for direct import into FEA material cards and homogenization solvers. |

---

## 📐 Mathematical Formulation

### 1. Structure Tensor & Gradient Field
For each 2D grayscale slice $I(x, y)$, image intensity gradients $I_x = \frac{\partial I}{\partial x}$ and $I_y = \frac{\partial I}{\partial y}$ are computed using $3 \times 3$ Sobel operators. The continuous structure tensor $\mathbf{J}$ is constructed via Gaussian spatial smoothing:

$$J_{xx} = G_\sigma * (I_x^2), \quad J_{yy} = G_\sigma * (I_y^2), \quad J_{xy} = G_\sigma * (I_x I_y)$$

### 2. Orientation Tensor Components
Because fiber orientation is orthogonal to the maximum intensity gradient:

$$A_{11} = \frac{J_{yy}}{J_{xx} + J_{yy} + \epsilon}, \qquad A_{22} = \frac{J_{xx}}{J_{xx} + J_{yy} + \epsilon}$$

- $A_{11} \to 1.0$: Fibers perfectly aligned with the $X$-axis (longitudinal tensile pull axis).
- $A_{22} \to 1.0$: Fibers aligned with the $Y$-axis (transverse cross-flow axis).
- $A_{11} \approx 0.5$: Planar random or 45° diagonal orientation.

### 3. Volume-Averaged Stiffness Estimation (Halpin-Tsai / Rule of Mixtures)
The global volume-averaged orientation tensor $A_{11,\text{global}}$ is computed across all $N$ slices:

$$A_{11,\text{global}} = \frac{1}{N} \sum_{z=1}^{N} A_{11}(z)$$

The effective longitudinal tensile modulus $E_{\text{predicted}}$ is estimated using orientation-weighted micro-mechanics:

$$E_{\text{predicted}} = E_m + A_{11,\text{global}} \cdot V_f \cdot (E_f - E_m)$$

---

## 📂 Repository Structure

```
ct-fiber-orientation-analyzer/
├── app.py                      # Main Streamlit application source code
├── requirements.txt            # Python package dependencies
├── README.md                   # Complete application documentation
├── sample_ct_0deg.png          # Reference CT slice: 0° longitudinal alignment
├── sample_ct_45deg.png         # Reference CT slice: 45° diagonal alignment
├── sample_ct_90deg.png         # Reference CT slice: 90° transverse alignment
└── CT Scan Images/             # Example CT scan cross-section images
    ├── black, 90deg, xz plane with core.png   # Full through-thickness cross section (Skin + Core)
    ├── black, 90deg, xz plane with core2.png  # Secondary through-thickness cross section
    ├── black, 90deg, xy plane, only skin.png  # Surface planar slice (Skin only)
    └── black, 90deg, xy plane, only skin2.png # Secondary planar skin slice
```

---

## 💡 Troubleshooting & FAQ

- **Error: `ModuleNotFoundError: No module named 'cv2'`**  
  Ensure you installed requirements using `pip install -r requirements.txt` within the activated virtual environment. OpenCV is provided by `opencv-python-headless`.
- **Large CT file upload limit:**  
  By default, Streamlit allows up to 200 MB uploads. For larger volumetric datasets, the upload limit has been configured in `.streamlit/config.toml` (`maxUploadSize = 999`).
- **Inverted coordinates / orientation:**  
  Ensure CT scans are aligned such that the specimen's longitudinal pull axis is horizontal ($X$-axis) and through-thickness slicing is along the $Z$-axis.

---

## 👤 Author & Context
Created as part of the **Advanced Architecture / Materials Science & Engineering (AA/MSE) Product Design Engineering Internship** research initiative to bridge micro-mechanics, automated image processing, and anisotropic FEA material modeling for high-reliability consumer hardware.
