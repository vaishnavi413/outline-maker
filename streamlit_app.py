import sys
import os

# Ensure the root directory is on the path for relative imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

# Set page config at the very beginning
st.set_page_config(
    page_title="Outline Maker - Die-Cut Sticker Generator",
    page_icon="✂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    import threading, time, tempfile
    import io
    from backend.processing import remove_bg, extract_contours, create_offset_contour, generate_exports
except Exception as e:
    st.error("Failed to start the application due to an import error. Please check your dependencies.")
    st.exception(e)
    st.stop()

# Inject premium custom CSS styles
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    /* Global styling overrides */
    html, body, [class*="css"], .stApp {
        font-family: 'Outfit', sans-serif !important;
        background-color: #F8FAFC !important;
    }
    
    /* Header/Hero Section Banner */
    .hero-container {
        background: linear-gradient(135deg, #4F46E5 0%, #06B6D4 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 15px -3px rgba(79, 70, 229, 0.15);
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.025em;
        line-height: 1.15;
    }
    .hero-subtitle {
        font-size: 1.15rem;
        opacity: 0.9;
        margin-top: 0.5rem;
        font-weight: 300;
    }

    /* Column Container Cards */
    div[data-testid="stColumn"] {
        background-color: white;
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03), 0 2px 4px -1px rgba(0, 0, 0, 0.02);
        transition: all 0.25s ease-in-out;
    }
    div[data-testid="stColumn"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 20px -3px rgba(0, 0, 0, 0.06), 0 4px 6px -2px rgba(0, 0, 0, 0.03);
        border-color: #CBD5E1;
    }

    /* Styled Download Buttons */
    .stDownloadButton button {
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.5rem !important;
        width: 100% !important;
        transition: all 0.2s ease-in-out !important;
        box-shadow: 0 4px 6px -1px rgba(79, 70, 229, 0.2) !important;
    }
    .stDownloadButton button:hover {
        transform: translateY(-1px) scale(1.01) !important;
        box-shadow: 0 6px 12px -1px rgba(79, 70, 229, 0.3) !important;
    }
    
    /* Clean uploader layout styling */
    div[data-testid="stFileUploader"] {
        background-color: white;
        border: 2px dashed #CBD5E1;
        border-radius: 14px;
        padding: 1.5rem;
        transition: border-color 0.2s ease;
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: #6366F1;
    }

    /* Custom sidebar header */
    .sidebar-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)

# Cache background removal so slider updates are instantaneous
@st.cache_data(show_spinner="🤖 Removing background using AI...")
def get_cached_bg_removed(image_bytes: bytes) -> bytes:
    return remove_bg(image_bytes)

# Cleanup thread to delete temporary files after 5 minutes
def cleanup_temp_dir(lifetime_seconds: int = 300):
    temp_dir = tempfile.gettempdir()
    while True:
        now = time.time()
        for fname in os.listdir(temp_dir):
            path = os.path.join(temp_dir, fname)
            if os.path.isfile(path) and now - os.path.getmtime(path) > lifetime_seconds:
                try:
                    os.remove(path)
                except Exception:
                    pass
        time.sleep(60)

def start_cleanup_thread():
    if not st.session_state.get("cleanup_started", False):
        thread = threading.Thread(target=cleanup_temp_dir, daemon=True)
        thread.start()
        st.session_state.cleanup_started = True

# Hero Header Banner
st.markdown("""
<div class="hero-container">
    <h1 class="hero-title">✂️ Outline Maker</h1>
    <div class="hero-subtitle">Generate professional print-ready white borders and cut lines for custom stickers & die-cuts instantly.</div>
</div>
""", unsafe_allow_html=True)

# Sidebar - Controls Setup
st.sidebar.markdown('<div class="sidebar-header">🎨 Outline Controls</div>', unsafe_allow_html=True)
offset_mm = st.sidebar.slider("White border offset (mm)", 0.0, 20.0, 5.0, step=0.5)
thickness_px = st.sidebar.slider("Outline thickness (px)", 1, 10, 2)
corner_type = st.sidebar.selectbox("Corner join style", ["Round", "Square", "Miter"])
smooth = st.sidebar.checkbox("Smooth contour (spline simplify)", True)

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Tips:**\n\n"
    "* **Round** corners give standard smooth sticker paths.\n"
    "* **Square** yields blockier polygonal borders.\n"
    "* Drag sliders to watch the preview update instantly."
)

# File Uploader
uploaded = st.file_uploader(
    "Upload your image (PNG, JPG, JPEG, WEBP, SVG)", 
    type=["png", "jpg", "jpeg", "webp", "svg"]
)

if uploaded:
    image_bytes = uploaded.read()
    
    # Remove background using cached function
    bg_removed = get_cached_bg_removed(image_bytes)
    
    # Extract contours
    contours, shape = extract_contours(bg_removed)
    img_h, img_w = shape[0], shape[1]

    # Convert mm to pixels (300 DPI layout calculations)
    offset_px = offset_mm * (300 / 25.4)
    join_style_map = {"Round": 1, "Square": 3, "Miter": 2}
    join_style = join_style_map[corner_type]

    # Calculate offset contour path
    offset_contour = create_offset_contour(contours, offset_px, join_style, smooth)

    # Generate print outputs (PNG, SVG, DXF, PDF)
    exports = generate_exports(bg_removed, offset_contour, thickness_px, img_w, img_h)

    # Visual Comparison columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖼️ Original (Background Removed)")
        st.image(bg_removed, use_container_width=True)

    with col2:
        st.subheader("✨ Sticker Preview (With Cut Path)")
        st.image(exports["png"], use_container_width=True)

    st.markdown("### 💾 Download Print-Ready Files (300 DPI)")
    
    # Download Button Layout Grid
    dl_col1, dl_col2, dl_col3, dl_col4 = st.columns(4)
    mime_map = {
        "png": "image/png", 
        "svg": "image/svg+xml", 
        "pdf": "application/pdf", 
        "dxf": "application/dxf"
    }

    with dl_col1:
        st.download_button(
            label="Download PNG (Transparent)",
            data=exports["png"],
            file_name="sticker_print.png",
            mime=mime_map["png"]
        )

    with dl_col2:
        st.download_button(
            label="Download SVG (Vector Cut)",
            data=exports["svg"],
            file_name="sticker_cutline.svg",
            mime=mime_map["svg"]
        )

    with dl_col3:
        st.download_button(
            label="Download PDF (DPI Aligned)",
            data=exports["pdf"],
            file_name="sticker_layout.pdf",
            mime=mime_map["pdf"]
        )

    with dl_col4:
        st.download_button(
            label="Download DXF (CAD/Plotter)",
            data=exports["dxf"],
            file_name="sticker_dxf.dxf",
            mime=mime_map["dxf"]
        )

    # Start temp folder cleanup
    start_cleanup_thread()

else:
    # Beautiful welcome placeholder when no file is uploaded
    st.info("👋 Upload an image above to start generating your cut paths and sticker layouts.")
