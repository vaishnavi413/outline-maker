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
    import threading, time, tempfile, importlib
    import io
    import numpy as np
    import cv2
    import backend.processing
    importlib.reload(backend.processing)
    from backend.processing import (
        remove_bg, 
        extract_contours, 
        create_offset_contour, 
        generate_exports,
        extract_individual_sticker_exports
    )
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

# Helper for instant white background thresholding (ideal for JPG sheets)
def remove_white_bg(image_bytes: bytes, threshold: int = 240) -> bytes:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return image_bytes
    bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    white_mask = (img[:, :, 0] >= threshold) & (img[:, :, 1] >= threshold) & (img[:, :, 2] >= threshold)
    bgra[white_mask, 3] = 0
    _, buffer = cv2.imencode(".png", bgra)
    return buffer.tobytes()

# Sidebar - Controls Setup
st.sidebar.markdown('<div class="sidebar-header">⚙️ Mode & Processing</div>', unsafe_allow_html=True)
bg_mode = st.sidebar.radio(
    "Background Removal Method",
    ["White Background Threshold (Best for JPG sheets)", "AI Background Removal (rembg for complex photos)"]
)
separate_objects = st.sidebar.checkbox("Separate Outlines for Each Picture", True)
clean_lines = st.sidebar.checkbox("Clean sheet guide lines & boxes", True)
disconnect_dist = st.sidebar.slider("Disconnect nearby pictures (Bridge Breaker px)", 0, 20, 5, step=1)

st.sidebar.markdown('---')
st.sidebar.markdown('<div class="sidebar-header">🎨 Outline Controls</div>', unsafe_allow_html=True)
offset_mm = st.sidebar.slider("White border offset (mm)", 0.0, 30.0, 5.0, step=0.5)
thickness_px = st.sidebar.slider("Outline thickness (px)", 1, 15, 2)
corner_type = st.sidebar.selectbox("Corner join style", ["Round", "Square", "Miter"])
smooth = st.sidebar.checkbox("Smooth contour (organic curves)", True)
fill_holes = st.sidebar.checkbox("Fill internal holes (solid sticker backing)", True)
min_area = st.sidebar.slider("Filter tiny speckles / noise (min px²)", 0, 2000, 300, step=50)

st.sidebar.markdown("---")
st.sidebar.markdown('<div class="sidebar-header">🎨 Style & Colors</div>', unsafe_allow_html=True)
fill_hex = st.sidebar.color_picker("Sticker backing fill color", "#FFFFFF")
stroke_hex = st.sidebar.color_picker("Cut line stroke color", "#000000")

def hex_to_rgba(hex_str, alpha=255):
    h = hex_str.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4)) + (alpha,)

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Multi-Picture JPG Tips:**\n\n"
    "* **Disconnect nearby pictures**: Severs thin pink lines and touching edges between separate pictures.\n"
    "* **Separate Outlines for Each Picture**: Generates an independent outline around EVERY picture on the sheet!\n"
    "* **White Background Threshold**: Instant background removal for JPG sheets with a white background."
)

# File Uploader
uploaded = st.file_uploader(
    "Upload your image (PNG, JPG, JPEG, WEBP, SVG)", 
    type=["png", "jpg", "jpeg", "webp", "svg"]
)

if uploaded:
    image_bytes = uploaded.read()
    
    # Remove background using chosen method
    if "White Background" in bg_mode:
        bg_removed = remove_white_bg(image_bytes)
    else:
        bg_removed = get_cached_bg_removed(image_bytes)
    
    # Extract contours with noise, line, and bridge-breaker filters
    contours, shape = extract_contours(
        bg_removed, 
        min_area=float(min_area), 
        clean_lines=clean_lines,
        disconnect_dist=int(disconnect_dist)
    )
    img_h, img_w = shape[0], shape[1]

    # Convert mm to pixels (300 DPI layout calculations)
    offset_px = offset_mm * (300 / 25.4)
    join_style_map = {"Round": 1, "Square": 3, "Miter": 2}
    join_style = join_style_map[corner_type]

    # Calculate offset contour path
    offset_contour = create_offset_contour(
        contours, 
        offset_px, 
        join_style=join_style, 
        smooth=smooth, 
        fill_holes=fill_holes,
        separate_objects=separate_objects
    )

    # Convert hex colors to RGBA
    fill_rgba = hex_to_rgba(fill_hex, 255)
    stroke_rgba = hex_to_rgba(stroke_hex, 255)

    # Generate print outputs (PNG, SVG, DXF, PDF)
    exports = generate_exports(bg_removed, offset_contour, thickness_px, img_w, img_h, fill_color=fill_rgba, stroke_color=stroke_rgba)

    # Visual Comparison columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖼️ Original (Background Isolated)")
        st.image(bg_removed, use_container_width=True)

    with col2:
        st.subheader("✨ Sticker Sheet Preview (Separate Outlines)")
        st.image(exports["png"], use_container_width=True)

    st.markdown("### 💾 Download Full Sheet Print-Ready Files (300 DPI)")
    
    # Global Sheet Download Button Layout Grid (2 rows x 3 cols)
    g_row1_col1, g_row1_col2, g_row1_col3 = st.columns(3)
    g_row2_col1, g_row2_col2, g_row2_col3 = st.columns(3)

    mime_map = {
        "png": "image/png", 
        "svg": "image/svg+xml", 
        "pdf": "application/pdf", 
        "dxf": "application/dxf"
    }

    with g_row1_col1:
        st.download_button(
            label="🖼️ Download Full Sticker PNG",
            data=exports["png"],
            file_name="sticker_full_sheet.png",
            mime=mime_map["png"],
            key="dl_full_sheet_png"
        )

    with g_row1_col2:
        st.download_button(
            label="🔲 Download Border ONLY PNG",
            data=exports["border_png"],
            file_name="border_only_sheet.png",
            mime=mime_map["png"],
            key="dl_border_sheet_png"
        )

    with g_row1_col3:
        st.download_button(
            label="📄 Download Outline ONLY PDF",
            data=exports["outline_pdf"],
            file_name="outline_only_sheet.pdf",
            mime=mime_map["pdf"],
            key="dl_outline_pdf_sheet"
        )

    with g_row2_col1:
        st.download_button(
            label="📐 Download SVG Vector Cut",
            data=exports["svg"],
            file_name="sticker_cutline.svg",
            mime=mime_map["svg"],
            key="dl_svg_sheet"
        )

    with g_row2_col2:
        st.download_button(
            label="📄 Download Full Sticker PDF",
            data=exports["pdf"],
            file_name="sticker_layout.pdf",
            mime=mime_map["pdf"],
            key="dl_pdf_sheet"
        )

    with g_row2_col3:
        st.download_button(
            label="💻 Download DXF (Plotter/CAD)",
            data=exports["dxf"],
            file_name="sticker_dxf.dxf",
            mime=mime_map["dxf"],
            key="dl_dxf_sheet"
        )

    # Individual Picture Exports (Separate Download Buttons per Picture)
    ind_exports = extract_individual_sticker_exports(
        bg_removed, 
        offset_contour, 
        thickness_px, 
        fill_color=fill_rgba, 
        stroke_color=stroke_rgba
    )

    if ind_exports:
        st.markdown("---")
        st.markdown(f"### ✂️ Separate Download Buttons per Picture ({len(ind_exports)} detected)")
        st.caption("Download independent outline PDFs, borders, sticker PNGs, or cut lines for each picture individually.")
        
        # Display in rows of 3 columns
        for row_idx in range(0, len(ind_exports), 3):
            row_items = ind_exports[row_idx:row_idx+3]
            cols = st.columns(len(row_items))
            for i, item in enumerate(row_items):
                with cols[i]:
                    st.markdown(f"#### 🖼️ Picture #{item['index']}")
                    st.image(item["full_png"], caption=f"Picture #{item['index']} Preview", use_container_width=True)
                    
                    st.download_button(
                        label=f"📄 Download Outline ONLY PDF #{item['index']}",
                        data=item["outline_pdf"],
                        file_name=f"picture_{item['index']}_outline.pdf",
                        mime=mime_map["pdf"],
                        key=f"dl_outline_pdf_ind_{item['index']}"
                    )
                    st.download_button(
                        label=f"🔲 Download Border ONLY PNG #{item['index']}",
                        data=item["border_png"],
                        file_name=f"picture_{item['index']}_border_only.png",
                        mime=mime_map["png"],
                        key=f"dl_border_ind_{item['index']}"
                    )
                    st.download_button(
                        label=f"🖼️ Download Sticker PNG #{item['index']}",
                        data=item["full_png"],
                        file_name=f"picture_{item['index']}_sticker.png",
                        mime=mime_map["png"],
                        key=f"dl_full_ind_{item['index']}"
                    )
                    st.download_button(
                        label=f"✂️ Download Cut Line SVG #{item['index']}",
                        data=item["svg"],
                        file_name=f"picture_{item['index']}_cutline.svg",
                        mime=mime_map["svg"],
                        key=f"dl_svg_ind_{item['index']}"
                    )

    # Start temp folder cleanup
    start_cleanup_thread()

else:
    # Beautiful welcome placeholder when no file is uploaded
    st.info("👋 Upload an image above to start generating your cut paths and sticker layouts.")
