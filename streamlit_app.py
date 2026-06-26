import sys
import os

# Ensure the root directory is on the path for relative imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import threading, time, tempfile
import io
from backend.processing import remove_bg, extract_contours, create_offset_contour, generate_exports

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

st.title("Contour Cut Line Generator (Streamlit)")
uploaded = st.file_uploader("Upload image (PNG, JPG, JPEG, WEBP, SVG)", type=["png", "jpg", "jpeg", "webp", "svg"]) 

if uploaded:
    image_bytes = uploaded.read()
    # Remove background using rembg
    bg_removed = remove_bg(image_bytes)
    st.image(bg_removed, caption="Background removed", use_column_width=True)

    # Extract contours from alpha mask
    contours, shape = extract_contours(bg_removed)
    img_h, img_w = shape[0], shape[1]

    # Adjustable parameters
    offset_mm = st.slider("White border offset (mm)", 0, 20, 5)
    thickness_px = st.slider("Outline thickness (px)", 1, 10, 2)
    corner_type = st.selectbox("Corner type", ["Round", "Square", "Miter"])
    smooth = st.checkbox("Smooth contour", True)

    # Convert mm to pixels (300 DPI)
    offset_px = offset_mm * (300 / 25.4)
    join_style_map = {"Round": 1, "Square": 3, "Miter": 2}
    join_style = join_style_map[corner_type]

    # Generate offset contour
    offset_contour = create_offset_contour(contours, offset_px, join_style, smooth)

    # Generate export files
    exports = generate_exports(bg_removed, offset_contour, thickness_px, img_w, img_h)

    # Show preview PNG
    st.image(exports["png"], caption="Sticker preview", use_column_width=True)

    # Start cleanup thread
    start_cleanup_thread()

    mime_map = {"png": "image/png", "svg": "image/svg+xml", "pdf": "application/pdf", "dxf": "application/dxf"}
    for fmt, data in exports.items():
        st.download_button(
            label=f"Download {fmt.upper()}",
            data=data,
            file_name=f"contour.{fmt}",
            mime=mime_map[fmt]
        )
