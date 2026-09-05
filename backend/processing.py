import cv2
import numpy as np
from rembg import remove
from PIL import Image
from shapely.geometry import Polygon, MultiPolygon
from shapely import buffer
import io
import svgpathtools
import ezdxf
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.pagesizes import A4
import base64

def remove_bg(image_bytes: bytes) -> bytes:
    """Removes background from image bytes and returns PNG bytes."""
    result = remove(image_bytes)
    return result

def extract_contours(image_bytes: bytes, min_area: float = 50.0) -> tuple:
    """Extracts base contours from an image with transparency and filters small noise artifacts."""
    # Read image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        raise ValueError("Invalid image file provided.")

    # If no alpha channel, return empty or handle threshold
    if len(img.shape) < 3 or img.shape[2] != 4:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) >= 3 else img
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        alpha = thresh
    else:
        alpha = img[:, :, 3]
        _, alpha = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)
        
    # Apply morphological closing and opening to clean mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel)
    
    # Find external contours
    contours, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out small noise artifacts based on area
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
    
    return filtered_contours, img.shape

def create_offset_contour(contours, offset_px: float, join_style: int = 1, smooth: bool = True, fill_holes: bool = True):
    """
    Creates an offset contour using Shapely with unary_union, hole filling, and smooth rounding.
    join_style: 1 for round, 2 for miter, 3 for bevel (square)
    """
    from shapely.ops import unary_union

    polygons = []
    for cnt in contours:
        if len(cnt) >= 3:
            pts = cnt.squeeze()
            if len(pts.shape) == 2 and len(pts) >= 3:
                poly = Polygon(pts)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if poly.is_valid and not poly.is_empty:
                    if fill_holes:
                        # Remove internal holes to create a solid sticker backing
                        poly = Polygon(poly.exterior.coords)
                    polygons.append(poly)
    
    if not polygons:
        return Polygon()

    # Seamlessly merge all polygons into a unified shape
    merged = unary_union(polygons)
    
    # Apply buffer offset
    if offset_px > 0:
        offset_poly = merged.buffer(offset_px, join_style=join_style, cap_style=1)
    else:
        offset_poly = merged
    
    # Apply curve smoothing for organic, professional cutlines
    if smooth and offset_poly and not offset_poly.is_empty:
        # Simplify slightly then round out jagged corners
        offset_poly = offset_poly.simplify(1.5, preserve_topology=True)
        offset_poly = offset_poly.buffer(2.0, join_style=1).buffer(-2.0, join_style=1)
        
    return offset_poly

def polygon_to_svg_path(poly, image_height: int):
    """Converts a Shapely polygon/multipolygon to SVG path data string."""
    def extract_path(p):
        if p.is_empty:
            return ""
        coords = list(p.exterior.coords)
        if not coords:
            return ""
        path = f"M {coords[0][0]:.2f} {coords[0][1]:.2f} "
        for x, y in coords[1:]:
            path += f"L {x:.2f} {y:.2f} "
        path += "Z "
        
        # Handle holes if any exist
        for interior in p.interiors:
            coords = list(interior.coords)
            if not coords:
                continue
            path += f"M {coords[0][0]:.2f} {coords[0][1]:.2f} "
            for x, y in coords[1:]:
                path += f"L {x:.2f} {y:.2f} "
            path += "Z "
        return path

    if isinstance(poly, MultiPolygon):
        return "".join(extract_path(p) for p in poly.geoms)
    elif isinstance(poly, Polygon):
        return extract_path(poly)
    return ""

def generate_exports(image_bytes: bytes, poly, thickness: float, width: int, height: int, fill_color=(255, 255, 255, 255), stroke_color=(0, 0, 0, 255)):
    """Generates print-ready export formats with dynamic canvas padding to eliminate border clipping."""
    from shapely.affinity import translate
    
    # Calculate dynamic padding based on polygon bounds to prevent edge clipping
    pad_left = 0
    pad_right = 0
    pad_top = 0
    pad_bottom = 0
    
    if poly and not poly.is_empty:
        minx, miny, maxx, maxy = poly.bounds
        pad_left = int(max(0, -minx) + thickness / 2 + 10)
        pad_right = int(max(0, maxx - width) + thickness / 2 + 10)
        pad_top = int(max(0, -miny) + thickness / 2 + 10)
        pad_bottom = int(max(0, maxy - height) + thickness / 2 + 10)
        
    new_width = width + pad_left + pad_right
    new_height = height + pad_top + pad_bottom
    
    # Translate polygon to padded coordinate space
    if poly and not poly.is_empty:
        poly = translate(poly, xoff=pad_left, yoff=pad_top)

    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    
    # Create padded BGRA canvas for the sticker background
    out_cv = np.zeros((new_height, new_width, 4), dtype=np.uint8)
    
    # Format colors for OpenCV (BGRA)
    fill_bgra = (fill_color[2], fill_color[1], fill_color[0], fill_color[3])
    stroke_bgra = (stroke_color[2], stroke_color[1], stroke_color[0], stroke_color[3])

    def draw_filled_poly(p, cv2_img):
        if not p.is_empty:
            coords = np.array(p.exterior.coords, dtype=np.int32)
            cv2.fillPoly(cv2_img, [coords], fill_bgra)
            for interior in p.interiors:
                coords = np.array(interior.coords, dtype=np.int32)
                cv2.fillPoly(cv2_img, [coords], (0, 0, 0, 0))

    def draw_outline_poly(p, cv2_img):
        if not p.is_empty:
            coords = np.array(p.exterior.coords, dtype=np.int32)
            cv2.polylines(cv2_img, [coords], True, stroke_bgra, int(max(1, thickness)), lineType=cv2.LINE_AA)
            for interior in p.interiors:
                coords = np.array(interior.coords, dtype=np.int32)
                cv2.polylines(cv2_img, [coords], True, stroke_bgra, int(max(1, thickness)), lineType=cv2.LINE_AA)

    if isinstance(poly, MultiPolygon):
        for p in poly.geoms:
            draw_filled_poly(p, out_cv)
        for p in poly.geoms:
            draw_outline_poly(p, out_cv)
    elif isinstance(poly, Polygon):
        draw_filled_poly(poly, out_cv)
        draw_outline_poly(poly, out_cv)
        
    # Convert drawn background to PIL and composite original image on top
    bg_img = Image.fromarray(cv2.cvtColor(out_cv, cv2.COLOR_BGRA2RGBA))
    bg_img.paste(img, (pad_left, pad_top), img)
    
    buffer = io.BytesIO()
    # Print-ready 300 DPI output
    bg_img.save(buffer, format="PNG", dpi=(300, 300))
    png_bytes = buffer.getvalue()

    # 2. SVG (just the cut line on padded viewbox)
    stroke_hex = f"#{stroke_color[0]:02x}{stroke_color[1]:02x}{stroke_color[2]:02x}"
    svg_path = polygon_to_svg_path(poly, new_height)
    svg_content = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {new_width} {new_height}" width="{new_width}" height="{new_height}">
    <path d="{svg_path}" fill="none" stroke="{stroke_hex}" stroke-width="{thickness}"/>
</svg>'''

    # 3. DXF
    doc = ezdxf.new()
    msp = doc.modelspace()
    def add_poly_to_dxf(p):
        if not p.is_empty:
            coords = list(p.exterior.coords)
            msp.add_lwpolyline(coords, close=True, dxfattribs={'color': 1})
            for interior in p.interiors:
                coords = list(interior.coords)
                msp.add_lwpolyline(coords, close=True, dxfattribs={'color': 1})
    
    if isinstance(poly, MultiPolygon):
        for p in poly.geoms:
            add_poly_to_dxf(p)
    elif isinstance(poly, Polygon):
        add_poly_to_dxf(poly)
        
    dxf_io = io.StringIO()
    doc.write(dxf_io)
    dxf_content = dxf_io.getvalue()
    
    # 4. PDF (image + outline on padded page size)
    pdf_buffer = io.BytesIO()
    c = pdf_canvas.Canvas(pdf_buffer, pagesize=(new_width, new_height))
    
    import tempfile
    import os
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            img.save(tmp.name)
            tmp_path = tmp.name
        c.drawImage(tmp_path, pad_left, pad_bottom, width, height, mask='auto')
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
    
    c.setStrokeColorRGB(stroke_color[0]/255.0, stroke_color[1]/255.0, stroke_color[2]/255.0)
    c.setLineWidth(thickness)
    
    def draw_poly_on_pdf(p):
        if not p.is_empty:
            path = c.beginPath()
            coords = list(p.exterior.coords)
            if coords:
                path.moveTo(coords[0][0], new_height - coords[0][1])
                for x, y in coords[1:]:
                    path.lineTo(x, new_height - y)
                path.close()
                c.drawPath(path, stroke=1, fill=0)
            
            for interior in p.interiors:
                path = c.beginPath()
                coords = list(interior.coords)
                if coords:
                    path.moveTo(coords[0][0], new_height - coords[0][1])
                    for x, y in coords[1:]:
                        path.lineTo(x, new_height - y)
                    path.close()
                    c.drawPath(path, stroke=1, fill=0)
                    
    if isinstance(poly, MultiPolygon):
        for p in poly.geoms:
            draw_poly_on_pdf(p)
    elif isinstance(poly, Polygon):
        draw_poly_on_pdf(poly)
        
    c.save()
    pdf_bytes = pdf_buffer.getvalue()

    return {
        "png": png_bytes,
        "svg": svg_content.encode("utf-8"),
        "dxf": dxf_content.encode("utf-8"),
        "pdf": pdf_bytes
    }
