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

def extract_contours(image_bytes: bytes) -> list:
    """Extracts base contours from an image with transparency."""
    # Read image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    
    # If no alpha channel, return empty or handle differently
    if img.shape[2] != 4:
        # Convert to grayscale and threshold
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        alpha = thresh
    else:
        alpha = img[:, :, 3]
        # Threshold alpha > 0 to create strict object mask
        _, alpha = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)
        
    # Apply morphological closing and opening to clean the mask, removing noise and small artifacts
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, img.shape

def create_offset_contour(contours, offset_px: float, join_style: int, smooth: bool = False):
    """
    Creates an offset contour using shapely.
    join_style: 1 for round, 2 for miter, 3 for bevel (square)
    """
    polygons = []
    for cnt in contours:
        if len(cnt) >= 3:
            # Squeeze to get standard points (N, 2)
            pts = cnt.squeeze()
            if len(pts.shape) == 2:
                poly = Polygon(pts)
                if poly.is_valid:
                    polygons.append(poly)
    
    if not polygons:
        return []

    # Merge polygons if needed, but here we can just offset all
    merged = MultiPolygon(polygons) if len(polygons) > 1 else polygons[0]
    
    # Offset
    # Shapely join_style: 1=round, 2=mitre, 3=bevel
    if offset_px > 0:
        offset_poly = merged.buffer(offset_px, join_style=join_style)
    else:
        offset_poly = merged
    
    # Smooth if required (simple simplification)
    if smooth and isinstance(offset_poly, (Polygon, MultiPolygon)):
        offset_poly = offset_poly.simplify(2.0, preserve_topology=True)
        
    return offset_poly

def polygon_to_svg_path(poly, image_height: int):
    """Converts a Shapely polygon/multipolygon to SVG path data string."""
    def extract_path(p):
        if p.is_empty:
            return ""
        coords = list(p.exterior.coords)
        if not coords:
            return ""
        path = f"M {coords[0][0]} {coords[0][1]} "
        for x, y in coords[1:]:
            path += f"L {x} {y} "
        path += "Z "
        
        # Handle holes
        for interior in p.interiors:
            coords = list(interior.coords)
            if not coords:
                continue
            path += f"M {coords[0][0]} {coords[0][1]} "
            for x, y in coords[1:]:
                path += f"L {x} {y} "
            path += "Z "
        return path

    if isinstance(poly, MultiPolygon):
        return "".join(extract_path(p) for p in poly.geoms)
    elif isinstance(poly, Polygon):
        return extract_path(poly)
    return ""

def generate_exports(image_bytes: bytes, poly, thickness: float, width: int, height: int):
    """Generates the different export formats with boundary padding to prevent outline clipping."""
    from shapely.affinity import translate
    
    # 1. Calculate dynamic padding based on polygon bounds to prevent edge clipping
    pad_left = 0
    pad_right = 0
    pad_top = 0
    pad_bottom = 0
    
    if poly and not poly.is_empty:
        minx, miny, maxx, maxy = poly.bounds
        pad_left = int(max(0, -minx) + thickness / 2 + 5)
        pad_right = int(max(0, maxx - width) + thickness / 2 + 5)
        pad_top = int(max(0, -miny) + thickness / 2 + 5)
        pad_bottom = int(max(0, maxy - height) + thickness / 2 + 5)
        
    new_width = width + pad_left + pad_right
    new_height = height + pad_top + pad_bottom
    
    # Translate polygon to padded coordinate space
    if poly and not poly.is_empty:
        poly = translate(poly, xoff=pad_left, yoff=pad_top)

    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    
    # Create padded BGRA canvas for the sticker background
    out_cv = np.zeros((new_height, new_width, 4), dtype=np.uint8)
    
    def draw_filled_poly(p, cv2_img):
        if not p.is_empty:
            coords = np.array(p.exterior.coords, dtype=np.int32)
            cv2.fillPoly(cv2_img, [coords], (255, 255, 255, 255))
            for interior in p.interiors:
                coords = np.array(interior.coords, dtype=np.int32)
                cv2.fillPoly(cv2_img, [coords], (0, 0, 0, 0))

    def draw_outline_poly(p, cv2_img):
        if not p.is_empty:
            coords = np.array(p.exterior.coords, dtype=np.int32)
            cv2.polylines(cv2_img, [coords], True, (0, 0, 0, 255), int(max(1, thickness)), lineType=cv2.LINE_AA)
            for interior in p.interiors:
                coords = np.array(interior.coords, dtype=np.int32)
                cv2.polylines(cv2_img, [coords], True, (0, 0, 0, 255), int(max(1, thickness)), lineType=cv2.LINE_AA)

    if isinstance(poly, MultiPolygon):
        for p in poly.geoms:
            draw_filled_poly(p, out_cv)
        for p in poly.geoms:
            draw_outline_poly(p, out_cv)
    elif isinstance(poly, Polygon):
        draw_filled_poly(poly, out_cv)
        draw_outline_poly(poly, out_cv)
        
    # Convert drawn background to PIL and composite original image on top (centered by padding offset)
    bg_img = Image.fromarray(cv2.cvtColor(out_cv, cv2.COLOR_BGRA2RGBA))
    bg_img.paste(img, (pad_left, pad_top), img)
    
    buffer = io.BytesIO()
    # Print-ready 300 DPI output
    bg_img.save(buffer, format="PNG", dpi=(300, 300))
    png_bytes = buffer.getvalue()

    # 2. SVG (just the cut line on padded viewbox)
    svg_path = polygon_to_svg_path(poly, new_height)
    svg_content = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {new_width} {new_height}" width="{new_width}" height="{new_height}">
    <path d="{svg_path}" fill="none" stroke="red" stroke-width="{thickness}"/>
</svg>'''

    # 3. DXF
    doc = ezdxf.new()
    msp = doc.modelspace()
    def add_poly_to_dxf(p):
        if not p.is_empty:
            coords = list(p.exterior.coords)
            msp.add_lwpolyline(coords, close=True, dxfattribs={'color': 1}) # 1 is red
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
    
    # Save the transparent PNG temporarily for reportlab
    import tempfile
    import os
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            img.save(tmp.name)
            tmp_path = tmp.name
        # ReportLab draws from bottom-left, which aligns with (pad_left, pad_bottom)
        c.drawImage(tmp_path, pad_left, pad_bottom, width, height, mask='auto')
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
    
    c.setStrokeColorRGB(1, 0, 0)
    c.setLineWidth(thickness)
    
    def draw_poly_on_pdf(p):
        if not p.is_empty:
            path = c.beginPath()
            coords = list(p.exterior.coords)
            if coords:
                # Invert Y axis for ReportLab (origin is bottom-left)
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
