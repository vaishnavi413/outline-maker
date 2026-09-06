from math import floor, ceil
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

def extract_contours(image_bytes: bytes, min_area: float = 200.0, clean_lines: bool = True, disconnect_dist: int = 5) -> tuple:
    """
    Extracts base contours from an image (PNG or JPG).
    Cleans thin guide lines, registration boxes, and severs thin connecting bridges between pictures.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        raise ValueError("Invalid image file provided.")

    # Determine alpha / object mask
    if len(img.shape) < 3 or img.shape[2] != 4:
        # Grayscale threshold for JPGs with light/white background
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) >= 3 else img
        _, alpha = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
    else:
        alpha = img[:, :, 3]
        _, alpha = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)

    # 1. Morphological opening to erase thin 1-3px standalone lines & registration boxes
    if clean_lines:
        line_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, line_kernel)

    # 2. Bridge-Breaking: Erode to sever thin connecting lines/touching borders between pictures
    if disconnect_dist > 0:
        kernel_disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (disconnect_dist, disconnect_dist))
        eroded = cv2.erode(alpha, kernel_disc, iterations=1)
    else:
        eroded = alpha

    # Find external contours of separated components
    raw_contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    filtered_contours = []
    h, w = alpha.shape[:2]

    for cnt in raw_contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            aspect_ratio = float(bw) / bh if bh > 0 else 0
            if aspect_ratio < 20 and aspect_ratio > 0.05:
                # Create isolated mask for this component
                comp_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(comp_mask, [cnt], -1, 255, -1)
                
                # Dilate component back to restore full boundary
                if disconnect_dist > 0:
                    comp_mask = cv2.dilate(comp_mask, kernel_disc, iterations=1)
                    comp_mask = cv2.bitwise_and(comp_mask, alpha)
                
                # Find exact boundary contour of restored component
                c_list, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in c_list:
                    if cv2.contourArea(c) >= min_area:
                        filtered_contours.append(c)

    return filtered_contours, img.shape

def create_offset_contour(contours, offset_px: float, join_style: int = 1, smooth: bool = True, fill_holes: bool = True, separate_objects: bool = True):
    """
    Creates offset contours for each separate picture in the image.
    If separate_objects=True, offsets each picture independently so outlines stay separated per figure.
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
                        # Remove internal holes to create a solid backing for each picture
                        poly = Polygon(poly.exterior.coords)
                    polygons.append(poly)
    
    if not polygons:
        return Polygon()

    offset_polys = []
    for p in polygons:
        if offset_px > 0:
            op = p.buffer(offset_px, join_style=join_style, cap_style=1)
        else:
            op = p
        
        if smooth and op and not op.is_empty:
            op = op.simplify(1.5, preserve_topology=True)
            op = op.buffer(2.0, join_style=1).buffer(-2.0, join_style=1)
            
        if op and not op.is_empty:
            offset_polys.append(op)
            
    if not offset_polys:
        return Polygon()
        
    if separate_objects:
        # Keep each picture's offset contour distinct (MultiPolygon)
        return MultiPolygon(offset_polys) if len(offset_polys) > 1 else offset_polys[0]
    else:
        # Merge overlapping offset contours into a single unified shape
        return unary_union(offset_polys)

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
        
    # 1. Full Sticker PNG (Image + Background Fill + Stroke Line)
    bg_img = Image.fromarray(cv2.cvtColor(out_cv, cv2.COLOR_BGRA2RGBA))
    sticker_img = bg_img.copy()
    sticker_img.paste(img, (pad_left, pad_top), img)
    
    buf_full = io.BytesIO()
    sticker_img.save(buf_full, format="PNG", dpi=(300, 300))
    png_bytes = buf_full.getvalue()

    # 2. Border Only PNG (Background Fill + Stroke Line, NO Image inside)
    buf_border = io.BytesIO()
    bg_img.save(buf_border, format="PNG", dpi=(300, 300))
    border_png_bytes = buf_border.getvalue()

    # 3. Cut Line Only PNG (Transparent Canvas + Stroke Line ONLY)
    stroke_cv = np.zeros((new_height, new_width, 4), dtype=np.uint8)
    if isinstance(poly, MultiPolygon):
        for p in poly.geoms:
            draw_outline_poly(p, stroke_cv)
    elif isinstance(poly, Polygon):
        draw_outline_poly(poly, stroke_cv)
        
    stroke_img = Image.fromarray(cv2.cvtColor(stroke_cv, cv2.COLOR_BGRA2RGBA))
    buf_cut = io.BytesIO()
    stroke_img.save(buf_cut, format="PNG", dpi=(300, 300))
    cutline_png_bytes = buf_cut.getvalue()

    # 4. SVG (just the cut line on padded viewbox)
    stroke_hex = f"#{stroke_color[0]:02x}{stroke_color[1]:02x}{stroke_color[2]:02x}"
    svg_path = polygon_to_svg_path(poly, new_height)
    svg_content = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {new_width} {new_height}" width="{new_width}" height="{new_height}">
    <path d="{svg_path}" fill="none" stroke="{stroke_hex}" stroke-width="{thickness}"/>
</svg>'''

    # 5. DXF
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
    
    # 6. PDF Exports (A. Sticker Sheet PDF with Image, B. Outline ONLY PDF)
    pdf_full_bytes = create_pdf_bytes(
        poly, new_width, new_height, thickness, stroke_color,
        image_pil=img, pad_left=pad_left, pad_bottom=pad_bottom, img_w=width, img_h=height
    )
    
    pdf_outline_bytes = create_pdf_bytes(
        poly, new_width, new_height, thickness, stroke_color,
        image_pil=None
    )

    return {
        "png": png_bytes,
        "border_png": border_png_bytes,
        "cutline_png": cutline_png_bytes,
        "svg": svg_content.encode("utf-8"),
        "dxf": dxf_content.encode("utf-8"),
        "pdf": pdf_full_bytes,
        "outline_pdf": pdf_outline_bytes
    }

def create_pdf_bytes(poly, canvas_w: int, canvas_h: int, thickness: float, stroke_color=(0, 0, 0, 255), image_pil=None, pad_left: int = 0, pad_bottom: int = 0, img_w: int = 0, img_h: int = 0) -> bytes:
    """Helper to generate vector PDF exports with or without image overlay."""
    pdf_buffer = io.BytesIO()
    c = pdf_canvas.Canvas(pdf_buffer, pagesize=(canvas_w, canvas_h))
    
    if image_pil is not None:
        import tempfile
        import os
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                image_pil.save(tmp.name)
                tmp_path = tmp.name
            c.drawImage(tmp_path, pad_left, pad_bottom, img_w if img_w > 0 else canvas_w, img_h if img_h > 0 else canvas_h, mask='auto')
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
                path.moveTo(coords[0][0], canvas_h - coords[0][1])
                for x, y in coords[1:]:
                    path.lineTo(x, canvas_h - y)
                path.close()
                c.drawPath(path, stroke=1, fill=0)
            
            for interior in p.interiors:
                path = c.beginPath()
                coords = list(interior.coords)
                if coords:
                    path.moveTo(coords[0][0], canvas_h - coords[0][1])
                    for x, y in coords[1:]:
                        path.lineTo(x, canvas_h - y)
                    path.close()
                    c.drawPath(path, stroke=1, fill=0)
                    
    if isinstance(poly, MultiPolygon):
        for p in poly.geoms:
            draw_poly_on_pdf(p)
    elif isinstance(poly, Polygon):
        draw_poly_on_pdf(poly)
        
    c.save()
    return pdf_buffer.getvalue()

def extract_individual_sticker_exports(image_bytes: bytes, poly, thickness: float, fill_color=(255, 255, 255, 255), stroke_color=(0, 0, 0, 255)) -> list:
    """Crops each distinct picture/polygon and returns individual sticker exports."""
    from shapely.affinity import translate

    geoms = list(poly.geoms) if isinstance(poly, MultiPolygon) else [poly] if isinstance(poly, Polygon) and not poly.is_empty else []
    if not geoms:
        return []

    img_full = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    full_w, full_h = img_full.size

    individual_list = []
    
    for idx, p in enumerate(geoms):
        if p.is_empty:
            continue
            
        minx, miny, maxx, maxy = p.bounds
        pad = int(thickness / 2 + 10)
        
        # Bounding crop box for this picture
        crop_minx = int(max(0, floor(minx) - pad))
        crop_miny = int(max(0, floor(miny) - pad))
        crop_maxx = int(min(full_w, ceil(maxx) + pad))
        crop_maxy = int(min(full_h, ceil(maxy) + pad))
        
        crop_w = crop_maxx - crop_minx
        crop_h = crop_maxy - crop_miny
        
        if crop_w <= 0 or crop_h <= 0:
            continue
            
        # Crop original artwork for this picture
        img_cropped = img_full.crop((crop_minx, crop_miny, crop_maxx, crop_maxy))
        
        # Translate polygon relative to crop origin
        p_cropped = translate(p, xoff=-crop_minx, yoff=-crop_miny)
        
        # Draw background fill & stroke line on crop canvas
        out_cv = np.zeros((crop_h, crop_w, 4), dtype=np.uint8)
        stroke_cv = np.zeros((crop_h, crop_w, 4), dtype=np.uint8)
        
        fill_bgra = (fill_color[2], fill_color[1], fill_color[0], fill_color[3])
        stroke_bgra = (stroke_color[2], stroke_color[1], stroke_color[0], stroke_color[3])

        coords = np.array(p_cropped.exterior.coords, dtype=np.int32)
        cv2.fillPoly(out_cv, [coords], fill_bgra)
        cv2.polylines(out_cv, [coords], True, stroke_bgra, int(max(1, thickness)), lineType=cv2.LINE_AA)
        cv2.polylines(stroke_cv, [coords], True, stroke_bgra, int(max(1, thickness)), lineType=cv2.LINE_AA)

        for interior in p_cropped.interiors:
            icoords = np.array(interior.coords, dtype=np.int32)
            cv2.fillPoly(out_cv, [icoords], (0, 0, 0, 0))
            cv2.polylines(out_cv, [icoords], True, stroke_bgra, int(max(1, thickness)), lineType=cv2.LINE_AA)
            cv2.polylines(stroke_cv, [icoords], True, stroke_bgra, int(max(1, thickness)), lineType=cv2.LINE_AA)

        # Full Sticker (Picture + Border)
        bg_pil = Image.fromarray(cv2.cvtColor(out_cv, cv2.COLOR_BGRA2RGBA))
        border_only_pil = bg_pil.copy()
        
        full_sticker_pil = bg_pil.copy()
        full_sticker_pil.paste(img_cropped, (0, 0), img_cropped)
        
        # Cut Line Only
        stroke_pil = Image.fromarray(cv2.cvtColor(stroke_cv, cv2.COLOR_BGRA2RGBA))

        # Save bytes
        b_full = io.BytesIO()
        full_sticker_pil.save(b_full, format="PNG", dpi=(300, 300))
        
        b_border = io.BytesIO()
        border_only_pil.save(b_border, format="PNG", dpi=(300, 300))
        
        b_stroke = io.BytesIO()
        stroke_pil.save(b_stroke, format="PNG", dpi=(300, 300))

        # PDF exports for individual picture (Outline ONLY PDF and Full Sticker PDF)
        ind_pdf_outline = create_pdf_bytes(
            p_cropped, crop_w, crop_h, thickness, stroke_color,
            image_pil=None
        )
        ind_pdf_full = create_pdf_bytes(
            p_cropped, crop_w, crop_h, thickness, stroke_color,
            image_pil=img_cropped
        )

        # Individual SVG path
        svg_p = polygon_to_svg_path(p_cropped, crop_h)
        stroke_hex = f"#{stroke_color[0]:02x}{stroke_color[1]:02x}{stroke_color[2]:02x}"
        svg_ind = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {crop_w} {crop_h}" width="{crop_w}" height="{crop_h}">
    <path d="{svg_p}" fill="none" stroke="{stroke_hex}" stroke-width="{thickness}"/>
</svg>'''

        individual_list.append({
            "index": idx + 1,
            "full_png": b_full.getvalue(),
            "border_png": b_border.getvalue(),
            "cutline_png": b_stroke.getvalue(),
            "svg": svg_ind.encode("utf-8"),
            "pdf": ind_pdf_full,
            "outline_pdf": ind_pdf_outline
        })

    return individual_list
