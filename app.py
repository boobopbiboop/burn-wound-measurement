import streamlit as st
import cv2
import numpy as np
import json
from pathlib import Path
import pandas as pd
from PIL import Image
import io
import base64
from datetime import datetime

# =====================================================================
# PAGE CONFIG  
# =====================================================================
st.set_page_config(
    page_title="Burn Wound Detection & Measurement",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
# CUSTOM CSS
# =====================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4ECDC4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        background-color: #D4EDDA;
        border-left: 5px solid #28A745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .warning-box {
        background-color: #FFF3CD;
        border-left: 5px solid #FFC107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .info-box {
        background-color: #D1ECF1;
        border-left: 5px solid #0C5460;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# EXACT BATCH PROCESSING CODE
# =====================================================================

def detect_coin_for_calibration(image, coin_diameter_cm=2):
    """EXACT COPY dari batch processing"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, w = image.shape[:2]

    lower_gold = np.array([10, 100, 100])
    upper_gold = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower_gold, upper_gold)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_circle = None
    max_circularity = 0

    margin = 0.2
    corner_regions = [
        (0, 0, w*margin, h*margin),
        (w*(1-margin), 0, w, h*margin),
        (0, h*(1-margin), w*margin, h),
        (w*(1-margin), h*(1-margin), w, h)
    ]

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)

        if circularity > 0.7:
            (x, y), radius = cv2.minEnclosingCircle(contour)

            is_in_corner = False
            for x1, y1, x2, y2 in corner_regions:
                if x1 <= x <= x2 and y1 <= y <= y2:
                    is_in_corner = True
                    break

            score = circularity
            if is_in_corner:
                score *= 2

            if score > max_circularity:
                max_circularity = score
                best_circle = {
                    'center': (int(x), int(y)),
                    'radius': int(radius),
                    'diameter_px': int(radius * 2),
                    'circularity': circularity,
                    'in_corner': is_in_corner
                }

    if best_circle:
        pixels_per_cm = best_circle['diameter_px'] / coin_diameter_cm
        best_circle['pixels_per_cm'] = pixels_per_cm
        best_circle['diameter_cm'] = coin_diameter_cm
        return pixels_per_cm, best_circle

    return None, None


def grayscale_otsu_segmentation(image):
    """EXACT COPY dari batch - Binary Gray"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return gray, binary


def hsv_otsu_segmentation(image):
    """EXACT COPY dari batch - Binary HSV"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    blur_v = cv2.GaussianBlur(v, (5,5), 0)
    _, v_otsu = cv2.threshold(blur_v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    lower = np.array([0, 30, 50])
    upper = np.array([25, 255, 255])
    hsv_mask = cv2.inRange(hsv, lower, upper)

    combined = cv2.bitwise_and(v_otsu, hsv_mask)
    return combined


def morphological_processing(binary):
    """EXACT COPY dari batch - Opening, Closing, Hole Filling"""
    kernel = np.ones((5,5), np.uint8)

    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    floodfill = closing.copy()
    h, w = closing.shape
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(floodfill, mask, (0,0), 255)
    hole_filled = closing | cv2.bitwise_not(floodfill)

    return opening, closing, hole_filled


def contour_extraction(image, binary):
    """EXACT COPY dari batch - Contour"""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = image.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    return contour_img, contours


def calculate_wound_area(binary_mask, pixels_per_cm):
    """EXACT COPY dari batch"""
    area_pixels = np.sum(binary_mask > 0)

    if pixels_per_cm is None or pixels_per_cm == 0:
        pixels_per_cm = 50

    area_cm2 = area_pixels / (pixels_per_cm ** 2)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        perimeter_px = cv2.arcLength(largest_contour, True)
        perimeter_cm = perimeter_px / pixels_per_cm

        x, y, w, h = cv2.boundingRect(largest_contour)
        width_cm = w / pixels_per_cm
        height_cm = h / pixels_per_cm
    else:
        perimeter_cm = 0
        width_cm = 0
        height_cm = 0

    area_info = {
        'area_pixels': int(area_pixels),
        'area_cm2': round(area_cm2, 2),
        'perimeter_cm': round(perimeter_cm, 2),
        'width_cm': round(width_cm, 2),
        'height_cm': round(height_cm, 2),
        'pixels_per_cm': round(pixels_per_cm, 2)
    }

    return area_info


def calculate_wound_ratio(wound_area_cm2, reference_area_cm2=None):
    """EXACT COPY dari batch"""
    if reference_area_cm2 is None:
        reference_area_cm2 = np.pi * (1 ** 2)

    ratio = wound_area_cm2 / reference_area_cm2

    return {
        'wound_area_cm2': wound_area_cm2,
        'reference_area_cm2': round(reference_area_cm2, 2),
        'ratio': round(ratio, 2),
        'percentage': round(ratio * 100, 2)
    }


def create_visualization(original, mask, coin_info, area_info, ratio_info):
    """EXACT COPY dari batch"""
    result_img = original.copy()

    if coin_info:
        center = coin_info['center']
        radius = coin_info['radius']
        coin_color = (255, 0, 0) if coin_info.get('in_corner', False) else (0, 165, 255)

        cv2.circle(result_img, center, radius, coin_color, 2)
        cv2.putText(result_img, f"{coin_info['diameter_cm']}cm REF",
                   (center[0]-30, center[1]-radius-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, coin_color, 2)

    mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(result_img, 0.7, mask_colored, 0.3, 0)

    y_offset = 30
    texts = [
        f"Area: {area_info['area_cm2']} sq.cm",
        f"Perimeter: {area_info['perimeter_cm']} cm",
        f"Size: {area_info['width_cm']}x{area_info['height_cm']} cm",
        f"Ratio: {ratio_info['ratio']}x ({ratio_info['percentage']}%)"
    ]

    if coin_info is None:
        texts.insert(0, "WARNING: No coin detected")
        text_color = (0, 0, 255)
    elif not coin_info.get('in_corner', False):
        texts.insert(0, "WARNING: Coin in center")
        text_color = (0, 165, 255)
    else:
        text_color = (0, 255, 0)

    for i, text in enumerate(texts):
        y_pos = y_offset + (i * 30)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        
        cv2.rectangle(overlay, (5, y_pos - 22), (15 + text_size[0], y_pos + 5), (0, 0, 0), -1)
        cv2.putText(overlay, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

    return overlay


def get_image_download_link(img, filename):
    """Generate download link"""
    buffered = io.BytesIO()
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/png;base64,{img_str}" download="{filename}">Download {filename}</a>'
    return href


# =====================================================================
# STREAMLIT APP
# =====================================================================

def main():
    st.markdown('<div class="main-header">🔥 Burn Wound Detection & Measurement</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Flexible Mode: Full Pipeline or Batch Exact</div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/medical-heart.png", width=100)
        st.title("⚙️ Settings")
        
        st.subheader("🎯 Processing Mode")
        processing_mode = st.radio(
            "Pilih mode:",
            ["📸 Full Pipeline (1 File)", "🎯 Exact Batch (2 Files)"],
            help="Full Pipeline = upload 1 gambar, segmentasi real-time\nExact Batch = upload original + mask untuk hasil yang sama persis"
        )
        
        # Settings untuk Full Pipeline
        if processing_mode == "📸 Full Pipeline (1 File)":
            st.subheader("📋 Segmentation Method")
            seg_method = st.selectbox(
                "Pilih metode:",
                [
                    "Hole Filled (Recommended)",
                    "Binary HSV",
                    "Binary Gray",
                    "Opening",
                    "Closing"
                ],
                help="Hole Filled = hasil terbaik (sama dengan batch)"
            )
        
        st.subheader("🪙 Koin Referensi")
        coin_diameter = st.number_input(
            "Diameter koin (cm):",
            min_value=0.5,
            max_value=5.0,
            value=2.0,
            step=0.1
        )
        
        st.subheader("📊 Export")
        export_json = st.checkbox("Export JSON", value=True)
        export_csv = st.checkbox("Export CSV", value=True)
        
        st.subheader("🔧 Advanced")
        show_debug = st.checkbox("Show Debug Info", value=True)
        if processing_mode == "📸 Full Pipeline (1 File)":
            show_steps = st.checkbox("Show Intermediate Steps", value=False)
        
        st.markdown("---")
        st.info("💡 **Tip:**\n\nFull Pipeline = fleksibel, coba berbagai metode\n\nExact Batch = hasil sama persis dengan Colab")

    # Main Content
    if processing_mode == "📸 Full Pipeline (1 File)":
        # ====== MODE 1: FULL PIPELINE ======
        st.markdown('<div class="info-box">📸 <strong>Full Pipeline Mode</strong><br>Upload 1 gambar (harus ada koin), pilih metode segmentasi, dan lihat hasil real-time!</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "📤 Upload Image (dengan koin di sudut)",
            type=["jpg", "jpeg", "png"],
            help="Upload gambar dari folder augmented/ atau gambar lain yang sudah ada koin"
        )
        
        if uploaded_file is not None:
            # Read image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📷 Original Image")
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
                st.caption(f"Size: {image.shape[1]}x{image.shape[0]} → Will be resized to 512x512")
                st.caption(f"File: {uploaded_file.name}")
            
            if st.button("🚀 Process Image", type="primary", use_container_width=True):
                with st.spinner("Processing..."):
                    
                    # STEP 0: Resize to 512x512 (Match Batch Preprocessing)
                    original_size = image.shape[:2]
                    image_resized = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
                    
                    st.info(f"ℹ️ Image resized: {original_size[1]}x{original_size[0]} → 512x512 (matching batch preprocessing)")
                    
                    # STEP 1: Detect coin from RESIZED image
                    pixels_per_cm, coin_info = detect_coin_for_calibration(image_resized, coin_diameter)
                    
                    # STEP 2: Segmentation from RESIZED image
                    if seg_method == "Binary Gray":
                        gray, binary_gray = grayscale_otsu_segmentation(image)
                        final_mask = binary_gray
                        intermediate = {"gray": gray, "binary": binary_gray}
                        
                    elif seg_method == "Binary HSV":
                        binary_hsv = hsv_otsu_segmentation(image)
                        final_mask = binary_hsv
                        intermediate = {"binary_hsv": binary_hsv}
                        
                    elif seg_method == "Opening":
                        binary_hsv = hsv_otsu_segmentation(image)
                        opening, closing, hole_filled = morphological_processing(binary_hsv)
                        final_mask = opening
                        intermediate = {"binary_hsv": binary_hsv, "opening": opening}
                        
                    elif seg_method == "Closing":
                        binary_hsv = hsv_otsu_segmentation(image)
                        opening, closing, hole_filled = morphological_processing(binary_hsv)
                        final_mask = closing
                        intermediate = {"binary_hsv": binary_hsv, "opening": opening, "closing": closing}
                        
                    else:  # Hole Filled (Default/Recommended)
                        binary_hsv = hsv_otsu_segmentation(image)
                        opening, closing, hole_filled = morphological_processing(binary_hsv)
                        final_mask = hole_filled
                        intermediate = {"binary_hsv": binary_hsv, "opening": opening, "closing": closing, "hole_filled": hole_filled}
                    
                    # STEP 3: Calculate
                    area_info = calculate_wound_area(final_mask, pixels_per_cm)
                    
                    if coin_info:
                        coin_area_cm2 = np.pi * (coin_info['diameter_cm'] / 2) ** 2
                        ratio_info = calculate_wound_ratio(area_info['area_cm2'], coin_area_cm2)
                    else:
                        ratio_info = calculate_wound_ratio(area_info['area_cm2'])
                    
                    # STEP 4: Visualize (use RESIZED image for visualization)
                    result_img = create_visualization(image_resized, final_mask, coin_info, area_info, ratio_info)
                    
                    # Display results
                    with col2:
                        st.subheader(f"🎯 Result: {seg_method}")
                        st.image(cv2.cvtColor(final_mask, cv2.COLOR_GRAY2RGB), use_container_width=True)
                        st.caption(f"Non-zero pixels: {np.sum(final_mask > 0):,}")
                    
                    # Show intermediate steps
                    if show_steps and len(intermediate) > 1:
                        st.markdown("---")
                        st.subheader("🔬 Processing Steps")
                        
                        cols = st.columns(len(intermediate))
                        for idx, (name, img) in enumerate(intermediate.items()):
                            with cols[idx]:
                                st.image(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 
                                        caption=name.replace('_', ' ').title(), 
                                        use_container_width=True)
                    
                    # Debug info
                    if show_debug:
                        st.markdown("---")
                        with st.expander("🐛 Debug Info", expanded=True):
                            col_d1, col_d2 = st.columns(2)
                            
                            with col_d1:
                                st.markdown("**Coin Detection:**")
                                if coin_info:
                                    st.write(f"✅ Detected: Yes")
                                    st.write(f"Position: {'Corner ✅' if coin_info['in_corner'] else 'Center ⚠️'}")
                                    st.write(f"Diameter: {coin_info['diameter_px']} px")
                                    st.write(f"**Pixels/cm: {coin_info['pixels_per_cm']:.2f}**")
                                else:
                                    st.write(f"❌ Not detected (using default: 50)")
                            
                            with col_d2:
                                st.markdown("**Mask Info:**")
                                st.write(f"Image: 512x512 (resized)")
                                st.write(f"Method: {seg_method}")
                                st.write(f"Total pixels: {final_mask.size:,}")
                                st.write(f"**Non-zero: {np.sum(final_mask > 0):,}**")
                                st.write(f"Coverage: {np.sum(final_mask > 0)/final_mask.size*100:.2f}%")
                    
                    # Results section
                    st.markdown("---")
                    st.subheader("📊 Measurement Visualization")
                    st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                    
                    # Metrics
                    st.markdown("---")
                    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
                    col_m1.metric("Area (cm²)", f"{area_info['area_cm2']}")
                    col_m2.metric("Perimeter (cm)", f"{area_info['perimeter_cm']}")
                    col_m3.metric("Width (cm)", f"{area_info['width_cm']}")
                    col_m4.metric("Height (cm)", f"{area_info['height_cm']}")
                    col_m5.metric("Ratio", f"{ratio_info['ratio']}x")
                    
                    # Export
                    st.markdown("---")
                    st.subheader("💾 Export")
                    col_e1, col_e2, col_e3 = st.columns(3)
                    
                    with col_e1:
                        st.markdown(get_image_download_link(result_img, "result.png"), unsafe_allow_html=True)
                    
                    with col_e2:
                        if export_json:
                            result_json = {
                                'filename': uploaded_file.name,
                                'method': seg_method,
                                'timestamp': datetime.now().isoformat(),
                                'coin_info': coin_info,
                                'area_info': area_info,
                                'ratio_info': ratio_info
                            }
                            st.download_button("Download JSON", json.dumps(result_json, indent=2), "result.json", "application/json")
                    
                    with col_e3:
                        if export_csv:
                            csv_data = pd.DataFrame([{
                                'filename': uploaded_file.name,
                                'method': seg_method,
                                'area_cm2': area_info['area_cm2'],
                                'perimeter_cm': area_info['perimeter_cm'],
                                'width_cm': area_info['width_cm'],
                                'height_cm': area_info['height_cm'],
                                'pixels_per_cm': area_info['pixels_per_cm']
                            }])
                            st.download_button("Download CSV", csv_data.to_csv(index=False), "result.csv", "text/csv")
    
    else:
        # ====== MODE 2: EXACT BATCH ======
        st.markdown('<div class="info-box">🎯 <strong>Exact Batch Mode</strong><br>Upload 2 files untuk hasil yang 100% sama dengan batch processing!</div>', unsafe_allow_html=True)
        
        col_up1, col_up2 = st.columns(2)
        
        with col_up1:
            st.subheader("1️⃣ Original (augmented/)")
            original_file = st.file_uploader("Original image (dengan koin)", type=["jpg", "jpeg", "png"], key="orig")
        
        with col_up2:
            st.subheader("2️⃣ Mask (segmented/)")
            mask_file = st.file_uploader("Mask (hole_filled_*.jpg)", type=["jpg", "jpeg", "png"], key="mask")
        
        if original_file and mask_file:
            # Read images
            original = cv2.imdecode(np.asarray(bytearray(original_file.read()), dtype=np.uint8), cv2.IMREAD_COLOR)
            mask = cv2.imdecode(np.asarray(bytearray(mask_file.read()), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(cv2.cvtColor(original, cv2.COLOR_BGR2RGB), caption=f"Original: {original_file.name}", use_container_width=True)
            
            with col2:
                st.image(mask, caption=f"Mask: {mask_file.name}", use_container_width=True, clamp=True)
            
            if st.button("🚀 Calculate (Exact Batch)", type="primary", use_container_width=True):
                with st.spinner("Calculating..."):
                    
                    # Detect coin from original
                    pixels_per_cm, coin_info = detect_coin_for_calibration(original, coin_diameter)
                    
                    # Calculate from mask
                    area_info = calculate_wound_area(mask, pixels_per_cm)
                    
                    if coin_info:
                        coin_area_cm2 = np.pi * (coin_info['diameter_cm'] / 2) ** 2
                        ratio_info = calculate_wound_ratio(area_info['area_cm2'], coin_area_cm2)
                    else:
                        ratio_info = calculate_wound_ratio(area_info['area_cm2'])
                    
                    result_img = create_visualization(original, mask, coin_info, area_info, ratio_info)
                    
                    # Debug
                    if show_debug:
                        st.markdown("---")
                        with st.expander("🐛 Debug Info", expanded=True):
                            col_d1, col_d2 = st.columns(2)
                            
                            with col_d1:
                                st.markdown("**Coin:**")
                                if coin_info:
                                    st.write(f"✅ Detected")
                                    st.write(f"Pixels/cm: {coin_info['pixels_per_cm']:.2f}")
                                else:
                                    st.write(f"❌ Not detected")
                            
                            with col_d2:
                                st.markdown("**Mask:**")
                                st.write(f"Non-zero: {np.sum(mask > 0):,}")
                                st.write(f"Formula: {np.sum(mask > 0):,} / {area_info['pixels_per_cm']**2:.2f} = {area_info['area_cm2']}")
                    
                    # Results
                    st.markdown("---")
                    st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                    
                    st.markdown("---")
                    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
                    col_m1.metric("Area (cm²)", f"{area_info['area_cm2']}")
                    col_m2.metric("Perimeter (cm)", f"{area_info['perimeter_cm']}")
                    col_m3.metric("Width (cm)", f"{area_info['width_cm']}")
                    col_m4.metric("Height (cm)", f"{area_info['height_cm']}")
                    col_m5.metric("Ratio", f"{ratio_info['ratio']}x")
                    
                    # Export
                    st.markdown("---")
                    col_e1, col_e2, col_e3 = st.columns(3)
                    
                    with col_e1:
                        st.markdown(get_image_download_link(result_img, "result.png"), unsafe_allow_html=True)
                    
                    with col_e2:
                        if export_json:
                            result_json = {
                                'original': original_file.name,
                                'mask': mask_file.name,
                                'mode': 'exact_batch',
                                'coin_info': coin_info,
                                'area_info': area_info,
                                'ratio_info': ratio_info
                            }
                            st.download_button("Download JSON", json.dumps(result_json, indent=2), "result.json")
                    
                    with col_e3:
                        if export_csv:
                            csv_data = pd.DataFrame([{
                                'original': original_file.name,
                                'mask': mask_file.name,
                                'area_cm2': area_info['area_cm2'],
                                'perimeter_cm': area_info['perimeter_cm'],
                                'pixels_per_cm': area_info['pixels_per_cm']
                            }])
                            st.download_button("Download CSV", csv_data.to_csv(index=False), "result.csv")


if __name__ == "__main__":
    main()
