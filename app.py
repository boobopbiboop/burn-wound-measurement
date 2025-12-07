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
    .error-box {
        background-color: #F8D7DA;
        border-left: 5px solid #DC3545;
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


def calculate_wound_area(binary_mask, pixels_per_cm):
    """EXACT COPY dari batch processing"""
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
    """EXACT COPY dari batch processing"""
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
        texts.insert(0, "WARNING: No coin detected (using default scale)")
        text_color = (0, 0, 255)
    elif not coin_info.get('in_corner', False):
        texts.insert(0, "WARNING: Coin detected in center (may be inaccurate)")
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
    """Generate download link for image"""
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
    st.markdown('<div class="sub-header">100% EXACT Batch Processing - Upload 2 Files</div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/medical-heart.png", width=100)
        st.title("⚙️ Settings")
        
        st.subheader("🪙 Koin Referensi")
        coin_diameter = st.number_input(
            "Diameter koin (cm):",
            min_value=0.5,
            max_value=5.0,
            value=2.0,
            step=0.1,
            help="Diameter koin yang digunakan sebagai referensi"
        )
        
        st.subheader("📊 Export Options")
        export_json = st.checkbox("Export JSON", value=True)
        export_csv = st.checkbox("Export CSV", value=True)
        
        st.subheader("🔧 Debug")
        show_debug = st.checkbox("Show Debug Info", value=True)
        
        st.markdown("---")
        st.success("✅ **100% EXACT Batch!**\n\nUpload 2 files untuk hasil yang sama!")

    # IMPORTANT INFO BOX
    st.markdown('<div class="info-box"><strong>📋 CARA PAKAI (EXACT seperti Batch):</strong><br>1. Upload <strong>Original Image</strong> dari folder <code>augmented/</code> (yang ada koin)<br>2. Upload <strong>Mask Image</strong> dari folder <code>segmented/</code> (file <code>hole_filled_*.jpg</code>)<br><br><strong>⚠️ PENTING:</strong> Folder <code>segmented/</code> punya 59k files (7 jenis output). Pastikan upload yang <code>hole_filled_*.jpg</code> SAJA!<br><br>Nama file harus sama! Contoh:<br>- Original: <code>burn_wound_0001.jpg</code><br>- Mask: <code>hole_filled_burn_wound_0001.jpg</code></div>', unsafe_allow_html=True)

    # Upload Section
    col_up1, col_up2 = st.columns(2)
    
    with col_up1:
        st.subheader("1️⃣ Upload Original (from augmented/)")
        original_file = st.file_uploader(
            "Original Image (dengan koin)",
            type=["jpg", "jpeg", "png"],
            key="original",
            help="Upload dari folder augmented/"
        )
    
    with col_up2:
        st.subheader("2️⃣ Upload Mask (from segmented/)")
        mask_file = st.file_uploader(
            "Mask Image (hole_filled_*.jpg)",
            type=["jpg", "jpeg", "png"],
            key="mask",
            help="Upload dari folder segmented/ - file hole_filled_*.jpg"
        )

    # Process if both files uploaded
    if original_file is not None and mask_file is not None:
        # Read images
        original_bytes = np.asarray(bytearray(original_file.read()), dtype=np.uint8)
        original_image = cv2.imdecode(original_bytes, cv2.IMREAD_COLOR)
        
        mask_bytes = np.asarray(bytearray(mask_file.read()), dtype=np.uint8)
        mask_image = cv2.imdecode(mask_bytes, cv2.IMREAD_GRAYSCALE)
        
        # Display uploaded images
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Original Image")
            st.image(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.caption(f"Size: {original_image.shape[1]}x{original_image.shape[0]} | File: {original_file.name}")
        
        with col2:
            st.subheader("🎭 Mask Image")
            st.image(mask_image, use_container_width=True, clamp=True)
            st.caption(f"Size: {mask_image.shape[1]}x{mask_image.shape[0]} | File: {mask_file.name}")
        
        # Check if sizes match - AUTO RESIZE if needed
        if original_image.shape[:2] != mask_image.shape[:2]:
            st.warning(f"⚠️ Ukuran berbeda! Original: {original_image.shape[:2]}, Mask: {mask_image.shape[:2]}")
            st.info("🔧 Auto-resizing mask ke ukuran original...")
            mask_image = cv2.resize(mask_image, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Check if filenames match
        original_name = original_file.name
        mask_name = mask_file.name.replace("hole_filled_", "")
        
        if original_name != mask_name:
            st.warning(f"⚠️ Warning: Nama file tidak match!\nOriginal: {original_name}\nMask: {mask_file.name}\n\nPastikan mask untuk gambar yang benar!")
        
        if st.button("🚀 Process (EXACT Batch Method)", type="primary", use_container_width=True):
            with st.spinner("Processing with EXACT batch code..."):
                
                # STEP 1: Detect coin from ORIGINAL (EXACT sama batch)
                pixels_per_cm, coin_info = detect_coin_for_calibration(original_image, coin_diameter)
                
                # STEP 2: Calculate area from MASK (EXACT sama batch)
                area_info = calculate_wound_area(mask_image, pixels_per_cm)
                
                # STEP 3: Calculate ratio
                if coin_info:
                    coin_area_cm2 = np.pi * (coin_info['diameter_cm'] / 2) ** 2
                    ratio_info = calculate_wound_ratio(area_info['area_cm2'], coin_area_cm2)
                else:
                    ratio_info = calculate_wound_ratio(area_info['area_cm2'])
                
                # STEP 4: Visualization
                result_img = create_visualization(original_image, mask_image, coin_info, area_info, ratio_info)
                
                # Debug info
                if show_debug:
                    st.markdown("---")
                    with st.expander("🐛 Debug Information", expanded=True):
                        col_d1, col_d2, col_d3 = st.columns(3)
                        
                        with col_d1:
                            st.markdown("**Image Info:**")
                            st.write(f"- Original: {original_image.shape[1]}x{original_image.shape[0]}")
                            st.write(f"- Mask: {mask_image.shape[1]}x{mask_image.shape[0]}")
                            st.write(f"- Match: {'✅' if original_image.shape[:2] == mask_image.shape[:2] else '❌'}")
                        
                        with col_d2:
                            st.markdown("**Coin Detection:**")
                            if coin_info:
                                st.write(f"- ✅ Detected: Yes")
                                st.write(f"- Position: {'Corner' if coin_info['in_corner'] else 'Center'}")
                                st.write(f"- Diameter: {coin_info['diameter_px']} px")
                                st.write(f"- **Pixels/cm: {coin_info['pixels_per_cm']:.2f}**")
                            else:
                                st.write(f"- ❌ Detected: No")
                                st.write(f"- Using default: 50 px/cm")
                        
                        with col_d3:
                            st.markdown("**Mask Info:**")
                            st.write(f"- Total pixels: {mask_image.size:,}")
                            st.write(f"- **Non-zero: {np.sum(mask_image > 0):,}**")
                            st.write(f"- Zero: {np.sum(mask_image == 0):,}")
                            st.write(f"- Percentage: {np.sum(mask_image > 0)/mask_image.size*100:.2f}%")
                        
                        st.markdown("---")
                        st.markdown("**📐 Calculation Formula:**")
                        st.code(f"""
area_pixels = {area_info['area_pixels']:,}
pixels_per_cm = {area_info['pixels_per_cm']}
area_cm2 = area_pixels / (pixels_per_cm)²
area_cm2 = {area_info['area_pixels']:,} / ({area_info['pixels_per_cm']})²
area_cm2 = {area_info['area_pixels']:,} / {area_info['pixels_per_cm']**2:.2f}
area_cm2 = {area_info['area_cm2']}
                        """)
                
                st.markdown("---")
                
                # Display measurement visualization
                st.subheader("📊 Measurement Visualization")
                st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                # Coin detection status
                if coin_info is None:
                    st.markdown('<div class="error-box">❌ <strong>KOIN TIDAK TERDETEKSI!</strong><br>Menggunakan skala default (50 px/cm). Pastikan upload gambar dari folder augmented/ yang ada koin nya!</div>', unsafe_allow_html=True)
                elif not coin_info.get('in_corner', False):
                    st.markdown('<div class="warning-box">⚠️ <strong>Koin terdeteksi di tengah gambar!</strong><br>Hasil mungkin kurang akurat.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="success-box">✅ <strong>Koin terdeteksi dengan baik di sudut!</strong><br>Hasil SAMA dengan batch processing! 🎯</div>', unsafe_allow_html=True)
                
                # Metrics
                st.markdown("---")
                st.subheader("📏 Measurement Results")
                
                col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
                
                with col_m1:
                    st.metric("Area (cm²)", f"{area_info['area_cm2']}")
                
                with col_m2:
                    st.metric("Perimeter (cm)", f"{area_info['perimeter_cm']}")
                
                with col_m3:
                    st.metric("Width (cm)", f"{area_info['width_cm']}")
                
                with col_m4:
                    st.metric("Height (cm)", f"{area_info['height_cm']}")
                
                with col_m5:
                    st.metric("Ratio", f"{ratio_info['ratio']}x")
                
                # Detailed table
                st.markdown("---")
                st.subheader("📋 Detailed Information")
                
                col_t1, col_t2, col_t3 = st.columns(3)
                
                with col_t1:
                    st.markdown("**Wound Measurements**")
                    wound_data = {
                        "Metric": ["Area (pixels)", "Area (cm²)", "Perimeter (cm)", "Width (cm)", "Height (cm)"],
                        "Value": [f"{area_info['area_pixels']:,}", area_info['area_cm2'], area_info['perimeter_cm'], area_info['width_cm'], area_info['height_cm']]
                    }
                    st.dataframe(pd.DataFrame(wound_data), use_container_width=True, hide_index=True)
                
                with col_t2:
                    st.markdown("**Calibration Info**")
                    if coin_info:
                        calib_data = {
                            "Parameter": ["Coin Detected", "Position", "Diameter (px)", "Pixels per cm"],
                            "Value": ["Yes", "Corner" if coin_info['in_corner'] else "Center", coin_info['diameter_px'], round(coin_info['pixels_per_cm'], 2)]
                        }
                    else:
                        calib_data = {
                            "Parameter": ["Coin Detected", "Pixels per cm"],
                            "Value": ["No", "50 (default)"]
                        }
                    st.dataframe(pd.DataFrame(calib_data), use_container_width=True, hide_index=True)
                
                with col_t3:
                    st.markdown("**Ratio Info**")
                    ratio_data = {
                        "Parameter": ["Wound Area", "Reference Area", "Ratio", "Percentage"],
                        "Value": [f"{ratio_info['wound_area_cm2']} cm²", f"{ratio_info['reference_area_cm2']} cm²", f"{ratio_info['ratio']}x", f"{ratio_info['percentage']}%"]
                    }
                    st.dataframe(pd.DataFrame(ratio_data), use_container_width=True, hide_index=True)
                
                # Export section
                st.markdown("---")
                st.subheader("💾 Export Results")
                
                col_e1, col_e2, col_e3 = st.columns(3)
                
                with col_e1:
                    st.markdown(get_image_download_link(result_img, "measurement_result.png"), unsafe_allow_html=True)
                
                with col_e2:
                    if export_json:
                        result_json = {
                            'original_file': original_file.name,
                            'mask_file': mask_file.name,
                            'timestamp': datetime.now().isoformat(),
                            'coin_detected': coin_info is not None,
                            'coin_info': coin_info,
                            'area_info': area_info,
                            'ratio_info': ratio_info
                        }
                        json_str = json.dumps(result_json, indent=2)
                        st.download_button(label="Download JSON", data=json_str, file_name="measurement_result.json", mime="application/json")
                
                with col_e3:
                    if export_csv:
                        csv_data = {
                            'original_file': [original_file.name],
                            'mask_file': [mask_file.name],
                            'coin_detected': [coin_info is not None],
                            'area_pixels': [area_info['area_pixels']],
                            'area_cm2': [area_info['area_cm2']],
                            'perimeter_cm': [area_info['perimeter_cm']],
                            'width_cm': [area_info['width_cm']],
                            'height_cm': [area_info['height_cm']],
                            'pixels_per_cm': [area_info['pixels_per_cm']],
                            'ratio': [ratio_info['ratio']],
                            'percentage': [ratio_info['percentage']]
                        }
                        df = pd.DataFrame(csv_data)
                        csv = df.to_csv(index=False)
                        st.download_button(label="Download CSV", data=csv, file_name="measurement_result.csv", mime="text/csv")

    else:
        st.info("👆 Upload 2 files untuk memulai (Original + Mask)")
        
        st.markdown("### 📖 Struktur Folder Batch Processing:")
        
        st.code("""
BurnDetection_ProcessedDataset/
├── augmented/              ← Upload FILE 1 dari sini (8,464 files)
│   ├── burn_wound_0001.jpg
│   ├── burn_wound_0002.jpg
│   └── ...
│
├── segmented/              ← Upload FILE 2 dari sini (59,248 files)
│   ├── gray_*.jpg          ← ❌ JANGAN ini
│   ├── binary_gray_*.jpg   ← ❌ JANGAN ini
│   ├── binary_hsv_*.jpg    ← ❌ JANGAN ini
│   ├── opening_*.jpg       ← ❌ JANGAN ini
│   ├── closing_*.jpg       ← ❌ JANGAN ini
│   ├── hole_filled_*.jpg   ← ✅ UPLOAD INI SAJA!
│   └── contour_*.jpg       ← ❌ JANGAN ini
│
└── measured/               ← Hasil batch (untuk compare)
    ├── measured_burn_wound_0001.jpg
    └── measurement_results.csv
        """)
        
        st.success("""
        **✅ Step-by-Step:**
        1. Buka folder `augmented/` → download `burn_wound_0001.jpg`
        2. Buka folder `segmented/` → cari dan download `hole_filled_burn_wound_0001.jpg`
        3. Upload kedua file di atas
        4. Klik "Process"
        5. **Hasil PASTI SAMA dengan batch!** 🎯
        
        ⚠️ **JANGAN salah pilih file di segmented/** - ada 7 jenis file per gambar!
        """)


if __name__ == "__main__":
    main()