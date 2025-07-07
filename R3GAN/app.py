import sys
import os
# ==== Sửa đường dẫn cho import module custom ====
sys.path.append(os.path.abspath("."))           # Thư mục gốc dự án
sys.path.append(os.path.abspath("./R3GAN"))     # Để import được torch_utils + R3GAN.R3GAN

import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import torch
import pickle
import json
import xml.etree.ElementTree as ET
from streamlit_drawable_canvas import st_canvas
import math

# ===== 1. Import Generator chuẩn từ R3GAN =====
from R3GAN.Networks import Generator

# ===== 2. Build Generator với đúng tham số =====
def build_generator():
    return Generator(
        InputChannels=1, 
        WidthPerStage=[256, 512, 512 , 256],
        CardinalityPerStage=[2, 2, 2, 2],
        BlocksPerStage=[6, 4, 4, 6],
        ExpansionFactor=2,
        KernelSize=3,
        ResamplingFilter=[1, 2, 1]
    )

# ===== 3. Load state_dict từ .pkl =====
def load_generator_state_dict(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    G_ema = data.get('G_ema', None)
    state_dict = None
    if G_ema is not None:
        if hasattr(G_ema, 'state_dict'):
            state_dict = G_ema.state_dict()
        elif isinstance(G_ema, dict) and 'state_dict' in G_ema:
            state_dict = G_ema['state_dict']
        elif hasattr(G_ema, '__dict__'):
            state_dict = G_ema.__dict__.get('state_dict', None)
    return state_dict

@st.cache_resource
def load_model():
    model_path = "network-snapshot-000000020.pkl"
    try:
        sd = load_generator_state_dict(model_path)
        if sd is None:
            st.error("❌ Không lấy được state_dict từ model.")
            return None
        gen = build_generator()
        gen.load_state_dict(sd, strict=False)
        gen.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gen = gen.to(device)
        st.success("✅ Model loaded thành công!")
        return gen
    except Exception as e:
        st.error(f"❌ Lỗi load model: {e}")
        return None

def preprocess_image_region(image_region):
    # Chuyển đổi ảnh về grayscale (1 kênh)
    if len(image_region.shape) == 3:
        # Nếu ảnh có 3 kênh (RGB), chuyển về grayscale
        image_region = cv2.cvtColor(image_region, cv2.COLOR_RGB2GRAY)
    
    # Resize về 256x256
    resized = cv2.resize(image_region, (256, 256), interpolation=cv2.INTER_CUBIC)
    
    # Normalize về [-1, 1]
    normalized = (resized.astype(np.float32) / 127.5) - 1.0
    
    # Tạo tensor với 1 kênh: [1, 1, 256, 256]
    tensor = torch.FloatTensor(normalized).unsqueeze(0).unsqueeze(0)
    
    return tensor

def postprocess_output(output_tensor, out_shape=None):
    output = (output_tensor.squeeze().detach().cpu().numpy() + 1.0) * 127.5
    output = np.clip(output, 0, 255).astype(np.uint8)
    if output.ndim == 3 and output.shape[0] == 3:
        output = np.transpose(output, (1, 2, 0))
    if out_shape is not None and (output.shape[0] != out_shape[0] or output.shape[1] != out_shape[1]):
        output = cv2.resize(output, (out_shape[1], out_shape[0]), interpolation=cv2.INTER_CUBIC)
    return output

def enhance_image_region(model, image_region):
    try:
        device = next(model.parameters()).device
        input_tensor = preprocess_image_region(image_region).to(device)
        with torch.no_grad():
            out = model(input_tensor)
        # Chuyển về đúng shape patch gốc
        enhanced_image = postprocess_output(out, out_shape=image_region.shape[:2])
        return enhanced_image
    except Exception as e:
        st.error(f"Lỗi khi enhance: {str(e)}")
        return None

def extract_32x32_patches(img, box, patch_size=32):
    """
    Trích xuất tất cả patches 32x32 từ một vùng lớn hơn
    """
    x1, y1, x2, y2 = box
    region_width = x2 - x1
    region_height = y2 - y1
    
    patches = []
    patch_positions = []
    
    # Tính số patches theo chiều ngang và dọc
    num_patches_x = math.ceil(region_width / patch_size)
    num_patches_y = math.ceil(region_height / patch_size)
    
    for i in range(num_patches_y):
        for j in range(num_patches_x):
            # Tính vị trí patch
            patch_x1 = x1 + j * patch_size
            patch_y1 = y1 + i * patch_size
            patch_x2 = min(patch_x1 + patch_size, x2)
            patch_y2 = min(patch_y1 + patch_size, y2)
            
            # Crop patch từ ảnh gốc
            patch = img[patch_y1:patch_y2, patch_x1:patch_x2]
            
            # Nếu patch nhỏ hơn 32x32, pad lại
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                patch = cv2.copyMakeBorder(
                    patch, 
                    0, patch_size - patch.shape[0], 
                    0, patch_size - patch.shape[1], 
                    borderType=cv2.BORDER_REFLECT
                )
            
            # Đảm bảo patch là grayscale (2D)
            if len(patch.shape) == 3:
                patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
            
            patches.append(patch)
            patch_positions.append((patch_x1, patch_y1, patch_x2, patch_y2))
    
    return patches, patch_positions

def reconstruct_enhanced_region(enhanced_patches, patch_positions, original_box, original_image_shape):
    """
    Ghép các patches đã tăng cường lại thành vùng hoàn chỉnh
    """
    x1, y1, x2, y2 = original_box
    region_width = x2 - x1
    region_height = y2 - y1
    
    # Tạo canvas để ghép patches (grayscale)
    reconstructed_region = np.zeros((region_height, region_width), dtype=np.uint8)
    
    for enhanced_patch, (px1, py1, px2, py2) in zip(enhanced_patches, patch_positions):
        # Tính vị trí tương đối trong vùng
        rel_x1 = px1 - x1
        rel_y1 = py1 - y1
        rel_x2 = px2 - x1
        rel_y2 = py2 - y1
        
        # Đảm bảo không vượt quá boundaries
        rel_x2 = min(rel_x2, region_width)
        rel_y2 = min(rel_y2, region_height)
        
        # Crop patch theo kích thước thực tế cần thiết
        actual_width = rel_x2 - rel_x1
        actual_height = rel_y2 - rel_y1
        
        if actual_width > 0 and actual_height > 0:
            cropped_patch = enhanced_patch[:actual_height, :actual_width]
            reconstructed_region[rel_y1:rel_y2, rel_x1:rel_x2] = cropped_patch
    
    return reconstructed_region

def calculate_display_size(orig_w, orig_h, max_size=600):
    """Tính toán kích thước hiển thị giữ nguyên tỷ lệ"""
    if orig_w > orig_h:
        if orig_w > max_size:
            display_w = max_size
            display_h = int(orig_h * max_size / orig_w)
        else:
            display_w = orig_w
            display_h = orig_h
    else:
        if orig_h > max_size:
            display_h = max_size
            display_w = int(orig_w * max_size / orig_h)
        else:
            display_w = orig_w
            display_h = orig_h
    return display_w, display_h

def extract_boxes(canvas_result, display_w, display_h, orig_w, orig_h):
    """Trích xuất tất cả bounding boxes (không giới hạn số lượng)"""
    boxes = []
    if canvas_result.json_data is not None:
        objects = canvas_result.json_data.get("objects", [])
        for obj in objects:
            if obj["type"] == "rect":
                left = int(obj["left"] * orig_w / display_w)
                top = int(obj["top"] * orig_h / display_h)
                width = int(obj["width"] * orig_w / display_w)
                height = int(obj["height"] * orig_h / display_h)
                x1, y1, x2, y2 = left, top, left+width, top+height
                boxes.append((x1, y1, x2, y2))
    return boxes

def validate_coordinates(x1, y1, x2, y2, img_width, img_height):
    """Kiểm tra tọa độ có hợp lệ không"""
    if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
        return False, "Tọa độ không được âm"
    if x1 >= x2 or y1 >= y2:
        return False, "x2 phải lớn hơn x1 và y2 phải lớn hơn y1"
    if x2 > img_width or y2 > img_height:
        return False, f"Tọa độ vượt quá kích thước ảnh ({img_width}x{img_height})"
    return True, "Tọa độ hợp lệ"

def visualize_boxes_on_image(image_np, boxes):
    """Vẽ tất cả bounding boxes lên ảnh"""
    img_vis = image_np.copy()
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_vis, f'Region {i+1}', (x1, max(10, y1-10)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return img_vis

# ========== MAIN STREAMLIT APP ==========
st.markdown('<h1 style="text-align:center">🩻 Demo Tăng Cường Chất Lượng Ảnh X-RAY bằng R3GAN</h1>', unsafe_allow_html=True)

# ===== MODEL STATUS =====
st.markdown("---")
st.header("🤖 Trạng thái Model")
model = load_model()

if not model:
    st.error("❌ Model chưa được load thành công")
    st.stop()

# ===== TABS =====
tab1, tab2 = st.tabs(["🎨 Khoanh Vùng Thủ Công", "📝 Nhập Tọa Độ"])

# ========== TAB 1: MANUAL ENHANCEMENT ==========
with tab1:
    st.header("📋 Hướng dẫn sử dụng")
    with st.expander("👉 Xem hướng dẫn chi tiết", expanded=False):
        st.markdown("""
        **Bước 1:** Upload ảnh X-ray của bạn
        
        **Bước 2:** Vẽ các vùng cần tăng cường
        
        **Bước 3:** Nhấn nút "🚀 Tăng Cường" để xử lý tất cả vùng
        
        **Lưu ý:** Nếu vùng lớn hơn 32x32, hệ thống sẽ tự động chia thành nhiều patch 32x32 để xử lý, sau đó ghép lại thành vùng hoàn chỉnh
        """)

    # ===== UPLOAD ẢNH =====
    st.markdown("---")
    st.header("📤 Upload ảnh X-ray")
    uploaded_file = st.file_uploader("Chọn ảnh X-ray", type=["png", "jpg", "jpeg", "bmp", "tiff"], key="manual_upload")

    if not uploaded_file:
        st.info("📁 Vui lòng upload ảnh X-ray để bắt đầu")
    else:
        # Load và hiển thị ảnh gốc
        image = Image.open(uploaded_file).convert("L")
        image_np = np.array(image)

        # Tính toán kích thước hiển thị giữ nguyên tỷ lệ
        orig_w, orig_h = image.size
        display_w, display_h = calculate_display_size(orig_w, orig_h)

        st.success(f"✅ Đã upload thành công! Kích thước gốc: {orig_w}x{orig_h}")
        st.info(f"📐 Kích thước hiển thị: {display_w}x{display_h}")

        # ===== VẼ VÙNG CẦN XỬ LÍ =====
        st.markdown("---")
        st.header("🖱️ Vẽ vùng cần xử lý")
        st.markdown("**Hướng dẫn:** Kéo chuột để tạo hình chữ nhật quanh các vùng bạn muốn tăng chất lượng.")

        # Canvas để vẽ bounding boxes
        canvas_result = st_canvas(
            fill_color="rgba(0, 255, 0, 0.2)",
            stroke_width=2,
            stroke_color="#00FF00",
            background_image=image,
            update_streamlit=True,
            height=display_h,
            width=display_w,
            drawing_mode="rect",
            key="manual_canvas",
        )

        # Hiển thị số lượng vùng đã chọn
        if canvas_result.json_data is not None:
            num_boxes = len([obj for obj in canvas_result.json_data.get("objects", []) if obj["type"] == "rect"])
            if num_boxes > 0:
                st.success(f"📍 Đã chọn {num_boxes} vùng để xử lý")
            else:
                st.info("✏️ Chưa có vùng nào được chọn. Vẽ hình chữ nhật trên ảnh để chọn vùng cần xử lý.")

        # ===== XỬ LÝ VÀ KẾT QUẢ =====
        st.markdown("---")
        st.header("🚀 Xử lý")

        if canvas_result and canvas_result.json_data is not None:
            boxes = extract_boxes(canvas_result, display_w, display_h, orig_w, orig_h)
            
            if not boxes:
                st.warning("⚠️ Vui lòng vẽ ít nhất một vùng trên ảnh để bắt đầu xử lý")
            else:
                # Hiển thị thông tin về patches sẽ được tạo
                total_patches = 0
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    region_width = x2 - x1
                    region_height = y2 - y1
                    num_patches = math.ceil(region_width / 32) * math.ceil(region_height / 32)
                    total_patches += num_patches
                    st.info(f"📊 Vùng {i+1}: {region_width}x{region_height} → {num_patches} patches 32x32")
                
                st.info(f"📋 Tổng cộng: {total_patches} patches sẽ được xử lý")
                
                # Nút xử lý
                if st.button("🚀 Tăng Cường", type="primary", use_container_width=True, key="manual_enhance"):
                    st.markdown("---")
                    st.subheader(f"🔄 Xử lý {len(boxes)} vùng...")
                    
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Container cho tất cả kết quả
                    results_container = st.container()
                    
                    with results_container:
                        processed_patches = 0
                        
                        # Xử lý từng vùng
                        for i, box in enumerate(boxes):
                            st.markdown(f"### 🔍 Vùng {i+1}")
                            x1, y1, x2, y2 = box
                            region_width = x2 - x1
                            region_height = y2 - y1
                            
                            # Trích xuất tất cả patches 32x32 từ vùng này
                            patches, patch_positions = extract_32x32_patches(image_np, box)
                            
                            st.info(f"📏 Kích thước vùng: {region_width}x{region_height}")
                            st.info(f"📦 Số patches: {len(patches)}")
                            
                            # Tăng cường từng patch
                            enhanced_patches = []
                            for patch_idx, patch in enumerate(patches):
                                # Update progress
                                progress = (processed_patches + 1) / total_patches
                                progress_bar.progress(progress)
                                status_text.text(f"Đang xử lý patch {processed_patches + 1}/{total_patches}...")
                                
                                # Enhancement
                                enhanced = enhance_image_region(model, patch)
                                if enhanced is not None:
                                    enhanced_patches.append(enhanced)
                                else:
                                    enhanced_patches.append(patch)  # Fallback to original if failed
                                
                                processed_patches += 1
                            
                            # Ghép lại vùng hoàn chỉnh
                            original_region = image_np[y1:y2, x1:x2]
                            enhanced_region = reconstruct_enhanced_region(enhanced_patches, patch_positions, box, image_np.shape)
                            
                            # Hiển thị kết quả so sánh
                            col1, col2 = st.columns(2)
                            with col1:
                                st.image(original_region, caption=f"Vùng gốc {i+1}", use_column_width=True)
                            with col2:
                                st.image(enhanced_region, caption=f"Vùng tăng cường {i+1}", use_column_width=True)
                            
                            # Hiển thị chi tiết patches nếu cần
                            with st.expander(f"Xem chi tiết patches vùng {i+1}", expanded=False):
                                cols_per_row = 4
                                for patch_idx, (original_patch, enhanced_patch, pos) in enumerate(zip(patches, enhanced_patches, patch_positions)):
                                    if patch_idx % cols_per_row == 0:
                                        cols = st.columns(cols_per_row)
                                    
                                    col_idx = patch_idx % cols_per_row
                                    with cols[col_idx]:
                                        st.markdown(f"**Patch {patch_idx + 1}**")
                                        st.image(original_patch, caption="Gốc", use_column_width=True)
                                        st.image(enhanced_patch, caption="Tăng cường", use_column_width=True)
                                        px1, py1, px2, py2 = pos
                                        st.caption(f"Vị trí: ({px1},{py1}) - ({px2},{py2})")
                            
                            # Thêm separator giữa các vùng
                            if i < len(boxes) - 1:
                                st.markdown("---")
                        
                        # Hoàn thành
                        progress_bar.progress(1.0)
                        status_text.text("✅ Hoàn thành xử lý tất cả vùng!")
                        st.success(f"🎉 Đã tăng cường thành công {total_patches} patches từ {len(boxes)} vùng!")
        else:
            st.info("✏️ Vẽ vùng cần tăng cường trên ảnh, sau đó nhấn nút Tăng Cường để xử lý.")

# ========== TAB 2: COORDINATE INPUT ==========
with tab2:
    st.header("📝 Nhập Tọa Độ Vùng")
    st.markdown("""
    Tính năng này cho phép bạn nhập chính xác tọa độ của các vùng cần tăng cường.
    Hệ thống sẽ tự động chia vùng lớn thành nhiều patches 32x32 để xử lý, sau đó ghép lại thành vùng hoàn chỉnh.
    """)
    
    # ===== UPLOAD ẢNH =====
    st.markdown("---")
    st.subheader("📁 Upload ảnh X-ray")
    coord_image = st.file_uploader("Chọn ảnh X-ray", 
                                 type=["png", "jpg", "jpeg", "bmp", "tiff"], 
                                 key="coord_image")
    
    if not coord_image:
        st.info("📁 Vui lòng upload ảnh X-ray để tiếp tục")
    else:
        # Load image
        image = Image.open(coord_image).convert("RGB")
        image_np = np.array(image)
        orig_w, orig_h = image.size
        display_w, display_h = calculate_display_size(orig_w, orig_h)
        
        st.success(f"✅ Đã upload ảnh: {coord_image.name} ({orig_w}x{orig_h})")
        
        # Hiển thị ảnh gốc
        st.image(image, caption="Ảnh gốc", use_column_width=True)
        
        # ===== QUẢN LÝ VÙNG =====
        st.markdown("---")
        st.subheader("📍 Quản lý vùng")
        
        # Initialize regions in session state với key unique cho tab này
        if 'coord_regions' not in st.session_state:
            st.session_state.coord_regions = []
        
        # Form để thêm vùng mới
        st.markdown("**Thêm vùng mới:**")
        col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 1])
        
        with col1:
            new_x1 = st.number_input("X1", min_value=0, max_value=orig_w-1, value=0, key="coord_new_x1")
        with col2:
            new_y1 = st.number_input("Y1", min_value=0, max_value=orig_h-1, value=0, key="coord_new_y1")
        with col3:
            new_x2 = st.number_input("X2", min_value=1, max_value=orig_w, value=min(32, orig_w), key="coord_new_x2")
        with col4:
            new_y2 = st.number_input("Y2", min_value=1, max_value=orig_h, value=min(32, orig_h), key="coord_new_y2")
        with col5:
            add_region_clicked = st.button("➕ Thêm", key="add_region_btn")
        
        if add_region_clicked:
            # Validate coordinates
            is_valid, message = validate_coordinates(new_x1, new_y1, new_x2, new_y2, orig_w, orig_h)
            
            if is_valid:
                st.session_state.coord_regions.append((new_x1, new_y1, new_x2, new_y2))
                st.success(f"✅ Đã thêm vùng: ({new_x1}, {new_y1}) - ({new_x2}, {new_y2})")
                # st.rerun()
            else:
                st.error(f"❌ {message}")
        
        # Hiển thị danh sách vùng hiện tại
        if st.session_state.coord_regions:
            st.markdown("**Danh sách vùng hiện tại:**")
            
            # Hiển thị ảnh với tất cả bounding boxes
            img_with_boxes = visualize_boxes_on_image(image_np, st.session_state.coord_regions)
            st.image(img_with_boxes, caption=f"Ảnh với {len(st.session_state.coord_regions)} vùng", use_column_width=True)
            
            # Hiển thị chi tiết từng vùng
            for i, (x1, y1, x2, y2) in enumerate(st.session_state.coord_regions):
                width = x2 - x1
                height = y2 - y1
                num_patches = math.ceil(width / 32) * math.ceil(height / 32)
                
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"**Vùng {i+1}:** ({x1}, {y1}) - ({x2}, {y2}) | "
                            f"Kích thước: {width}x{height} | "
                            f"Patches: {num_patches}")
                with col2:
                    if st.button("🗑️", key=f"coord_delete_{i}", help="Xóa vùng này"):
                        st.session_state.coord_regions.pop(i)
                        # st.rerun()
            
            # Tính tổng số patches
            total_patches = sum(math.ceil((x2-x1) / 32) * math.ceil((y2-y1) / 32) 
                              for x1, y1, x2, y2 in st.session_state.coord_regions)
            st.info(f"📊 Tổng cộng: {total_patches} patches sẽ được xử lý")
            
            # Nút xóa tất cả
            if st.button("🗑️ Xóa tất cả vùng", type="secondary", key="coord_clear_all"):
                st.session_state.coord_regions = []
                # st.rerun()
            
            # ===== XỬ LÝ TĂNG CƯỜNG =====
            st.markdown("---")
            st.subheader("🚀 Tăng cường chất lượng")
            
            if st.button("🚀 Tăng Cường Tất Cả Vùng", 
                        type="primary", use_container_width=True, key="coord_enhance"):
                
                st.markdown("---")
                st.subheader(f"🔄 Đang xử lý {len(st.session_state.coord_regions)} vùng...")
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Container cho tất cả kết quả
                results_container = st.container()
                
                with results_container:
                    processed_patches = 0
                    
                    # Xử lý từng vùng
                    for i, box in enumerate(st.session_state.coord_regions):
                        st.markdown(f"### 🔍 Vùng {i+1}")
                        x1, y1, x2, y2 = box
                        region_width = x2 - x1
                        region_height = y2 - y1
                        
                        # Trích xuất tất cả patches 32x32 từ vùng này
                        patches, patch_positions = extract_32x32_patches(image_np, box)
                        
                        st.info(f"📏 Kích thước vùng: {region_width}x{region_height}")
                        st.info(f"📦 Số patches: {len(patches)}")
                        
                        # Enhance từng patch
                        enhanced_patches = []
                        for patch_idx, patch in enumerate(patches):
                            progress = (processed_patches + 1) / total_patches
                            progress_bar.progress(progress)
                            status_text.text(f"Đang xử lý patch {processed_patches + 1}/{total_patches}...")
                            enhanced = enhance_image_region(model, patch)
                            enhanced_patches.append(enhanced if enhanced is not None else patch)
                            processed_patches += 1

                        # Ghép lại thành vùng lớn đã tăng cường
                        original_region = image_np[y1:y2, x1:x2]
                        enhanced_region = reconstruct_enhanced_region(enhanced_patches, patch_positions, box, image_np.shape)

                        # Hiển thị kết quả so sánh
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(original_region, caption=f"Vùng gốc {i+1}", use_column_width=True)
                        with col2:
                            st.image(enhanced_region, caption=f"Vùng tăng cường {i+1}", use_column_width=True)

                        # Xem chi tiết patch nếu cần
                        with st.expander(f"Xem chi tiết patches vùng {i+1}", expanded=False):
                            cols_per_row = 4
                            for patch_idx, (original_patch, enhanced_patch, pos) in enumerate(zip(patches, enhanced_patches, patch_positions)):
                                if patch_idx % cols_per_row == 0:
                                    cols = st.columns(cols_per_row)
                                col_idx = patch_idx % cols_per_row
                                with cols[col_idx]:
                                    st.markdown(f"**Patch {patch_idx + 1}**")
                                    st.image(original_patch, caption="Gốc", use_column_width=True)
                                    st.image(enhanced_patch, caption="Tăng cường", use_column_width=True)
                                    px1, py1, px2, py2 = pos
                                    st.caption(f"Vị trí: ({px1},{py1}) - ({px2},{py2})")

                        # Separator
                        if i < len(st.session_state.coord_regions) - 1:
                            st.markdown("---")

                    # Hoàn thành
                    progress_bar.progress(1.0)
                    status_text.text("✅ Đã hoàn thành tăng cường tất cả vùng!")
                    st.success(f"🎉 Đã tăng cường thành công {total_patches} patches từ {len(st.session_state.coord_regions)} vùng!")