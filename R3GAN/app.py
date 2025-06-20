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

# ===== 1. Import Generator chuẩn từ R3GAN =====
from R3GAN.Networks import Generator

# ===== 2. Build Generator với đúng tham số =====
def build_generator():
    return Generator(
        InputChannels=3,
        WidthPerStage=[32, 64, 64, 32],
        CardinalityPerStage=[2, 4, 4, 2],
        BlocksPerStage=[1, 1, 1, 1],
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
    model_path = "network-snapshot-000000008.pkl"
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
    # Nếu ảnh là 2D (grayscale), convert sang RGB (3 kênh)
    if len(image_region.shape) == 2:
        image_region = cv2.cvtColor(image_region, cv2.COLOR_GRAY2RGB)
    # Nếu là BGR (do OpenCV), đổi về RGB
    if image_region.shape[2] == 3:
        pass  # giữ nguyên
    resized = cv2.resize(image_region, (256,256), interpolation=cv2.INTER_CUBIC)
    # Normalize về [-1, 1]
    normalized = (resized.astype(np.float32) / 127.5) - 1.0
    tensor = torch.FloatTensor(normalized).permute(2, 0, 1).unsqueeze(0)  # [1, 3, 256, 256]
    return tensor

def postprocess_output(output_tensor):
    output = (output_tensor.squeeze().detach().cpu().numpy() + 1.0) * 127.5
    output = np.clip(output, 0, 255).astype(np.uint8)
    # Đảm bảo output shape là (H, W, 3) nếu có 3 kênh
    if output.ndim == 3 and output.shape[0] == 3:
        output = np.transpose(output, (1, 2, 0))
    return output

def enhance_image_region(model, image_region):
    try:
        device = next(model.parameters()).device
        input_tensor = preprocess_image_region(image_region).to(device)
        with torch.no_grad():
            out = model(input_tensor)
        enhanced_image = postprocess_output(out)
        return enhanced_image
    except Exception as e:
        st.error(f"Lỗi khi enhance: {str(e)}")
        return None

def crop_center(img, box, crop_size=32):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    h, w = img.shape[:2]
    half = crop_size // 2
    left = min(max(cx - half, 0), w - crop_size)
    top = min(max(cy - half, 0), h - crop_size)
    # Đảm bảo không out of bounds
    left = max(0, left)
    top = max(0, top)
    right = min(left + crop_size, w)
    bottom = min(top + crop_size, h)
    crop = img[top:bottom, left:right]
    # Nếu thiếu shape, pad lại
    if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
        crop = cv2.copyMakeBorder(
            crop, 0, crop_size - crop.shape[0], 0, crop_size - crop.shape[1], borderType=cv2.BORDER_REFLECT)
    # Nếu còn thiếu kênh màu, convert sang RGB
    if len(crop.shape) == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
    elif crop.shape[2] == 1:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
    return crop

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

# ===== ANNOTATION PARSING FUNCTIONS =====
def parse_csv_annotation(csv_file):
    """
    Parse CSV annotation file - Đúng cấu trúc: 
    rad_ID, class_name, x_min, y_min, x_max, y_max, class_id
    """
    try:
        df = pd.read_csv(csv_file)
        df.columns = df.columns.str.strip()
        required_cols = ['rad_ID', 'class_name', 'x_min', 'y_min', 'x_max', 'y_max', 'class_id']
        for col in required_cols:
            if col not in df.columns:
                return None, f"Thiếu cột bắt buộc: {col}"
        boxes = []
        classes = []
        for _, row in df.iterrows():
            x1 = int(float(row['x_min']))
            y1 = int(float(row['y_min']))
            x2 = int(float(row['x_max']))
            y2 = int(float(row['y_max']))
            boxes.append((x1, y1, x2, y2))
            # Lưu class_name/class_id để hiển thị nếu cần
            classes.append({
                "rad_ID": row['rad_ID'],
                "class_name": row['class_name'],
                "class_id": row['class_id']
            })
        return (boxes, classes), f"Tìm thấy {len(boxes)} annotation(s)"
    except Exception as e:
        return None, f"Lỗi parse CSV: {str(e)}"


def parse_json_annotation(json_file, image_filename):
    """Parse JSON annotation file - supports COCO, custom JSON formats"""
    try:
        data = json.load(json_file)
        
        # COCO format
        if 'annotations' in data and 'images' in data:
            # Find image ID
            image_id = None
            for img in data['images']:
                if image_filename in img['file_name']:
                    image_id = img['id']
                    break
            
            if image_id is None:
                return None, f"Không tìm thấy ảnh {image_filename} trong COCO JSON"
            
            # Get annotations for this image
            boxes = []
            for ann in data['annotations']:
                if ann['image_id'] == image_id:
                    bbox = ann['bbox']  # [x, y, width, height]
                    x, y, w, h = bbox
                    boxes.append((int(x), int(y), int(x + w), int(y + h)))
            
            return boxes, f"Tìm thấy {len(boxes)} annotation(s) (COCO format)"
        
        # Custom JSON format (assume direct structure)
        elif image_filename in data or 'annotations' in data:
            boxes = []
            annotations = data.get(image_filename, data.get('annotations', []))
            
            for ann in annotations:
                if 'bbox' in ann:
                    bbox = ann['bbox']
                    if len(bbox) == 4:
                        x, y, w, h = bbox
                        boxes.append((int(x), int(y), int(x + w), int(y + h)))
                elif all(key in ann for key in ['x', 'y', 'width', 'height']):
                    x, y, w, h = ann['x'], ann['y'], ann['width'], ann['height']
                    boxes.append((int(x), int(y), int(x + w), int(y + h)))
            
            return boxes, f"Tìm thấy {len(boxes)} annotation(s) (Custom JSON)"
        
        return None, "Format JSON không được hỗ trợ"
        
    except Exception as e:
        return None, f"Lỗi parse JSON: {str(e)}"

def parse_xml_annotation(xml_file, image_filename):
    """Parse XML annotation file - supports PASCAL VOC format"""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Check if this XML is for the right image
        filename_elem = root.find('filename')
        if filename_elem is not None and image_filename not in filename_elem.text:
            return None, f"XML không khớp với ảnh {image_filename}"
        
        boxes = []
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            if bbox is not None:
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                boxes.append((xmin, ymin, xmax, ymax))
        
        return boxes, f"Tìm thấy {len(boxes)} annotation(s) (PASCAL VOC)"
        
    except Exception as e:
        return None, f"Lỗi parse XML: {str(e)}"

def parse_txt_annotation(txt_file, image_filename, img_width, img_height):
    """Parse TXT annotation file - supports YOLO format"""
    try:
        lines = txt_file.read().decode('utf-8').strip().split('\n')
        
        boxes = []
        for line in lines:
            if line.strip():
                parts = line.strip().split()
                if len(parts) >= 5:
                    # YOLO format: class_id center_x center_y width height (normalized)
                    _, cx, cy, w, h = parts[:5]
                    cx, cy, w, h = float(cx), float(cy), float(w), float(h)
                    
                    # Convert from normalized to absolute coordinates
                    x1 = int((cx - w/2) * img_width)
                    y1 = int((cy - h/2) * img_height)
                    x2 = int((cx + w/2) * img_width)
                    y2 = int((cy + h/2) * img_height)
                    
                    boxes.append((x1, y1, x2, y2))
        
        return boxes, f"Tìm thấy {len(boxes)} annotation(s) (YOLO format)"
        
    except Exception as e:
        return None, f"Lỗi parse TXT: {str(e)}"

def visualize_annotations(image_np, boxes, classes=None):
    """Visualize bounding boxes + class_name lên ảnh"""
    img_vis = image_np.copy()
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
        label = f'Region {i+1}'
        if classes is not None and len(classes) > i:
            label += f": {classes[i]['class_name']}"
        cv2.putText(img_vis, label, (x1, max(10, y1-10)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
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
tab1, tab2 = st.tabs(["🎨 Khoanh Vùng Thủ Công", "📊 Khoanh Vùng Tự Động"])

# ========== TAB 1: MANUAL ENHANCEMENT ==========
with tab1:
    st.header("📋 Hướng dẫn sử dụng")
    with st.expander("👉 Xem hướng dẫn chi tiết", expanded=False):
        st.markdown("""
        **Bước 1:** Upload ảnh X-ray của bạn
        
        **Bước 2:** Vẽ các vùng cần tăng cường
        
        **Bước 3:** Nhấn nút "🚀 Tăng Cường" để xử lý tất cả vùng
        """)

    # ===== UPLOAD ẢNH =====
    st.markdown("---")
    st.header("📤 Upload ảnh X-ray")
    uploaded_file = st.file_uploader("Chọn ảnh X-ray", type=["png", "jpg", "jpeg", "bmp", "tiff"], key="manual_upload")

    if not uploaded_file:
        st.info("📁 Vui lòng upload ảnh X-ray để bắt đầu")
    else:
        # Load và hiển thị ảnh gốc
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        # Tính toán kích thước hiển thị giữ nguyên tỷ lệ
        orig_w, orig_h = image.size
        display_w, display_h = calculate_display_size(orig_w, orig_h)

        st.success(f"✅ Đã upload thành công! Kích thước gốc: {orig_w}x{orig_h}")
        st.info(f"📐 Kích thước hiển thị: {display_w}x{display_h}")

        # ===== VẼ VÙNG CẦN XỬ LÍ =====
        st.markdown("---")
        st.header("🖱️ Vẽ vùng cần xử lý")
        st.markdown("**Hướng dẫn:** Kéo chuột để tạo hình chữ nhật quanh các vùng bạn muốn tăng chất lượng. ")

        # Canvas để vẽ bounding boxes
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.2)",
            stroke_width=2,
            stroke_color="#FF0000",
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
                        # Xử lý từng vùng
                        for i, box in enumerate(boxes):
                            # Update progress
                            progress = (i + 1) / len(boxes)
                            progress_bar.progress(progress)
                            status_text.text(f"Đang xử lý vùng {i+1}/{len(boxes)}...")
                            
                            st.markdown(f"### 🔍 Vùng {i+1}")
                            
                            # Crop vùng 32x32 từ center của bounding box
                            region_32 = crop_center(image_np, box, crop_size=32)
                            
                            # Upscale lên 256x256 cho hiển thị
                            region_256_display = cv2.resize(region_32, (256, 256), interpolation=cv2.INTER_CUBIC)
                            
                            # Enhancement
                            enhanced = enhance_image_region(model, region_32)
                            
                            if enhanced is not None:
                                # Hiển thị kết quả so sánh trong 2 cột
                                col_orig, col_enh = st.columns(2)
                                
                                with col_orig:
                                    st.markdown("**🔸 Ảnh Gốc**")
                                    st.image(region_256_display, use_column_width=True)
                                
                                with col_enh:
                                    st.markdown("**✨ Ảnh Đã Tăng Cường**")
                                    st.image(enhanced, use_column_width=True)
                                
                                # Thêm separator giữa các vùng
                                if i < len(boxes) - 1:
                                    st.markdown("---")
                            else:
                                st.error(f"❌ Không thể xử lý vùng {i+1}")
                        
                        # Hoàn thành
                        progress_bar.progress(1.0)
                        status_text.text("✅ Hoàn thành xử lý tất cả vùng!")
                        st.success(f"🎉 Đã tăng cường thành công {len(boxes)} vùng!")
        else:
            st.info("✏️ Vẽ vùng cần tăng cường trên ảnh, sau đó nhấn nút Tăng Cường để xử lý.")

# ========== TAB 2: DATASET AUTO ENHANCEMENT ==========
with tab2:
    st.header("📊 Dataset Auto Enhancement")
    st.markdown("""
    Tính năng này cho phép tự động tăng cường các vùng đã được annotation sẵn trong dataset.
    Hỗ trợ nhiều format annotation phổ biến.
    """)
    
    # ===== BƯỚC 1: UPLOAD FILES =====
    st.markdown("---")
    st.subheader("📁 Bước 1: Upload Files")
    
    col_img, col_ann = st.columns(2)
    
    with col_img:
        st.markdown("**Upload ảnh X-ray:**")
        dataset_image = st.file_uploader("Chọn ảnh từ dataset", 
                                       type=["png", "jpg", "jpeg", "bmp", "tiff"], 
                                       key="dataset_image")
    
    with col_ann:
        st.markdown("**Upload file annotation:**")
        annotation_file = st.file_uploader("Chọn file annotation", 
                                         type=["csv", "json", "xml", "txt"], 
                                         key="annotation_file")
    
    if not dataset_image or not annotation_file:
        st.info("📁 Vui lòng upload cả ảnh và file annotation để tiếp tục")
    else:
        # Load image
        image = Image.open(dataset_image).convert("RGB")
        image_np = np.array(image)
        orig_w, orig_h = image.size
        display_w, display_h = calculate_display_size(orig_w, orig_h)
        
        st.success(f"✅ Đã upload ảnh: {dataset_image.name} ({orig_w}x{orig_h})")
        
        # ===== BƯỚC 2: PARSE ANNOTATION =====
        st.markdown("---")
        st.subheader("🔍 Bước 2: Parse Annotation")
        
        # Parse annotation based on file type
        file_ext = annotation_file.name.split('.')[-1].lower()
        boxes = None
        parse_message = ""
        
        if file_ext == 'csv':
            result, parse_message = parse_csv_annotation(annotation_file)
            if result is not None:
                boxes, classes = result
            else:
                boxes, classes = None, None
        elif file_ext == 'json':
            boxes, parse_message = parse_json_annotation(annotation_file, dataset_image.name)
        elif file_ext == 'xml':
            boxes, parse_message = parse_xml_annotation(annotation_file, dataset_image.name)
        elif file_ext == 'txt':
            boxes, parse_message = parse_txt_annotation(annotation_file, dataset_image.name, orig_w, orig_h)
        else:
            parse_message = f"Format {file_ext} chưa được hỗ trợ"
        
        if boxes is None:
            st.error(f"❌ {parse_message}")
            st.info("**Các format được hỗ trợ:**")
            st.markdown("""
            - **CSV**: filename, x, y, width, height (hoặc các biến thể)
            - **JSON**: COCO format hoặc custom format
            - **XML**: PASCAL VOC format
            - **TXT**: YOLO format (normalized coordinates)
            """)
        else:
            st.success(f"✅ {parse_message}")
            
            # Visualize annotations
            # st.markdown("**Xem trước thông tin:**")
            img_with_boxes = visualize_annotations(image_np, boxes, classes)             
            # Display với size phù hợp
            # img_display = cv2.resize(img_with_boxes, (display_w, display_h))
            # st.image(img_display, caption=f"Ảnh gốc{len(boxes)} annotation(s)", use_column_width=True) 
            # st.subheader(f"Có {len(boxes)} vùng")
            
            # Show annotation details
            with st.expander("📋 Chi tiết annotations", expanded=False):
                for i, (x1, y1, x2, y2) in enumerate(boxes):
                    if classes is not None and len(classes) > i:
                        st.write(
                            f"**Vùng {i+1}:** x1={x1}, y1={y1}, x2={x2}, y2={y2} "
                            f"(size: {x2-x1}x{y2-y1}) | class: {classes[i]['class_name']} (ID: {classes[i]['class_id']})"
                        )
                    else:
                        st.write(f"**Vùng {i+1}:** x1={x1}, y1={y1}, x2={x2}, y2={y2} (size: {x2-x1}x{y2-y1})")

            
            # ===== BƯỚC 3: TỰ ĐỘNG TĂNG CƯỜNG =====
            st.markdown("---")
            st.subheader("🚀 Bước 3: Tự động tăng cường")

            if st.button("🚀 Tăng Cường Tất Cả Các Vùng Đã Được Annotation!", 
                        type="primary", use_container_width=True, key="auto_enhance"):
                
                st.markdown("---")
                st.subheader(f"🔄 Đang xử lý {len(boxes)} vùng được annotation...")
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Container cho tất cả kết quả
                results_container = st.container()
                
                with results_container:
                    # Xử lý từng vùng
                    for i, box in enumerate(boxes):
                        # Update progress
                        progress = (i + 1) / len(boxes)
                        progress_bar.progress(progress)
                        status_text.text(f"Đang xử lý vùng annotation {i+1}/{len(boxes)}...")
                        
                        st.markdown(f"### 🔍 Annotation Vùng {i+1}")
                        
                        # Crop vùng 32x32 từ center của bounding box
                        region_32 = crop_center(image_np, box, crop_size=32)
                        
                        # Upscale lên 256x256 cho hiển thị
                        region_256_display = cv2.resize(region_32, (256, 256), interpolation=cv2.INTER_CUBIC)
                        
                        # Enhancement
                        enhanced = enhance_image_region(model, region_32)
                        
                        if enhanced is not None:
                            # Hiển thị kết quả so sánh
                            col_orig, col_enh = st.columns(2)
                            
                            with col_orig:
                                st.markdown("**🔸 Ảnh Gốc**")
                                st.image(region_256_display, use_column_width=True)
                                x1, y1, x2, y2 = box
                                st.caption(f"Vị trí: x1={x1}, y1={y1}, x2={x2}, y2={y2} (size: {x2-x1}x{y2-y1})")
                            
                            with col_enh:
                                st.markdown("**✨ Ảnh Đã Tăng Cường**")
                                st.image(enhanced, use_column_width=True)
                            
                            # Thêm separator giữa các vùng
                            if i < len(boxes) - 1:
                                st.markdown("---")
                        else:
                            st.error(f"❌ Không thể xử lý annotation vùng {i+1}")
                    
                    # Hoàn thành
                    progress_bar.progress(1.0)
                    status_text.text("✅ Đã hoàn thành tăng cường tất cả vùng annotation!")
                    st.success(f"🎉 Đã tăng cường thành công {len(boxes)} vùng annotation!")

                    # Hoàn thành
                    progress_bar.progress(1.0)
                    status_text.text("✅ Đã hoàn thành tăng cường tất cả vùng annotation!")
                    st.success(f"🎉 Đã tăng cường thành công {len(boxes)} vùng annotation!")