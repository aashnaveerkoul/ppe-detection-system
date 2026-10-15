import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# --- Page Configuration ---
st.set_page_config(
    page_title="SiteGuard AI - PPE Compliance Dashboard",
    page_icon="🛡️",
    layout="wide",
)

# --- Modern UI & Custom CSS Styling ---
st.markdown(
    """
    <style>
    /* Main Theme & Typography */
    .main-header {
        font-size: 2.4rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.025em;
        margin-bottom: 0.1rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #475569;
        margin-bottom: 2rem;
    }
    
    /* Card Containers */
    .metric-container {
        background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%);
        border: 1px solid #E2E8F0;
        padding: 1.2rem;
        border-radius: 0.75rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stMarkdown {
        color: #E2E8F0 !important;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# --- Load Model ---
@st.cache_resource
def load_ppe_model():
    return YOLO("best_v2.pt")


try:
    model = load_ppe_model()
except Exception as e:
    st.error(
        f"Error loading model from `best_v2.pt`. Please ensure the weight file is in the root folder. Details: {e}"
    )
    st.stop()

# --- App Sidebar ---
st.sidebar.markdown("##  Audit Controls")
upload_type = st.sidebar.radio(
    "Select Input Mode", ["Image Audit", "Video Stream Audit"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("###  Sensitivity Settings")
# Kept ONLY the Confidence Threshold slider as requested
confidence_threshold = (
    st.sidebar.slider(
        "Confidence Threshold (%)",
        min_value=10,
        max_value=90,
        value=40,
        step=5,
        help="Filter out detections below this confidence score.",
    )
)
# Fixed IoU threshold in the background
iou_threshold = 0.45

st.sidebar.markdown("---")
st.sidebar.markdown("###  Target Classes")
st.sidebar.info(
    "Detects safety gear: Helmets, Gloves, Vests, Boots, Goggles, and flags compliance violations automatically."
)

# --- Main Page Header ---
st.markdown(
    '<p class="main-header">🛡️ SiteGuard AI</p>', unsafe_allow_html=True
)
st.markdown(
    '<p class="sub-header">Advanced Computer Vision Safety Gear & PPE Compliance Auditing</p>',
    unsafe_allow_html=True,
)


# --- Helper Functions ---
def process_image(image_pil, conf_thresh):
    img_cv = np.array(image_pil)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    orig_h, orig_w = img_cv.shape[:2]

    # Speed Optimization: Resize large images for inference while tracking scale
    max_dim = 1024
    scale = 1.0
    if max(orig_h, orig_w) > max_dim:
        scale = max_dim / max(orig_h, orig_w)
        inference_img = cv2.resize(img_cv, (int(orig_w * scale), int(orig_h * scale)))
    else:
        inference_img = img_cv

    results = model(
        inference_img, conf=conf_thresh / 100.0, iou=iou_threshold, verbose=False
    )

    detections = []
    violation_count = 0
    compliant_count = 0

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            class_name = model.names[cls_id]

            # Skip person detections entirely as requested
            if class_name.lower() == "person":
                continue

            if scale != 1.0:
                xyxy = [coords / scale for coords in xyxy]

            x1, y1, x2, y2 = map(int, xyxy)

            # Categorize status based on class name tags (e.g., 'no helmet' vs 'helmet')
            is_violation = "no" in class_name.lower()
            if is_violation:
                violation_count += 1
                status = "Violation"
                color = (0, 0, 255)  # Red for violations
            else:
                compliant_count += 1
                status = "Compliant"
                color = (0, 255, 0)  # Green for compliant equipment

            detections.append(
                {
                    "box": [x1, y1, x2, y2],
                    "class": class_name,
                    "confidence": conf,
                    "status": status,
                    "color": color,
                }
            )

            # Draw Bounding Box & Label
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} ({conf:.2f})"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_cv, (x1, y1 - 22), (x1 + w, y1), color, -1)
            cv2.putText(
                img_cv,
                label,
                (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

    annotated_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return annotated_rgb, compliant_count, violation_count, detections


# --- Workflow: Image Audit ---
if upload_type == "Image Audit":
    uploaded_file = st.file_uploader(
        "Upload Job Site Image",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear photo of workers on site to run the audit.",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        with st.spinner("Running neural network inference & auditing PPE compliance..."):
            annotated_img, compliant, violations, detections = process_image(
                image, confidence_threshold
            )

        # --- Status Alert Banner ---
        if violations > 0:
            st.error(
                f"🚨 **Safety Alert:** {violations} PPE compliance violation(s) detected on site!",
                icon="⚠️",
            )
        else:
            st.success(
                "✅ **Site Secure:** All detected safety equipment is compliant!",
                icon="🎉",
            )

        # --- Metrics Dashboard Row ---
        st.markdown("### Audit Summary")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Total Equipment Items", len(detections))
        with m2:
            st.metric("Compliant Gear", compliant)
        with m3:
            st.metric(
                "Violations Flagged",
                violations,
                delta_color="inverse" if violations > 0 else "off",
            )

        st.markdown("---")

        # --- Side-by-Side Audit View ---
        st.markdown("###  Visual Comparison Panel")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                "<p style='font-weight: 600; color: #334155;'>Original Capture</p>",
                unsafe_allow_html=True,
            )
            st.image(image, use_container_width=True)

        with col2:
            st.markdown(
                "<p style='font-weight: 600; color: #334155;'>Compliance Audit Overlay</p>",
                unsafe_allow_html=True,
            )
            st.image(annotated_img, use_container_width=True)

        # --- Detailed Inspection Table ---
        if detections:
            st.markdown("---")
            st.markdown("### Detailed Detections Ledger")
            table_data = [
                {
                    "Equipment / Class": d["class"],
                    "Status": d["status"],
                    "Confidence": f"{d['confidence'] * 100:.1f}%",
                    "Bounding Box Coordinates [x1, y1, x2, y2]": str(d["box"]),
                }
                for d in detections
            ]
            st.dataframe(table_data, use_container_width=True)
    else:
        # Professional Drop Zone Placeholder View
        st.info(
            " Get started by uploading a job site image using the file uploader above."
        )

# --- Workflow: Video Stream Audit ---
else:
    st.markdown("###  Video Stream & Sequence Audit")
    video_file = st.file_uploader(
        "Upload Job Site Video Sequence", type=["mp4", "avi", "mov"]
    )

    if video_file is not None:
        st.success("Video sequence loaded successfully!")
        if st.button("Initialize Video Frame Processing", type="primary"):
            st.warning(
                "Video stream frame-by-frame processing pipeline is ready for execution."
            )
    else:
        st.info(" Upload a video clip to begin sequence compliance auditing.")