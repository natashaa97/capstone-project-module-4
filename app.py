import numpy as np
import streamlit as st
import supervision as sv
from ultralytics import YOLO
from PIL import Image
from io import BytesIO 
import plotly.graph_objects as go
import plotly.express as px

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="🚗 Vehicle Detection System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS — Clean & Professional UI
# ============================================================
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }

    /* Header styling */
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        margin-bottom: 1rem;
    }
    .main-header h1 {
        font-size: 2.5rem;
        background: linear-gradient(90deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: #8892b0;
        font-size: 1.05rem;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e293b, #334155);
        border: 1px solid #475569;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0.3rem 0;
    }
    .metric-label {
        color: #94a3b8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .car-color { color: #3b82f6; }
    .bus-color { color: #ef4444; }
    .van-color { color: #22c55e; }
    .total-color {
        background: linear-gradient(90deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Result table */
    .result-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
    }
    .result-table th {
        background: #1e293b;
        color: #94a3b8;
        padding: 0.7rem 1rem;
        text-align: left;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        border-bottom: 2px solid #334155;
    }
    .result-table td {
        padding: 0.7rem 1rem;
        border-bottom: 1px solid #1e293b;
        color: #e2e8f0;
        font-size: 0.95rem;
    }

    /* Info box */
    .info-box {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border-left: 4px solid #3b82f6;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.2rem;
        margin: 0.8rem 0;
        color: #cbd5e1;
        font-size: 0.9rem;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a, #1e293b) !important;
    }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Upload area */
    [data-testid="stFileUploader"] {
        border: 2px dashed #334155;
        border-radius: 12px;
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL (cached so it only loads once)
# ============================================================
@st.cache_resource
def load_model():
    """Load the trained YOLOv12 model."""
    model = YOLO('best_vehicle.pt')
    return model


# ============================================================
# DETECTION FUNCTION
# ============================================================
def detect_vehicles(image_bytes, model, confidence):
    """
    Run vehicle detection on an uploaded image.

    Args:
        image_bytes: Raw bytes from uploaded file
        model: YOLO model
        confidence: Confidence threshold (0.0 - 1.0)

    Returns:
        annotated_image: Image with bounding boxes (numpy array, RGB)
        vehicle_counts: Dict with count per vehicle class
        total: Total vehicles detected
        detections: Raw supervision Detections object
    """
    # Convert bytes to PIL Image → numpy array
    pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(pil_image)

    # Run inference
    results = model(image_np, conf=confidence, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results).with_nms()

    # Annotate image
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.6, text_thickness=1)

    annotated = image_np.copy()
    annotated = box_annotator.annotate(scene=annotated, detections=detections)
    annotated = label_annotator.annotate(scene=annotated, detections=detections)

    # Count per class
    vehicle_counts = {}
    for class_name in detections.data.get("class_name", []):
        vehicle_counts[class_name] = vehicle_counts.get(class_name, 0) + 1

    total = sum(vehicle_counts.values())

    return annotated, vehicle_counts, total, detections


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    # Confidence slider
    confidence = st.slider(
        "🎯 Confidence Threshold",
        min_value=0.10,
        max_value=0.95,
        value=0.25,
        step=0.05,
        help="Seberapa yakin model harus sebelum mendeteksi objek. "
             "Makin tinggi = makin selektif, makin rendah = makin banyak deteksi."
    )

    st.markdown("---")
    st.markdown("## 📖 How to Use")
    st.markdown("""
    1. Upload gambar kendaraan
    2. Atur confidence threshold
    3. Klik **Detect Vehicles**
    4. Lihat hasil deteksi & analisis
    """)

    st.markdown("---")
    st.markdown("## 🏷️ Supported Classes")
    st.markdown("""
    - 🚗 **Car**
    - 🚌 **Bus**
    - 🚐 **Van**
    """)

    st.markdown("---")
    st.markdown(
        "<p style='color: #64748b; font-size: 0.75rem; text-align: center;'>"
        "Capstone Project Module 4<br>"
        "Vehicle Detection with YOLOv12"
        "</p>",
        unsafe_allow_html=True
    )


# ============================================================
# MAIN CONTENT
# ============================================================

# Header
st.markdown("""
<div class="main-header">
    <h1>🚗 Vehicle Detection System</h1>
    <p>Detect and count vehicles (car, bus, van) from images using YOLOv12</p>
</div>
""", unsafe_allow_html=True)

# Load model
model = load_model()

# File uploader
uploaded_file = st.file_uploader(
    "📸 Upload an image",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=False,
    help="Upload a photo containing vehicles to detect"
)

if uploaded_file is not None:
    # Show original image preview
    bytes_data = uploaded_file.getvalue()

    # Detect button
    detect_btn = st.button("🔍 Detect Vehicles", type="primary", use_container_width=True)

    if detect_btn:
        # Run detection with spinner
        with st.spinner("🔍 Detecting vehicles..."):
            annotated_image, vehicle_counts, total, detections = detect_vehicles(
                bytes_data, model, confidence
            )

        # ---- RESULTS SECTION ----
        st.markdown("---")

        # Metric cards
        col1, col2, col3, col4 = st.columns(4)

        car_count = vehicle_counts.get("car", 0)
        bus_count = vehicle_counts.get("bus", 0)
        van_count = vehicle_counts.get("van", 0)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Vehicles</div>
                <div class="metric-value total-color">{total}</div>
                <div class="metric-label">detected</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">🚗 Car</div>
                <div class="metric-value car-color">{car_count}</div>
                <div class="metric-label">detected</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">🚌 Bus</div>
                <div class="metric-value bus-color">{bus_count}</div>
                <div class="metric-label">detected</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">🚐 Van</div>
                <div class="metric-value van-color">{van_count}</div>
                <div class="metric-label">detected</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Image comparison: Original vs Detected
        col_img1, col_img2 = st.columns(2)

        with col_img1:
            st.markdown("#### 📷 Original Image")
            original_image = Image.open(BytesIO(bytes_data)).convert("RGB")
            st.image(original_image, use_container_width=True)

        with col_img2:
            st.markdown("#### 🔍 Detection Result")
            st.image(annotated_image, use_container_width=True)

        # ---- ANALYSIS SECTION ----
        st.markdown("---")
        st.markdown("### 📊 Detection Analysis")

        if total > 0:
            col_chart1, col_chart2 = st.columns(2)

            # Color mapping
            color_map = {
                "car": "#3b82f6",
                "bus": "#ef4444",
                "van": "#22c55e"
            }

            with col_chart1:
                # Bar chart
                fig_bar = go.Figure()
                for vehicle_type, count in vehicle_counts.items():
                    fig_bar.add_trace(go.Bar(
                        x=[vehicle_type],
                        y=[count],
                        name=vehicle_type,
                        marker_color=color_map.get(vehicle_type, "#8b5cf6"),
                        text=[count],
                        textposition='outside',
                        textfont=dict(size=16, color='white')
                    ))

                fig_bar.update_layout(
                    title=dict(text="Vehicle Count by Type", font=dict(size=16, color='white')),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    showlegend=False,
                    yaxis=dict(
                        gridcolor='rgba(255,255,255,0.1)',
                        title='Count'
                    ),
                    xaxis=dict(title='Vehicle Type'),
                    height=350
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            with col_chart2:
                # Pie chart
                fig_pie = go.Figure(data=[go.Pie(
                    labels=list(vehicle_counts.keys()),
                    values=list(vehicle_counts.values()),
                    marker=dict(
                        colors=[color_map.get(v, "#8b5cf6") for v in vehicle_counts.keys()]
                    ),
                    textinfo='label+percent+value',
                    textfont=dict(size=13),
                    hole=0.4
                )])

                fig_pie.update_layout(
                    title=dict(text="Vehicle Distribution", font=dict(size=16, color='white')),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    showlegend=True,
                    legend=dict(font=dict(color='white')),
                    height=350
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            # Summary table
            st.markdown("#### 📋 Detection Summary")

            table_html = """
            <table class="result-table">
                <thead>
                    <tr>
                        <th>Vehicle Type</th>
                        <th>Count</th>
                        <th>Percentage</th>
                    </tr>
                </thead>
                <tbody>
            """
            for vehicle_type, count in sorted(vehicle_counts.items(), key=lambda x: x[1], reverse=True):
                pct = (count / total * 100) if total > 0 else 0
                emoji = {"car": "🚗", "bus": "🚌", "van": "🚐"}.get(vehicle_type, "🚙")
                table_html += f"""
                    <tr>
                        <td>{emoji} {vehicle_type.capitalize()}</td>
                        <td><strong>{count}</strong></td>
                        <td>{pct:.1f}%</td>
                    </tr>
                """

            table_html += f"""
                    <tr style="border-top: 2px solid #334155; font-weight: bold;">
                        <td>🚦 Total</td>
                        <td><strong>{total}</strong></td>
                        <td>100%</td>
                    </tr>
                </tbody>
            </table>
            """
            st.markdown(table_html, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="info-box">
                ℹ️ <strong>No vehicles detected.</strong><br>
                Coba turunkan confidence threshold di sidebar, atau upload gambar lain yang mengandung kendaraan.
            </div>
            """, unsafe_allow_html=True)

        # Detection details (expandable)
        with st.expander("🔎 Detection Details (Advanced)"):
            st.markdown(f"**Confidence Threshold:** {confidence}")
            st.markdown(f"**Total Detections:** {len(detections)}")
            st.markdown(f"**Image Size:** {original_image.size[0]} x {original_image.size[1]} px")

            if len(detections) > 0:
                st.markdown("**Individual Detections:**")
                for i in range(len(detections)):
                    class_name = detections.data.get('class_name', [])[i]
                    conf = detections.confidence[i]
                    bbox = detections.xyxy[i]
                    st.markdown(
                        f"- **{class_name}** — confidence: `{conf:.2%}` — "
                        f"bbox: `[{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]`"
                    )

else:
    # Placeholder when no image uploaded
    st.markdown("""
    <div class="info-box">
        👆 <strong>Upload an image to get started!</strong><br>
        Upload a photo containing vehicles (car, bus, van) and the system will detect and count them automatically.
    </div>
    """, unsafe_allow_html=True)

    # Example use case
    st.markdown("---")
    st.markdown("### 💡 What This App Does")

    col_f1, col_f2, col_f3 = st.columns(3)

    with col_f1:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">🔍</div>
            <div style="color: #e2e8f0; font-weight: 600; margin-bottom: 0.3rem;">Detect</div>
            <div style="color: #94a3b8; font-size: 0.85rem;">
                Automatically detect vehicles in uploaded images using YOLOv12
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_f2:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">🔢</div>
            <div style="color: #e2e8f0; font-weight: 600; margin-bottom: 0.3rem;">Count</div>
            <div style="color: #94a3b8; font-size: 0.85rem;">
                Count vehicles by type: car, bus, and van with detailed breakdown
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_f3:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">📊</div>
            <div style="color: #e2e8f0; font-weight: 600; margin-bottom: 0.3rem;">Analyze</div>
            <div style="color: #94a3b8; font-size: 0.85rem;">
                Visualize detection results with charts and detailed statistics
            </div>
        </div>
        """, unsafe_allow_html=True)
