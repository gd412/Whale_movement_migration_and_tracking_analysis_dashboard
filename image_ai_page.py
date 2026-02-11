import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
import plotly.express as px
import cv2

MODEL_PATH = "whale_model.pth"
CSV_PATH = "whales_only.csv"
TOP_K = 3
CONF_THRESHOLD = 0.50

# ================================================================
# LOAD LABELS
# ================================================================
@st.cache_data
def load_labels():
    df = pd.read_csv(CSV_PATH)
    col = [c for c in df.columns if ("species" in c.lower() or "id" in c.lower())][0]
    labels = df[col].unique()
    label_map = {l: i for i, l in enumerate(labels)}
    inv_map = {v: k for k, v in label_map.items()}
    return labels, inv_map

labels, inv_label_map = load_labels()
num_classes = len(labels)

# ================================================================
# LOAD MODEL
# ================================================================
@st.cache_resource
def load_model():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ================================================================
# TRANSFORM
# ================================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ================================================================
# GRADCAM
# ================================================================
def generate_gradcam(model, img_tensor, original_img):

    gradients = []
    activations = []

    def f_hook(m, i, o):
        activations.append(o)

    def b_hook(m, gi, go):
        gradients.append(go[0])

    layer = model.layer4[-1]
    fh = layer.register_forward_hook(f_hook)
    bh = layer.register_backward_hook(b_hook)

    out = model(img_tensor)
    pred = out.argmax(dim=1)

    model.zero_grad()
    out[0, pred].backward()

    grads = gradients[0][0].cpu().data.numpy()
    acts = activations[0][0].cpu().data.numpy()

    weights = grads.mean(axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam - cam.min()
    cam = cam / cam.max()

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    base = np.array(original_img.resize((224, 224)))
    overlay = cv2.addWeighted(base, 0.6, heatmap, 0.4, 0)

    fh.remove()
    bh.remove()

    return overlay

# ================================================================
# MAIN UI
# ================================================================
def run_image_ai():

    st.title("🧠 Whale Image AI Classifier")

    st.sidebar.subheader("⚙ Model Info")
    st.sidebar.write("Architecture: ResNet18")
    st.sidebar.write("Classes:", num_classes)
    st.sidebar.write("Top-K:", TOP_K)
    st.sidebar.write("Threshold:", CONF_THRESHOLD)

    mode = st.radio("Mode", ["Single Image", "Batch Images"])

    results_log = []

    # ---------------- SINGLE ----------------
    if mode == "Single Image":

        file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

        if file:
            image = Image.open(file).convert("RGB")

            col1, col2 = st.columns(2)

            with col1:
                st.image(image, caption="Original", use_container_width=True)

            img_tensor = transform(image)

            with col2:
                st.image(
                    np.transpose(img_tensor.numpy(), (1, 2, 0)),
                    caption="Preprocessed 224x224",
                    use_container_width=True
                )

            input_tensor = img_tensor.unsqueeze(0)

            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.softmax(outputs, dim=1)[0]

            top_probs, top_idxs = torch.topk(probs, TOP_K)

            st.subheader("🔮 Top Predictions")

            rows = []

            for p, idx in zip(top_probs, top_idxs):
                label = inv_label_map[idx.item()]
                conf = float(p)

                rows.append((label, conf))

                if conf < CONF_THRESHOLD:
                    st.warning(f"{label}: {conf:.2%}")
                else:
                    st.success(f"{label}: {conf:.2%}")

            df_chart = pd.DataFrame(rows, columns=["Species", "Confidence"])
            st.plotly_chart(px.bar(df_chart, x="Species", y="Confidence"),
                            use_container_width=True)

            st.subheader("🔥 Activation Map")
            overlay = generate_gradcam(model, input_tensor, image)
            st.image(overlay, use_container_width=True)

            results_log.append({
                "file": file.name,
                "prediction": rows[0][0],
                "confidence": rows[0][1]
            })

    # ---------------- BATCH ----------------
    else:

        files = st.file_uploader(
            "Upload Multiple",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )

        if files:
            for file in files:
                image = Image.open(file).convert("RGB")
                tensor = transform(image).unsqueeze(0)

                with torch.no_grad():
                    out = model(tensor)
                    prob = torch.softmax(out, dim=1)[0]
                    pred = prob.argmax().item()

                results_log.append({
                    "file": file.name,
                    "prediction": inv_label_map[pred],
                    "confidence": float(prob[pred])
                })

            df = pd.DataFrame(results_log)
            st.dataframe(df)

            st.download_button(
                "📥 Download CSV",
                df.to_csv(index=False),
                "predictions.csv"
            )

    # ---------------- SESSION ANALYTICS ----------------
    if results_log:
        st.subheader("📊 Session Analytics")
        df = pd.DataFrame(results_log)
        st.plotly_chart(px.histogram(df, x="prediction"),
                        use_container_width=True)
