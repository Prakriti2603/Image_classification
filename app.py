import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import os
import gc
from functools import lru_cache

# ========== Streamlit Config ==========
st.set_page_config(page_title="Grocery Classifier", layout="centered")

# ========== Efficient Dataset Configuration ==========
@st.cache_data
def get_efficient_config():
    """Cache configuration for efficient dataset handling"""
    return {
        'img_height': 180,
        'img_width': 180,
        'batch_size': 32,
        'prefetch_size': tf.data.AUTOTUNE,
        'cache_size': 1000,
        'shuffle_buffer': 1000
    }

config = get_efficient_config()

# ========== Efficient Image Preprocessing ==========
@st.cache_data
def preprocess_image_efficient(image, target_size=(180, 180)):
    """Efficient image preprocessing with caching"""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Resize efficiently
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert to array and normalize
    img_array = np.array(image, dtype=np.float32)
    img_array = img_array / 255.0
    
    return img_array

# ========== Efficient Model Loading ==========
@st.cache_resource
def load_model_efficient():
    """Load model with memory optimization"""
    try:
        # Clear GPU memory if available
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        
        # Load model with optimization
        model = tf.keras.models.load_model(
            "Image_classify.keras",
            compile=False  # Don't compile during loading for efficiency
        )
        
        # Optimize model for inference
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# ========== Efficient Prediction Function ==========
@lru_cache(maxsize=100)
def predict_efficient(img_array):
    """Cached prediction function for efficiency"""
    try:
        # Ensure correct shape
        if len(img_array.shape) == 3:
            img_array = np.expand_dims(img_array, axis=0)
        
        # Predict with error handling
        predictions = model.predict(img_array, verbose=0)
        return predictions
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# ========== Memory Management ==========
def clear_memory():
    """Clear memory efficiently"""
    gc.collect()
    tf.keras.backend.clear_session()

# ========== Efficient Data Loading ==========
@st.cache_data
def load_class_names():
    """Cache class names for efficiency"""
    return [
        'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower',
        'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno',
        'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas',
        'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn',
        'sweetpotato', 'tomato', 'turnip', 'watermelon'
    ]

@st.cache_data
def load_nutrition_info():
    """Cache nutrition info for efficiency"""
    return {
        'apple': 'Calories: 95 | Fiber: 4g | Vitamin C: 14%',
        'banana': 'Calories: 105 | Potassium: 422mg | Vitamin B6: 20%',
        'carrot': 'Calories: 25 | Beta-carotene | Vitamin A: 334%',
        'tomato': 'Calories: 22 | Vitamin C: 40% | Potassium: 292mg',
        'orange': 'Calories: 62 | Vitamin C: 116% | Fiber: 3g',
        'potato': 'Calories: 163 | Potassium: 897mg | Vitamin C: 28%',
        'bell pepper': 'Calories: 31 | Vitamin C: 169% | Vitamin A: 12%',
        'cucumber': 'Calories: 16 | Water: 95% | Vitamin K: 16%',
        'onion': 'Calories: 40 | Vitamin C: 7% | Fiber: 1.7g',
        'garlic': 'Calories: 149 | Vitamin C: 31% | Manganese: 23%',
        'lettuce': 'Calories: 15 | Vitamin A: 166% | Vitamin K: 126%',
        'spinach': 'Calories: 23 | Iron: 2.7mg | Vitamin K: 483%',
        'watermelon': 'Calories: 30 | Water: 92% | Vitamin C: 13%',
        'grapes': 'Calories: 62 | Vitamin C: 4% | Potassium: 176mg',
        'mango': 'Calories: 60 | Vitamin C: 54% | Vitamin A: 54%',
        'pineapple': 'Calories: 50 | Vitamin C: 58% | Manganese: 44%',
        'pomegranate': 'Calories: 83 | Vitamin C: 17% | Fiber: 4g',
        'pear': 'Calories: 57 | Fiber: 3.1g | Vitamin C: 7%',
        'kiwi': 'Calories: 42 | Vitamin C: 64% | Vitamin K: 34%',
        'lemon': 'Calories: 17 | Vitamin C: 51% | Fiber: 1.6g',
        'cauliflower': 'Calories: 25 | Vitamin C: 77% | Fiber: 2.5g',
        'cabbage': 'Calories: 22 | Vitamin C: 36% | Vitamin K: 63%',
        'eggplant': 'Calories: 25 | Fiber: 3g | Potassium: 229mg',
        'ginger': 'Calories: 80 | Vitamin C: 5% | Potassium: 415mg',
        'jalepeno': 'Calories: 4 | Vitamin C: 7% | Capsaicin: Natural',
        'paprika': 'Calories: 20 | Vitamin A: 71% | Vitamin E: 29%',
        'peas': 'Calories: 84 | Protein: 5.4g | Fiber: 5.7g',
        'raddish': 'Calories: 16 | Vitamin C: 14% | Fiber: 1.6g',
        'soy beans': 'Calories: 173 | Protein: 16.6g | Fiber: 6g',
        'sweetcorn': 'Calories: 86 | Fiber: 2.7g | Vitamin C: 7%',
        'sweetpotato': 'Calories: 103 | Vitamin A: 438% | Fiber: 3.8g',
        'turnip': 'Calories: 28 | Vitamin C: 21% | Fiber: 2.3g',
        'beetroot': 'Calories: 43 | Folate: 20% | Nitrates: Natural',
        'capsicum': 'Calories: 31 | Vitamin C: 169% | Vitamin A: 12%',
        'chilli pepper': 'Calories: 40 | Vitamin C: 242% | Capsaicin: Natural',
        'corn': 'Calories: 86 | Fiber: 2.7g | Vitamin C: 7%',
    }

# ========== UI Styling ==========
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #2d5a27 0%, #4a7c59 50%, #6b8e23 100%);
        color: white;
    }
    .title {
        text-align: center;
        font-size: 48px;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 30px;
    }
    .upload-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 30px 40px;
        margin: 20px 0;
        color: #4CAF50;
        text-align: left;
        font-size: 1.5em;
        font-weight: bold;
        min-width: 500px;
        max-width: 700px;
    }
    .upload-area {
        background-color: #ffffff10;
        padding: 30px;
        border-radius: 20px;
        margin: 20px 0;
    }
    .efficiency-info {
        background: rgba(255, 255, 255, 0.1);
        padding: 10px;
        border-radius: 8px;
        margin: 10px 0;
        font-size: 0.9em;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">🍎 Grocery Image Classifier</div>', unsafe_allow_html=True)

# ========== Load Efficient Resources ==========
with st.spinner("Loading model efficiently..."):
    model = load_model_efficient()

if model is None:
    st.error("Failed to load model. Please check the model file.")
    st.stop()

class_names = load_class_names()
nutrition_info = load_nutrition_info()

# ========== Upload or Webcam ==========
st.markdown('<div class="upload-card">📤 Upload or Capture Image</div>', unsafe_allow_html=True)
st.markdown('<div class="upload-area">', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📁 Upload an image", type=["jpg", "jpeg", "png"])
use_webcam = st.checkbox("📸 Use Webcam Instead")

image = None

if use_webcam:
    class WebcamProcessor(VideoProcessorBase):
        def __init__(self):
            self.frame = None

        def recv(self, frame):
            self.frame = frame.to_ndarray(format="bgr24")
            return frame

    ctx = webrtc_streamer(
        key="webcam-capture",
        video_processor_factory=WebcamProcessor,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if ctx.video_processor:
        if st.button("📷 Capture Image"):
            frame = ctx.video_processor.frame
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
                st.image(image, caption="Captured from Webcam", use_container_width=True)
            else:
                st.warning("No frame captured. Please wait or refresh.")
else:
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ========== Efficient Prediction ==========
if image is not None:
    with st.spinner("Processing image efficiently..."):
        # Efficient preprocessing
        img_array = preprocess_image_efficient(image, (config['img_height'], config['img_width']))
        
        # Efficient prediction
        predictions = predict_efficient(img_array)
        
        if predictions is not None:
            score = tf.nn.softmax(predictions[0])
            class_index = np.argmax(score)
            label = class_names[class_index]
            confidence = round(100 * np.max(score), 2)

            st.markdown(f"### 🔍 Prediction: **{label}** ({confidence}%)")
            
            # Debug information
            st.markdown(f"**Debug Info:**")
            st.markdown(f"- Predicted class index: {class_index}")
            st.markdown(f"- Total classes in model: {len(class_names)}")
            st.markdown(f"- Top 3 predictions:")
            
            # Show top 3 predictions
            top_3_indices = np.argsort(score)[-3:][::-1]
            for i, idx in enumerate(top_3_indices):
                st.markdown(f"  {i+1}. {class_names[idx]}: {score[idx]*100:.2f}%")

            if label in nutrition_info:
                st.markdown(f"### 🥗 Nutrition Info for **{label}**")
                st.success(nutrition_info[label])
            
            # Efficiency info
            st.markdown('<div class="efficiency-info">', unsafe_allow_html=True)
            st.markdown("**Efficiency Features:**")
            st.markdown("- Cached model loading")
            st.markdown("- Optimized image preprocessing")
            st.markdown("- Memory-efficient prediction")
            st.markdown("- LRU cache for repeated predictions")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error("Prediction failed. Please try again.")

# Clear memory periodically
if st.button("Clear Memory"):
    clear_memory()
    st.success("Memory cleared successfully!")