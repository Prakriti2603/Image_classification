import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import sys

def load_model(model_path):
    """Load the trained model"""
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def load_class_indices(json_path):
    """Load class indices from JSON file"""
    try:
        with open(json_path, 'r') as f:
            class_indices = json.load(f)
        print(f"✅ Class indices loaded successfully from {json_path}")
        return class_indices
    except Exception as e:
        print(f"❌ Error loading class indices: {e}")
        return None

def preprocess_image(image_path, target_size=(180, 180)):
    """Preprocess image for prediction"""
    try:
        image = Image.open(image_path).convert('RGB')
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        img_array = np.array(image, dtype=np.float32)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        print(f"✅ Image preprocessed successfully: {image_path}")
        return img_array
    except Exception as e:
        print(f"❌ Error preprocessing image: {e}")
        return None

def predict_image(model, img_array, class_indices):
    """Predict class for the given image"""
    try:
        predictions = model.predict(img_array, verbose=0)
        score = tf.nn.softmax(predictions[0])
        
        # Get class names (invert the dictionary)
        class_names = {v: k for k, v in class_indices.items()}
        
        # Get top 5 predictions
        top_5_indices = np.argsort(score)[-5:][::-1]
        
        print("\n🔍 Top 5 Predictions:")
        for i, idx in enumerate(top_5_indices):
            confidence = 100 * score[idx]
            print(f"{i+1}. {class_names[idx]}: {confidence:.2f}%")
            
        return class_names[top_5_indices[0]], score[top_5_indices[0]]
    except Exception as e:
        print(f"❌ Error making prediction: {e}")
        return None, None

def main():
    """Main function to test the model"""
    print("🍎 Model Testing Tool")
    print("="*50)
    
    # Check if image path is provided
    if len(sys.argv) < 2:
        print("❌ Please provide an image path as argument")
        print("Example: python test_model.py path/to/image.jpg")
        return
    
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    # Load model and class indices
    model_path = "Image_classify.keras"  # Default model
    if len(sys.argv) >= 3:
        model_path = sys.argv[2]
    
    model = load_model(model_path)
    if model is None:
        return
    
    class_indices = load_class_indices("class_indices.json")
    if class_indices is None:
        return
    
    # Preprocess image
    img_array = preprocess_image(image_path)
    if img_array is None:
        return
    
    # Make prediction
    print(f"\n🔍 Analyzing image: {image_path}")
    label, confidence = predict_image(model, img_array, class_indices)
    
    if label is not None:
        print(f"\n✅ Final Prediction: {label} ({confidence*100:.2f}%)")
    
    print("\n💡 Debug Information:")
    print(f"- Model: {model_path}")
    print(f"- Image shape: {img_array.shape}")
    print(f"- Number of classes: {len(class_indices)}")

if __name__ == "__main__":
    main()