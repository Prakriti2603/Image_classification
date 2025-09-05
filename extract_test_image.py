import os
import shutil
import random

def extract_test_images():
    """Extract test images from the training dataset"""
    print("🍎 Extracting Test Images")
    print("="*50)
    
    # Source directories
    tomato_dir = os.path.join("Fruits_Vegetables", "Fruits_Vegetables", "train", "tomato")
    eggplant_dir = os.path.join("Fruits_Vegetables", "Fruits_Vegetables", "train", "eggplant")
    
    # Check if directories exist
    if not os.path.exists(tomato_dir):
        print(f"❌ Tomato directory not found: {tomato_dir}")
        return
    
    if not os.path.exists(eggplant_dir):
        print(f"❌ Eggplant directory not found: {eggplant_dir}")
        return
    
    # Get list of images
    tomato_images = [f for f in os.listdir(tomato_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    eggplant_images = [f for f in os.listdir(eggplant_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not tomato_images:
        print("❌ No tomato images found")
        return
    
    if not eggplant_images:
        print("❌ No eggplant images found")
        return
    
    # Select random images
    tomato_sample = random.sample(tomato_images, min(5, len(tomato_images)))
    eggplant_sample = random.sample(eggplant_images, min(5, len(eggplant_images)))
    
    # Copy images to root directory
    for i, img in enumerate(tomato_sample):
        src = os.path.join(tomato_dir, img)
        dst = f"tomato_test_{i+1}.jpg"
        shutil.copy2(src, dst)
        print(f"✅ Copied tomato image: {dst}")
    
    for i, img in enumerate(eggplant_sample):
        src = os.path.join(eggplant_dir, img)
        dst = f"eggplant_test_{i+1}.jpg"
        shutil.copy2(src, dst)
        print(f"✅ Copied eggplant image: {dst}")
    
    print("\n🎯 Next steps:")
    print("1. Run: python test_model.py tomato_test_1.jpg")
    print("2. Run: python test_model.py eggplant_test_1.jpg")
    print("3. Compare the results to identify potential issues")

if __name__ == "__main__":
    extract_test_images()