import os
import requests
import urllib.request
from bs4 import BeautifulSoup
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from datetime import datetime
import json
import hashlib
from tqdm import tqdm
import zipfile
import shutil
from pathlib import Path

class DatasetExpansion:
    def __init__(self, base_dir="expanded_dataset"):
        self.base_dir = base_dir
        self.classes = [
            'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower',
            'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno',
            'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas',
            'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn',
            'sweetpotato', 'tomato', 'turnip', 'watermelon'
        ]
        self.setup_directories()
        
    def setup_directories(self):
        """Create directory structure for expanded dataset"""
        for class_name in self.classes:
            class_dir = os.path.join(self.base_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            os.makedirs(os.path.join(class_dir, 'original'), exist_ok=True)
            os.makedirs(os.path.join(class_dir, 'augmented'), exist_ok=True)
            os.makedirs(os.path.join(class_dir, 'scraped'), exist_ok=True)
    
    def web_scraping(self, class_name, num_images=50):
        """Scrape images from the web for a specific class"""
        print(f"Scraping {num_images} images for {class_name}...")
        
        # Search terms for better results
        search_terms = [
            f"{class_name} fresh",
            f"{class_name} organic",
            f"{class_name} close up",
            f"{class_name} on white background"
        ]
        
        scraped_count = 0
        for search_term in search_terms:
            if scraped_count >= num_images:
                break
                
            try:
                # Use Unsplash API for high-quality images
                url = f"https://unsplash.com/s/photos/{search_term.replace(' ', '-')}"
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                
                response = requests.get(url, headers=headers)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find image links (this is a simplified version)
                img_tags = soup.find_all('img')
                
                for img in img_tags[:10]:  # Limit to 10 images per search term
                    if scraped_count >= num_images:
                        break
                        
                    img_url = img.get('src')
                    if img_url and img_url.startswith('http'):
                        try:
                            # Download image
                            img_path = os.path.join(
                                self.base_dir, class_name, 'scraped',
                                f"{class_name}_{scraped_count}.jpg"
                            )
                            
                            urllib.request.urlretrieve(img_url, img_path)
                            
                            # Verify image is valid
                            if self.validate_image(img_path):
                                scraped_count += 1
                                print(f"Downloaded: {img_path}")
                            else:
                                os.remove(img_path)
                                
                        except Exception as e:
                            print(f"Error downloading {img_url}: {e}")
                            
            except Exception as e:
                print(f"Error scraping {search_term}: {e}")
        
        print(f"Successfully scraped {scraped_count} images for {class_name}")
        return scraped_count
    
    def validate_image(self, image_path):
        """Validate if image is usable"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return False
            
            # Check minimum size
            if img.shape[0] < 100 or img.shape[1] < 100:
                return False
            
            # Check if image is not corrupted
            if img.size == 0:
                return False
                
            return True
        except:
            return False
    
    def augment_dataset(self, class_name, augmentations_per_image=5):
        """Create augmented versions of existing images"""
        print(f"Augmenting images for {class_name}...")
        
        class_dir = os.path.join(self.base_dir, class_name)
        original_dir = os.path.join(class_dir, 'original')
        augmented_dir = os.path.join(class_dir, 'augmented')
        
        # Data augmentation configuration
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=False,
            zoom_range=0.2,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        
        # Get all original images
        image_files = [f for f in os.listdir(original_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        augmented_count = 0
        for img_file in tqdm(image_files, desc=f"Augmenting {class_name}"):
            img_path = os.path.join(original_dir, img_file)
            
            try:
                # Load and preprocess image
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (180, 180))
                img = img.reshape((1,) + img.shape)
                
                # Generate augmented images
                for i, batch in enumerate(datagen.flow(img, batch_size=1)):
                    if i >= augmentations_per_image:
                        break
                    
                    augmented_img = batch[0]
                    augmented_img = (augmented_img * 255).astype(np.uint8)
                    
                    # Save augmented image
                    aug_filename = f"aug_{img_file.split('.')[0]}_{i}.jpg"
                    aug_path = os.path.join(augmented_dir, aug_filename)
                    
                    cv2.imwrite(aug_path, cv2.cvtColor(augmented_img, cv2.COLOR_RGB2BGR))
                    augmented_count += 1
                    
            except Exception as e:
                print(f"Error augmenting {img_path}: {e}")
        
        print(f"Generated {augmented_count} augmented images for {class_name}")
        return augmented_count
    
    def download_public_datasets(self):
        """Download and integrate public fruit/vegetable datasets"""
        print("Downloading public datasets...")
        
        # Dataset URLs (you'll need to replace with actual URLs)
        datasets = {
            'fruits-360': 'https://www.kaggle.com/datasets/moltean/fruits/download',
            'vegetable-image-dataset': 'https://www.kaggle.com/datasets/...',
            'food-101': 'https://www.kaggle.com/datasets/...'
        }
        
        for dataset_name, url in datasets.items():
            try:
                print(f"Downloading {dataset_name}...")
                # Implementation for downloading datasets
                # This would require proper API keys and authentication
                pass
            except Exception as e:
                print(f"Error downloading {dataset_name}: {e}")
    
    def generate_synthetic_images(self, class_name, num_synthetic=20):
        """Generate synthetic images using various techniques"""
        print(f"Generating {num_synthetic} synthetic images for {class_name}...")
        
        # This is a placeholder for synthetic image generation
        # You could use GANs, StyleGAN, or other generative models
        
        synthetic_dir = os.path.join(self.base_dir, class_name, 'synthetic')
        os.makedirs(synthetic_dir, exist_ok=True)
        
        # For now, we'll create simple synthetic images using basic techniques
        for i in range(num_synthetic):
            # Create a simple synthetic image (placeholder)
            synthetic_img = np.random.randint(0, 255, (180, 180, 3), dtype=np.uint8)
            
            # Apply some basic transformations to make it more realistic
            synthetic_img = cv2.GaussianBlur(synthetic_img, (5, 5), 0)
            
            # Save synthetic image
            synth_path = os.path.join(synthetic_dir, f"synthetic_{class_name}_{i}.jpg")
            cv2.imwrite(synth_path, synthetic_img)
        
        print(f"Generated {num_synthetic} synthetic images for {class_name}")
        return num_synthetic
    
    def quality_control(self, class_name):
        """Perform quality control on dataset"""
        print(f"Performing quality control for {class_name}...")
        
        class_dir = os.path.join(self.base_dir, class_name)
        all_images = []
        
        # Collect all images
        for subdir in ['original', 'augmented', 'scraped', 'synthetic']:
            subdir_path = os.path.join(class_dir, subdir)
            if os.path.exists(subdir_path):
                images = [os.path.join(subdir_path, f) for f in os.listdir(subdir_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                all_images.extend(images)
        
        # Quality control checks
        valid_images = []
        for img_path in tqdm(all_images, desc=f"Quality control for {class_name}"):
            try:
                img = cv2.imread(img_path)
                
                # Check image quality
                if img is not None and img.size > 0:
                    # Check minimum size
                    if img.shape[0] >= 100 and img.shape[1] >= 100:
                        # Check for reasonable aspect ratio
                        aspect_ratio = img.shape[1] / img.shape[0]
                        if 0.5 <= aspect_ratio <= 2.0:
                            # Check for reasonable brightness
                            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            mean_brightness = np.mean(gray)
                            if 20 <= mean_brightness <= 235:
                                valid_images.append(img_path)
                            else:
                                os.remove(img_path)
                        else:
                            os.remove(img_path)
                    else:
                        os.remove(img_path)
                else:
                    os.remove(img_path)
                    
            except Exception as e:
                print(f"Error in quality control for {img_path}: {e}")
                if os.path.exists(img_path):
                    os.remove(img_path)
        
        print(f"Quality control complete for {class_name}: {len(valid_images)} valid images")
        return len(valid_images)
    
    def create_dataset_summary(self):
        """Create a summary of the expanded dataset"""
        print("Creating dataset summary...")
        
        summary = {
            'total_classes': len(self.classes),
            'classes': {},
            'total_images': 0,
            'creation_date': str(datetime.now())
        }
        
        for class_name in self.classes:
            class_dir = os.path.join(self.base_dir, class_name)
            class_summary = {}
            
            for subdir in ['original', 'augmented', 'scraped', 'synthetic']:
                subdir_path = os.path.join(class_dir, subdir)
                if os.path.exists(subdir_path):
                    count = len([f for f in os.listdir(subdir_path) 
                               if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                    class_summary[subdir] = count
                else:
                    class_summary[subdir] = 0
            
            class_summary['total'] = sum(class_summary.values())
            summary['classes'][class_name] = class_summary
            summary['total_images'] += class_summary['total']
        
        # Save summary
        summary_path = os.path.join(self.base_dir, 'dataset_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Dataset summary saved to {summary_path}")
        print(f"Total images: {summary['total_images']}")
        
        return summary
    
    def expand_dataset(self, target_images_per_class=200):
        """Main function to expand the dataset"""
        print("Starting dataset expansion...")
        
        for class_name in self.classes:
            print(f"\n{'='*50}")
            print(f"Processing class: {class_name}")
            print(f"{'='*50}")
            
            # 1. Web scraping
            scraped_count = self.web_scraping(class_name, num_images=50)
            
            # 2. Data augmentation
            augmented_count = self.augment_dataset(class_name, augmentations_per_image=5)
            
            # 3. Synthetic image generation
            synthetic_count = self.generate_synthetic_images(class_name, num_synthetic=20)
            
            # 4. Quality control
            valid_count = self.quality_control(class_name)
            
            print(f"Class {class_name} summary:")
            print(f"  - Scraped: {scraped_count}")
            print(f"  - Augmented: {augmented_count}")
            print(f"  - Synthetic: {synthetic_count}")
            print(f"  - Valid after QC: {valid_count}")
        
        # Create final summary
        summary = self.create_dataset_summary()
        
        print(f"\n{'='*50}")
        print("Dataset expansion complete!")
        print(f"Total images: {summary['total_images']}")
        print(f"Dataset location: {os.path.abspath(self.base_dir)}")
        print(f"{'='*50}")

def main():
    """Main function to run dataset expansion"""
    print("🍎 Fruit/Vegetable Dataset Expansion Tool")
    print("="*50)
    
    # Initialize dataset expansion
    expander = DatasetExpansion()
    
    # Run expansion
    expander.expand_dataset()
    
    print("\nDataset expansion completed successfully!")
    print("You can now use the expanded dataset for training your model.")

if __name__ == "__main__":
    main() 