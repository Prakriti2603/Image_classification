# 🍎 Fruit/Vegetable Dataset Expansion Tool

This tool helps you expand your fruit and vegetable dataset using multiple techniques to improve your model's performance.

## 🚀 Features

### 1. **Web Scraping**
- Automatically downloads images from Unsplash
- Multiple search terms for better variety
- Quality validation for downloaded images

### 2. **Data Augmentation**
- Rotation, scaling, and brightness adjustments
- Horizontal flipping
- Zoom and shift transformations
- Generates 5 augmented versions per original image

### 3. **Synthetic Image Generation**
- Placeholder for GAN-based synthetic images
- Basic synthetic image generation
- Expandable for advanced techniques

### 4. **Quality Control**
- Image size validation
- Aspect ratio checks
- Brightness validation
- Corruption detection

### 5. **Public Dataset Integration**
- Framework for downloading public datasets
- Easy integration with Kaggle datasets
- Automatic dataset organization

## 📁 Directory Structure

```
expanded_dataset/
├── apple/
│   ├── original/      # Original images
│   ├── augmented/     # Augmented images
│   ├── scraped/       # Web-scraped images
│   └── synthetic/     # Synthetic images
├── banana/
│   ├── original/
│   ├── augmented/
│   ├── scraped/
│   └── synthetic/
└── ... (for all 36 classes)
```

## 🛠️ Installation

1. **Install Dependencies:**
```bash
pip install -r requirements_dataset.txt
```

2. **Run the Expansion Tool:**
```bash
python dataset_expansion.py
```

## 📊 Usage

### Basic Usage
```python
from dataset_expansion import DatasetExpansion

# Initialize the expansion tool
expander = DatasetExpansion(base_dir="my_expanded_dataset")

# Expand dataset for all classes
expander.expand_dataset()
```

### Custom Configuration
```python
# Expand specific classes only
for class_name in ['capsicum', 'bell pepper', 'chilli pepper']:
    expander.web_scraping(class_name, num_images=100)
    expander.augment_dataset(class_name, augmentations_per_image=10)
    expander.quality_control(class_name)
```

## 🎯 Target Classes

The tool supports 36 fruit and vegetable classes:
- **Fruits**: apple, banana, grapes, kiwi, lemon, mango, orange, pear, pineapple, pomegranate, watermelon
- **Vegetables**: beetroot, bell pepper, cabbage, capsicum, carrot, cauliflower, chilli pepper, corn, cucumber, eggplant, garlic, ginger, jalepeno, lettuce, onion, paprika, peas, raddish, soy beans, spinach, sweetcorn, sweetpotato, tomato, turnip

## 📈 Expected Results

After running the expansion tool, you can expect:
- **50+ scraped images** per class
- **5x augmented images** per original image
- **20 synthetic images** per class
- **Quality-controlled dataset** with valid images only

## 🔧 Customization

### Modify Search Terms
```python
# In web_scraping method
search_terms = [
    f"{class_name} fresh",
    f"{class_name} organic",
    f"{class_name} close up",
    f"{class_name} on white background",
    f"{class_name} high resolution"  # Add custom terms
]
```

### Adjust Augmentation Parameters
```python
# In augment_dataset method
datagen = ImageDataGenerator(
    rotation_range=30,        # Increase rotation
    width_shift_range=0.3,    # Increase shift
    height_shift_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,       # Enable vertical flip
    zoom_range=0.3,          # Increase zoom
    brightness_range=[0.7, 1.3],  # Wider brightness range
    fill_mode='reflect'       # Change fill mode
)
```

### Quality Control Thresholds
```python
# In quality_control method
if img.shape[0] >= 150 and img.shape[1] >= 150:  # Increase minimum size
    aspect_ratio = img.shape[1] / img.shape[0]
    if 0.3 <= aspect_ratio <= 3.0:  # Wider aspect ratio
        mean_brightness = np.mean(gray)
        if 15 <= mean_brightness <= 240:  # Wider brightness range
```

## 📊 Monitoring Progress

The tool provides detailed progress information:
- Real-time progress bars
- Per-class statistics
- Quality control results
- Final dataset summary

## 🎯 Integration with Training

After expansion, use the dataset for training:

```python
# Load expanded dataset
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    'expanded_dataset',
    target_size=(180, 180),
    batch_size=32,
    class_mode='sparse',
    subset='training'
)
```

## 🔍 Troubleshooting

### Common Issues:
1. **Web scraping fails**: Check internet connection and website availability
2. **Memory issues**: Reduce batch size or number of augmentations
3. **Quality control too strict**: Adjust thresholds in quality_control method
4. **Slow processing**: Use GPU acceleration if available

### Performance Tips:
- Use SSD storage for faster I/O
- Enable GPU acceleration for TensorFlow
- Increase batch size for faster processing
- Use parallel processing for multiple classes

## 📝 License

This tool is provided as-is for educational and research purposes.

## 🤝 Contributing

Feel free to contribute improvements:
- Add more web scraping sources
- Implement advanced synthetic image generation
- Add more quality control checks
- Optimize performance

---

**Happy Dataset Expansion! 🍎🥕🥬** 