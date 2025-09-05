import os
import shutil
from pathlib import Path

def organize_dataset():
    """Organize existing images into the required folder structure"""
    
    # Define the classes
    classes = [
        'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower',
        'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno',
        'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas',
        'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn',
        'sweetpotato', 'tomato', 'turnip', 'watermelon'
    ]
    
    # Create the main directory
    base_dir = "expanded_dataset"
    os.makedirs(base_dir, exist_ok=True)
    
    # Create folder structure for each class
    for class_name in classes:
        class_dir = os.path.join(base_dir, class_name)
        original_dir = os.path.join(class_dir, 'original')
        os.makedirs(original_dir, exist_ok=True)
        print(f"Created directory: {original_dir}")
    
    print(f"\n✅ Folder structure created!")
    print(f"📁 Base directory: {os.path.abspath(base_dir)}")
    print(f"\n📋 Next steps:")
    print(f"1. Move your existing images to the appropriate 'original' folders")
    print(f"2. For example, move capsicum images to: {base_dir}/capsicum/original/")
    print(f"3. Run: python dataset_expansion.py")

def move_images_to_structure():
    """Helper function to move images from current location to organized structure"""
    
    # Define source directories (modify these to match your current setup)
    source_dirs = [
        "your_current_images",  # Change this to your actual image folder
        "dataset",              # Or this folder
        "."                     # Or current directory
    ]
    
    # Define class mappings (modify based on your image names)
    class_mappings = {
        'capsicum': ['capsicum', 'capsicums', 'bell_pepper', 'pepper'],
        'bell pepper': ['bell_pepper', 'bellpepper', 'sweet_pepper'],
        'apple': ['apple', 'apples'],
        'banana': ['banana', 'bananas'],
        'carrot': ['carrot', 'carrots'],
        'tomato': ['tomato', 'tomatoes'],
        'chilli pepper': ['chilli', 'chili', 'chilli_pepper', 'chili_pepper', 'hot_pepper'],
        'corn': ['corn', 'maize', 'sweet_corn'],
        'cucumber': ['cucumber', 'cucumbers'],
        'eggplant': ['eggplant', 'aubergine', 'brinjal'],
        'garlic': ['garlic'],
        'ginger': ['ginger'],
        'grapes': ['grape', 'grapes'],
        'jalepeno': ['jalapeno', 'jalepeno', 'jalapeño'],
        'kiwi': ['kiwi', 'kiwifruit'],
        'lemon': ['lemon', 'lemons'],
        'lettuce': ['lettuce'],
        'mango': ['mango', 'mangoes'],
        'onion': ['onion', 'onions'],
        'orange': ['orange', 'oranges'],
        'paprika': ['paprika'],
        'pear': ['pear', 'pears'],
        'peas': ['pea', 'peas'],
        'pineapple': ['pineapple', 'pineapples'],
        'pomegranate': ['pomegranate', 'pomegranates'],
        'potato': ['potato', 'potatoes'],
        'raddish': ['radish', 'raddish', 'radishes'],
        'soy beans': ['soy', 'soybean', 'soy_beans'],
        'spinach': ['spinach'],
        'sweetcorn': ['sweetcorn', 'sweet_corn'],
        'sweetpotato': ['sweet_potato', 'sweetpotato', 'yam'],
        'turnip': ['turnip', 'turnips'],
        'beetroot': ['beetroot', 'beet', 'beets'],
        'cabbage': ['cabbage'],
        'cauliflower': ['cauliflower'],
        'watermelon': ['watermelon', 'watermelons'],
        # Add more mappings as needed
    }
    
    base_dir = "expanded_dataset"
    
    for source_dir in source_dirs:
        if not os.path.exists(source_dir):
            continue
            
        print(f"Scanning {source_dir} for images...")
        
        for file in os.listdir(source_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_lower = file.lower()
                
                # Find matching class
                target_class = None
                for class_name, keywords in class_mappings.items():
                    if any(keyword in file_lower for keyword in keywords):
                        target_class = class_name
                        break
                
                if target_class:
                    # Move file to appropriate directory
                    target_dir = os.path.join(base_dir, target_class, 'original')
                    target_path = os.path.join(target_dir, file)
                    
                    source_path = os.path.join(source_dir, file)
                    
                    try:
                        shutil.copy2(source_path, target_path)
                        print(f"✅ Moved {file} to {target_class}/original/")
                    except Exception as e:
                        print(f"❌ Error moving {file}: {e}")
                else:
                    print(f"⚠️  Could not classify {file} - please move manually")

if __name__ == "__main__":
    print("🍎 Dataset Organization Helper")
    print("="*40)
    
    # Create folder structure
    organize_dataset()
    
    # Ask if user wants to auto-move images
    response = input("\nDo you want to automatically move images? (y/n): ")
    if response.lower() == 'y':
        move_images_to_structure()
    else:
        print("\n📝 Manual organization required:")
        print("1. Move your images to the appropriate 'original' folders")
        print("2. Example: move capsicum images to expanded_dataset/capsicum/original/")
        print("3. Then run: python dataset_expansion.py") 