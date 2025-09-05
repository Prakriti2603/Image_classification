import os
import shutil

def manual_classify_specific_files():
    """Manually classify the specific files that couldn't be auto-classified"""
    
    # Define the problematic files and their correct classes
    manual_mappings = {
        'chilli.jpg': 'chilli pepper',
        'corn.jpg': 'corn', 
        'grocery (1).png': 'unknown'  # This might be a general grocery image
    }
    
    base_dir = "expanded_dataset"
    
    print("🔍 Manual Classification Helper")
    print("="*40)
    
    for filename, suggested_class in manual_mappings.items():
        print(f"\n📁 File: {filename}")
        print(f"🤖 Suggested class: {suggested_class}")
        
        if suggested_class == 'unknown':
            print("❓ This appears to be a general grocery image.")
            print("💡 You can either:")
            print("   1. Skip this file (it's not a specific fruit/vegetable)")
            print("   2. Manually decide which class it belongs to")
            
            choice = input("Skip this file? (y/n): ")
            if choice.lower() == 'y':
                print(f"⏭️  Skipping {filename}")
                continue
            else:
                # Show available classes
                print("\nAvailable classes:")
                classes = [
                    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 
                    'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant',
                    'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce',
                    'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple',
                    'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn',
                    'sweetpotato', 'tomato', 'turnip', 'watermelon'
                ]
                
                for i, class_name in enumerate(classes, 1):
                    print(f"   {i:2d}. {class_name}")
                
                try:
                    choice = int(input("\nEnter the number for the correct class: "))
                    if 1 <= choice <= len(classes):
                        suggested_class = classes[choice - 1]
                        print(f"✅ Classified as: {suggested_class}")
                    else:
                        print("❌ Invalid choice, skipping...")
                        continue
                except ValueError:
                    print("❌ Invalid input, skipping...")
                    continue
        
        # Move the file
        source_path = filename
        target_dir = os.path.join(base_dir, suggested_class, 'original')
        target_path = os.path.join(target_dir, filename)
        
        if os.path.exists(source_path):
            try:
                # Create target directory if it doesn't exist
                os.makedirs(target_dir, exist_ok=True)
                
                # Move the file
                shutil.copy2(source_path, target_path)
                print(f"✅ Moved {filename} to {suggested_class}/original/")
                
                # Optionally remove the original
                remove_original = input(f"Remove original {filename}? (y/n): ")
                if remove_original.lower() == 'y':
                    os.remove(source_path)
                    print(f"🗑️  Removed original {filename}")
                    
            except Exception as e:
                print(f"❌ Error moving {filename}: {e}")
        else:
            print(f"⚠️  File {filename} not found in current directory")
    
    print(f"\n✅ Manual classification complete!")
    print(f"📁 Check the expanded_dataset folder for organized images")

def check_current_files():
    """Check what files are currently in the directory"""
    print("📋 Current files in directory:")
    print("="*40)
    
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
    files = [f for f in os.listdir('.') if f.lower().endswith(image_extensions)]
    
    if not files:
        print("No image files found in current directory")
        return
    
    for i, filename in enumerate(files, 1):
        print(f"{i:2d}. {filename}")
    
    print(f"\nTotal: {len(files)} image files")

if __name__ == "__main__":
    print("🍎 Manual Image Classification Helper")
    print("="*50)
    
    # First, show what files are available
    check_current_files()
    
    # Then help with manual classification
    print(f"\n{'='*50}")
    manual_classify_specific_files()
    
    print(f"\n🎯 Next steps:")
    print("1. Run: python organize_dataset.py (to catch any remaining files)")
    print("2. Run: python dataset_expansion.py (to expand your dataset)")
    print("3. Train your model with the expanded dataset") 