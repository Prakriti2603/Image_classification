import os
import shutil
import json

def update_app():
    """Update app.py to use the fixed model"""
    print("🔧 Updating app.py")
    print("="*50)
    
    # Backup original app.py
    if os.path.exists("app.py"):
        backup_path = "app.py.backup"
        shutil.copy2("app.py", backup_path)
        print(f"✅ Created backup of app.py at {backup_path}")
    
    # Read the original app.py
    with open("app.py", "r") as f:
        app_content = f.read()
    
    # Update model path
    app_content = app_content.replace(
        'model = tf.keras.models.load_model(\n            "Image_classify.keras",',
        'model = tf.keras.models.load_model(\n            "fixed_model.keras",'
    )
    
    # Add debug information for tomato/eggplant
    debug_info = '''
    # Add special debug for tomato/eggplant confusion
    if label in ['tomato', 'eggplant']:
        st.markdown("**Special Debug for Tomato/Eggplant:**")
        tomato_idx = class_names.index('tomato')
        eggplant_idx = class_names.index('eggplant')
        st.markdown(f"- Tomato confidence: {score[tomato_idx]*100:.2f}%")
        st.markdown(f"- Eggplant confidence: {score[eggplant_idx]*100:.2f}%")
        st.markdown(f"- Difference: {abs(score[tomato_idx] - score[eggplant_idx])*100:.2f}%")
    '''
    
    # Insert debug info after the top 3 predictions section
    app_content = app_content.replace(
        '            for i, idx in enumerate(top_3_indices):\n                st.markdown(f"  {i+1}. {class_names[idx]}: {score[idx]*100:.2f}%")',
        '            for i, idx in enumerate(top_3_indices):\n                st.markdown(f"  {i+1}. {class_names[idx]}: {score[idx]*100:.2f}%")' + debug_info
    )
    
    # Write the updated app.py
    with open("app.py", "w") as f:
        f.write(app_content)
    
    print("✅ Updated app.py to use fixed_model.keras")
    print("✅ Added special debug for tomato/eggplant confusion")

def create_readme():
    """Create a README file with instructions"""
    readme_content = """# Fruit and Vegetable Image Classification

## 🔧 Model Fix for Tomato/Eggplant Misclassification

This project has been updated to fix an issue where tomato images were being misclassified as eggplant.

### 🚀 How to Use

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Upload an image of a fruit or vegetable

3. View the prediction results

### 📋 Files

- `app.py`: The main Streamlit application
- `fixed_model.keras`: The fixed model that correctly classifies tomatoes
- `fixed_model_best.keras`: The best model from training
- `class_indices.json`: Class mapping for the model
- `test_model.py`: Script to test the model on specific images
- `fix_model.py`: Script used to fix the model

### 🔍 Debugging

If you encounter any issues with classification, you can use the `test_model.py` script:

```
python test_model.py path/to/image.jpg
```

This will show detailed prediction information including the top 5 predictions.

### 📊 Model Information

The model has been retrained with additional focus on distinguishing between tomatoes and eggplants. The app now includes special debug information when these classes are detected.
"""
    
    with open("README.md", "w") as f:
        f.write(readme_content)
    
    print("✅ Created README.md with instructions")

def main():
    """Main function to update the app"""
    print("🍎 App Update Tool")
    print("="*50)
    
    # Update app.py
    update_app()
    
    # Create README
    create_readme()
    
    print("\n🎉 Updates completed!")
    print("📁 Files updated:")
    print("  - app.py (updated to use fixed model)")
    print("  - README.md (created with instructions)")
    print("\n🚀 Next step: Run 'streamlit run app.py' to test the updated app!")

if __name__ == "__main__":
    main()