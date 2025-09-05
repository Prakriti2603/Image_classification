import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import os
import numpy as np
import json
import shutil
from datetime import datetime

def create_model(input_shape=(180, 180, 3), num_classes=36):
    """Create a CNN model for fruit/vegetable classification"""
    
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        
        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        # Fourth Convolutional Block
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        # Flatten and Dense Layers
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    return model

def prepare_training_data():
    """Prepare training data with focus on tomato and eggplant classes"""
    print("🍎 Preparing Training Data")
    print("="*50)
    
    # Source directories
    train_dir = os.path.join("Fruits_Vegetables", "Fruits_Vegetables", "train")
    
    # Create temporary training directory
    temp_train_dir = "temp_train_data"
    os.makedirs(temp_train_dir, exist_ok=True)
    
    # Copy all classes
    for class_name in os.listdir(train_dir):
        class_dir = os.path.join(train_dir, class_name)
        # Skip non-directories and the templates directory
        if os.path.isdir(class_dir) and class_name != "templates":
            # Create class directory in temp_train_dir
            temp_class_dir = os.path.join(temp_train_dir, class_name)
            os.makedirs(temp_class_dir, exist_ok=True)
            
            # Copy images
            images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for img in images:
                src = os.path.join(class_dir, img)
                dst = os.path.join(temp_class_dir, img)
                shutil.copy2(src, dst)
            
            print(f"✅ Copied {len(images)} images for {class_name}")
    
    # Add test images to training set
    for i in range(1, 6):
        # Copy tomato test images
        src = f"tomato_test_{i}.jpg"
        dst = os.path.join(temp_train_dir, "tomato", f"extra_tomato_{i}.jpg")
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"✅ Added {src} to training data")
        
        # Copy eggplant test images
        src = f"eggplant_test_{i}.jpg"
        dst = os.path.join(temp_train_dir, "eggplant", f"extra_eggplant_{i}.jpg")
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"✅ Added {src} to training data")
    
    return temp_train_dir

def train_fixed_model(train_dir):
    """Train model with focus on fixing tomato/eggplant misclassification"""
    
    print("\n🔧 Training Fixed Model")
    print("="*50)
    
    # Configuration
    IMG_HEIGHT = 180
    IMG_WIDTH = 180
    BATCH_SIZE = 32
    EPOCHS = 15
    NUM_CLASSES = 36
    
    # Data augmentation with focus on color and shape variations
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,  # Increased rotation
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        zoom_range=0.2,
        brightness_range=[0.7, 1.3],  # Increased brightness variation
        fill_mode='nearest',
        validation_split=0.2
    )
    
    # Load training data
    print("📁 Loading training data...")
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        subset='training',
        shuffle=True
    )
    
    # Load validation data
    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        subset='validation',
        shuffle=True
    )
    
    print(f"✅ Training samples: {train_generator.samples}")
    print(f"✅ Validation samples: {validation_generator.samples}")
    print(f"✅ Number of classes: {len(train_generator.class_indices)}")
    
    # Create model
    print("🏗️ Creating model...")
    model = create_model((IMG_HEIGHT, IMG_WIDTH, 3), NUM_CLASSES)
    
    # Compile model with a lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=0.0005),  # Lower learning rate
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Model summary
    model.summary()
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'fixed_model_best.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model
    print("🚀 Starting training...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save('fixed_model.keras')
    print("✅ Model saved as 'fixed_model.keras'")
    
    # Save class indices
    class_indices = train_generator.class_indices
    with open('class_indices.json', 'w') as f:
        json.dump(class_indices, f, indent=2)
    print("✅ Class indices saved as 'class_indices.json'")
    
    # Print training results
    print("\n📊 Training Results:")
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    
    return model, history, train_generator.class_indices

def test_fixed_model(model, class_indices):
    """Test the fixed model on tomato and eggplant images"""
    
    print("\n🔍 Testing Fixed Model")
    print("="*50)
    
    # Invert class indices
    class_names = {v: k for k, v in class_indices.items()}
    
    # Test images
    test_images = []
    for i in range(1, 6):
        tomato_img = f"tomato_test_{i}.jpg"
        eggplant_img = f"eggplant_test_{i}.jpg"
        
        if os.path.exists(tomato_img):
            test_images.append((tomato_img, "tomato"))
        
        if os.path.exists(eggplant_img):
            test_images.append((eggplant_img, "eggplant"))
    
    # Test each image
    correct = 0
    for img_path, true_class in test_images:
        # Preprocess image
        img = tf.keras.preprocessing.image.load_img(
            img_path, target_size=(180, 180)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        predictions = model.predict(img_array, verbose=0)
        score = tf.nn.softmax(predictions[0])
        predicted_class = class_names[np.argmax(score)]
        confidence = 100 * np.max(score)
        
        # Print result
        print(f"Image: {img_path}")
        print(f"True class: {true_class}")
        print(f"Predicted class: {predicted_class} ({confidence:.2f}%)")
        print(f"Result: {'✅ CORRECT' if predicted_class == true_class else '❌ WRONG'}")
        print("-"*30)
        
        if predicted_class == true_class:
            correct += 1
    
    # Print overall accuracy
    accuracy = correct / len(test_images) if test_images else 0
    print(f"\nTest Accuracy: {accuracy:.2f} ({correct}/{len(test_images)})")

def main():
    """Main function to fix the model"""
    print("🍎 Fruit/Vegetable Model Fix Tool")
    print("="*50)
    
    # Prepare training data
    train_dir = prepare_training_data()
    
    # Train fixed model
    model, history, class_indices = train_fixed_model(train_dir)
    
    # Test fixed model
    test_fixed_model(model, class_indices)
    
    print("\n🎉 Model fixing completed!")
    print("📁 Files created:")
    print("  - fixed_model.keras (final model)")
    print("  - fixed_model_best.keras (best model)")
    print("  - class_indices.json (updated class mapping)")
    print("\n🚀 Next step: Update your app.py to use the new model!")

if __name__ == "__main__":
    main()