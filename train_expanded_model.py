import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import os
import numpy as np
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

def train_model_with_expanded_dataset():
    """Train model using the expanded dataset"""
    
    print("🍎 Training Model with Expanded Dataset")
    print("="*50)
    
    # Configuration
    IMG_HEIGHT = 180
    IMG_WIDTH = 180
    BATCH_SIZE = 32
    EPOCHS = 20
    NUM_CLASSES = 36
    
    # Data paths
    train_dir = "expanded_dataset"
    
    # Check if expanded dataset exists
    if not os.path.exists(train_dir):
        print(f"❌ Error: {train_dir} not found!")
        print("Please run dataset_expansion.py first")
        return None
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
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
        shuffle=True,
        classes=[d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d)) and d != "templates"]
    )
    
    # Load validation data
    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        subset='validation',
        shuffle=True,
        classes=[d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d)) and d != "templates"]
    )
    
    print(f"✅ Training samples: {train_generator.samples}")
    print(f"✅ Validation samples: {validation_generator.samples}")
    print(f"✅ Number of classes: {len(train_generator.class_indices)}")
    
    # Create model
    print("🏗️ Creating model...")
    model = create_model((IMG_HEIGHT, IMG_WIDTH, 3), NUM_CLASSES)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
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
            'best_expanded_model.keras',
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
    model.save('expanded_model.keras')
    print("✅ Model saved as 'expanded_model.keras'")
    
    # Save class indices
    class_indices = train_generator.class_indices
    import json
    with open('class_indices.json', 'w') as f:
        json.dump(class_indices, f, indent=2)
    print("✅ Class indices saved as 'class_indices.json'")
    
    # Print training results
    print("\n📊 Training Results:")
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    
    return model, history

def evaluate_model(model, validation_generator):
    """Evaluate the trained model"""
    
    print("\n🔍 Evaluating model...")
    
    # Evaluate on validation set
    loss, accuracy = model.evaluate(validation_generator, verbose=1)
    print(f"Validation Loss: {loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")
    
    # Predict on a few samples
    print("\n🎯 Sample Predictions:")
    for i in range(5):
        batch = next(validation_generator)
        predictions = model.predict(batch[0], verbose=0)
        predicted_class = int(np.argmax(predictions[0]))
        actual_class = int(batch[1][0])
        
        class_names = list(validation_generator.class_indices.keys())
        print(f"Sample {i+1}: Predicted={class_names[predicted_class]}, Actual={class_names[actual_class]}")

if __name__ == "__main__":
    print("🍎 Fruit/Vegetable Model Training with Expanded Dataset")
    print("="*60)
    
    # Train the model
    model, history = train_model_with_expanded_dataset()
    
    if model is not None:
        # Create validation generator for evaluation
        validation_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
        validation_generator = validation_datagen.flow_from_directory(
            "expanded_dataset",
            target_size=(180, 180),
            batch_size=32,
            class_mode='sparse',
            subset='validation',
            shuffle=False,
            classes=[d for d in os.listdir("expanded_dataset") if os.path.isdir(os.path.join("expanded_dataset", d)) and d != "templates"]
        )
        
        # Evaluate the model
        evaluate_model(model, validation_generator)
        
        print("\n🎉 Training completed successfully!")
        print("📁 Files created:")
        print("  - expanded_model.keras (final model)")
        print("  - best_expanded_model.keras (best model)")
        print("  - class_indices.json (class mapping)")
        print("\n🚀 Next step: Update your app.py to use the new model!")
    else:
        print("❌ Training failed. Please check the expanded dataset.")