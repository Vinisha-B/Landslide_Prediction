import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ----------------------------
# 1️⃣ Data Preprocessing
# ----------------------------
train_path = "dataset/train"
test_path = "dataset/test"

# Image augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

# Only rescale for test
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

test_data = test_datagen.flow_from_directory(
    test_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# ----------------------------
# 2️⃣ Build the Model
# ----------------------------
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False  # freeze pretrained weights

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ----------------------------
# 3️⃣ Callbacks
# ----------------------------
checkpoint = ModelCheckpoint(
    "model/landslide_model.h5",
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)

# ----------------------------
# 4️⃣ Train the Model
# ----------------------------
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=15,              # you can increase to 20–30 if GPU available
    callbacks=[checkpoint, early_stop]
)

# ----------------------------
# 5️⃣ Save the Final Model
# ----------------------------
model.save("model/landslide_model.h5")
print("✅ Model training completed and saved!")