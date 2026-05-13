import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import classification_report

# ── 1. LOAD & PREPARE DATA ──────────────────────────────────────────────────

print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixels from 0-255 to 0.0-1.0, add channel dimension
x_train = (x_train.astype('float32') / 255.0)[..., None]
x_test  = (x_test.astype('float32')  / 255.0)[..., None]

print(f"Training samples : {len(x_train)}")
print(f"Testing  samples : {len(x_test)}")


# ── 2. BUILD THE MODEL ───────────────────────────────────────────────────────

print("\nBuilding model...")
model = models.Sequential([
    # --- Feature extraction (eyes of the network) ---
    layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),

    # --- Classification (brain of the network) ---
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),               # prevents over-memorising
    layers.Dense(10, activation='softmax')  # 10 outputs: one per digit
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


# ── 3. CALLBACKS ─────────────────────────────────────────────────────────────

# Early stopping: stop training if val_accuracy doesn't improve for 3 epochs
# restore_best_weights: automatically keeps the best version of the model
early_stop = callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=3,
    restore_best_weights=True,
    verbose=1
)


# ── 4. TRAIN ─────────────────────────────────────────────────────────────────

print("\nTraining...")
history = model.fit(
    x_train, y_train,
    epochs=10,                  # up to 10 epochs (early stop may cut it short)
    batch_size=128,
    validation_split=0.1,       # use 10% of training data to monitor progress
    callbacks=[early_stop],
    verbose=1
)


# ── 5. EVALUATE ON TEST SET ──────────────────────────────────────────────────

print("\nEvaluating on test set...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy : {test_acc * 100:.2f}%")
print(f"Test loss     : {test_loss:.4f}")


# ── 6. PER-DIGIT ACCURACY REPORT ─────────────────────────────────────────────

print("\nPer-digit accuracy report:")
y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
print(classification_report(
    y_test, y_pred,
    target_names=[f"Digit {i}" for i in range(10)]
))


# ── 7. SAVE TRAINING GRAPH ───────────────────────────────────────────────────

epochs_ran = len(history.history['accuracy'])

plt.figure(figsize=(10, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs_ran + 1), history.history['accuracy'],     label='Train accuracy', marker='o')
plt.plot(range(1, epochs_ran + 1), history.history['val_accuracy'], label='Val accuracy',   marker='o')
plt.title('Accuracy over epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs_ran + 1), history.history['loss'],     label='Train loss', marker='o')
plt.plot(range(1, epochs_ran + 1), history.history['val_loss'], label='Val loss',   marker='o')
plt.title('Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
os.makedirs('models', exist_ok=True)
plt.savefig('models/training_history.png', dpi=150)
print("\nTraining graph saved → models/training_history.png")


# ── 8. SAVE MODEL ────────────────────────────────────────────────────────────

model.save('models/mnist_cnn.h5')
print("Model saved      → models/mnist_cnn.h5")
print("\nDone!")