"""
CNN (Convolutional Neural Network) - Full Pipeline
Dataset : CIFAR-10 (60,000 color images, 10 classes)
Features: data augmentation, VGG-style conv blocks with BatchNorm + Dropout,
          global average pooling, callbacks, training curves, confusion matrix,
          classification report, feature-map visualization, model saving

Install : pip install tensorflow numpy matplotlib scikit-learn
Run     : python cnn_full.py
Tip     : a GPU makes this much faster; on CPU reduce EPOCHS.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
EPOCHS = 40
BATCH_SIZE = 64
OUT_DIR = "cnn_outputs"
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

np.random.seed(SEED)
keras.utils.set_random_seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

(x_train_full, y_train_full), (x_test, y_test) = keras.datasets.cifar10.load_data()
y_train_full = y_train_full.flatten()
y_test = y_test.flatten()
print("Train:", x_train_full.shape, "| Test:", x_test.shape)

x_train_full = x_train_full.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

val_size = 5000
x_val, y_val = x_train_full[:val_size], y_train_full[:val_size]
x_train, y_train = x_train_full[val_size:], y_train_full[val_size:]
print(f"Train: {x_train.shape} | Val: {x_val.shape} | Test: {x_test.shape}")

fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(x_train[i])
    ax.set_title(CLASS_NAMES[y_train[i]])
    ax.axis("off")
plt.suptitle("Sample CIFAR-10 images")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/samples.png", dpi=120)
plt.close()

def conv_block(x, filters, dropout):
    """Two Conv-BN-ReLU layers followed by MaxPool and Dropout."""
    for _ in range(2):
        x = layers.Conv2D(filters, (3, 3), padding="same", use_bias=False,
                          kernel_regularizer=keras.regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout)(x)
    return x

def build_cnn(input_shape=(32, 32, 3), num_classes=10):
    inputs = keras.Input(shape=input_shape)

    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.08)(x)
    x = layers.RandomZoom(0.1)(x)
    x = layers.RandomTranslation(0.1, 0.1)(x)

    x = conv_block(x, 32, 0.2)
    x = conv_block(x, 64, 0.3)
    x = conv_block(x, 128, 0.4)

    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu",
                      name="last_conv")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs, name="cnn_cifar10")

model = build_cnn()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                  restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                      patience=3, min_lr=1e-6, verbose=1),
    keras.callbacks.ModelCheckpoint(f"{OUT_DIR}/best_cnn.keras",
                                    monitor="val_accuracy",
                                    save_best_only=True, verbose=0),
]

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=2,
)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(history.history["accuracy"], label="train")
ax[0].plot(history.history["val_accuracy"], label="validation")
ax[0].set_title("Accuracy"); ax[0].set_xlabel("Epoch"); ax[0].legend()

ax[1].plot(history.history["loss"], label="train")
ax[1].plot(history.history["val_loss"], label="validation")
ax[1].set_title("Loss"); ax[1].set_xlabel("Epoch"); ax[1].legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/training_curves.png", dpi=120)
plt.close()

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest loss    : {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4f}")

y_prob = model.predict(x_test, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

print("\nClassification report:\n")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4))

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm, cmap="Blues")
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_xticks(range(10)); ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
ax.set_yticks(range(10)); ax.set_yticklabels(CLASS_NAMES)
for i in range(10):
    for j in range(10):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.colorbar(im)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=120)
plt.close()

fig, axes = plt.subplots(3, 6, figsize=(13, 7))
for ax, i in zip(axes.ravel(), range(18)):
    ax.imshow(x_test[i])
    ok = y_pred[i] == y_test[i]
    ax.set_title(f"{CLASS_NAMES[y_pred[i]]} ({y_prob[i].max():.0%})\n"
                 f"true: {CLASS_NAMES[y_test[i]]}",
                 color="green" if ok else "red", fontsize=8)
    ax.axis("off")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/predictions.png", dpi=120)
plt.close()

conv_layers = [l for l in model.layers if isinstance(l, layers.Conv2D)][:2]
feature_model = keras.Model(model.inputs, [l.output for l in conv_layers])
maps = feature_model.predict(x_test[:1], verbose=0)

for layer, fmap in zip(conv_layers, maps):
    n = min(16, fmap.shape[-1])
    fig, axes = plt.subplots(2, 8, figsize=(14, 4))
    for k, ax in enumerate(axes.ravel()[:n]):
        ax.imshow(fmap[0, :, :, k], cmap="viridis")
        ax.axis("off")
    plt.suptitle(f"Feature maps - {layer.name}")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/feature_maps_{layer.name}.png", dpi=120)
    plt.close()

model.save(f"{OUT_DIR}/cnn_final.keras")
loaded = keras.models.load_model(f"{OUT_DIR}/cnn_final.keras")
p = loaded.predict(x_test[:1], verbose=0)[0]
print(f"\nSingle prediction -> {CLASS_NAMES[np.argmax(p)]} ({p.max():.2%}), "
      f"true: {CLASS_NAMES[y_test[0]]}")
print(f"\nAll outputs saved in: {OUT_DIR}/")
