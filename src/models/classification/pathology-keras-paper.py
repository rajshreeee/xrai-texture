import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import itertools

# ========= 1. Paths and Parameters =========
train_dir = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/train/masked_images"
test_dir = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/test/masked_images"
csv_file = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/CBIS_DDSM_PATCHED_ANNOTATIONS_CONTEXT.csv"

image_size = (224, 224)
batch_size = 32
num_epochs = 40
save_dir = "keras_training_outputs"
os.makedirs(save_dir, exist_ok=True)

# ========= 2. Load Dataset =========
df_csv = pd.read_csv(csv_file)
# df = df[df['pathology'].str.lower().isin(['benign', 'malignant'])].reset_index(drop=True)
# label_mapping = {'benign': 0, 'malignant': 1}

# def get_image_path(row, folders):
#     img_base = row['image_name']
#     for folder in folders:
#         candidates = [f for f in os.listdir(folder) if img_base in f]
#         if candidates:
#             return os.path.join(folder, candidates[0])
#     return None

# folders = [train_dir, test_dir]
# df['image_path'] = df.apply(lambda row: get_image_path(row, folders), axis=1)
# df = df[df['image_path'].notnull()].reset_index(drop=True)
# df['label'] = df['pathology'].str.lower().map(label_mapping)

# print(f"[INFO] Total samples after merging: {len(df)}")

# df_csv = df_csv[df_csv['pathology'].str.lower().isin(['benign', 'malignant'])].reset_index(drop=True)
# print(f"[INFO] CSV entries after filtering benign/malignant: {len(df_csv)}")

# # Map labels
# label_mapping = {'benign': 0, 'malignant': 1}

# # ========= 3. Load all images from both folders =========
# image_files = []
# for folder in [train_dir, test_dir]:
#     folder_images = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
#     image_files.extend(folder_images)

# print(f"[INFO] Total images found (train+test): {len(image_files)}")

# # ========= 4. Build final dataframe =========

# records = []

# # Build a quick lookup for CSV metadata
# csv_lookup = {row['image_name']: row for _, row in df_csv.iterrows()}

# for image_path in image_files:
#     filename = os.path.basename(image_path)
#     base_name = filename.split('#')[0]  # Remove augmentation suffix if present
#     base_name = os.path.splitext(base_name)[0]  # Remove .png/.jpg
    
#     # Check if base_name exists in original CSV
#     if base_name in csv_lookup:
#         pathology = csv_lookup[base_name]['pathology'].lower()
#         label = label_mapping[pathology]
#         records.append({
#             'image_path': image_path,
#             'label': label
#         })

# final_df = pd.DataFrame(records)

# print(f"[INFO] Final dataset size after matching: {len(final_df)}")

# # Shuffle it!
# final_df = final_df.sample(frac=1.0, random_state=42).reset_index(drop=True)


# # ========= 3. Split into Train/Test =========
# train_df, test_df = train_test_split(final_df, test_size=0.2, stratify=final_df['label'], random_state=42)

# train_df["label"] = train_df["label"].map({0: "benign", 1: "malignant"})
# test_df["label"] = test_df["label"].map({0: "benign", 1: "malignant"})


df_csv = pd.read_csv(csv_file)
df_csv['pathology'] = df_csv['pathology'].str.lower()
df_csv['pathology'] = df_csv['pathology'].replace('benign_without_callback', 'benign')

df_csv = df_csv[df_csv['pathology'].isin(['benign', 'malignant'])].reset_index(drop=True)
print(f"[INFO] CSV entries after filtering benign/malignant: {len(df_csv)}")

label_mapping = {'benign': 0, 'malignant': 1}

# ========= 3. Build CSV lookup =========
csv_lookup = {row['image_name']: row for _, row in df_csv.iterrows()}

# ========= 4. Load Training Images =========
train_records = []

train_images = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for image_path in train_images:
    filename = os.path.basename(image_path)
    base_name = filename.split('#')[0]  # Remove augmentation suffix if any
    base_name = os.path.splitext(base_name)[0]
    
    if base_name in csv_lookup:
        pathology = csv_lookup[base_name]['pathology'].lower()
        label = label_mapping[pathology]
        train_records.append({
            'image_path': image_path,
            'label': label
        })

train_df = pd.DataFrame(train_records)
print(f"[INFO] Training set size: {len(train_df)}")

# ========= 5. Load Testing Images =========
test_records = []

test_images = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for image_path in test_images:
    filename = os.path.basename(image_path)
    base_name = filename.split('#')[0]  # remove augmentation suffix if any
    base_name = os.path.splitext(base_name)[0]
    
    if base_name in csv_lookup:
        pathology = csv_lookup[base_name]['pathology'].lower()
        label = label_mapping[pathology]
        test_records.append({
            'image_path': image_path,
            'label': label
        })

test_df = pd.DataFrame(test_records)
print(f"[INFO] Testing set size: {len(test_df)}")

# ========= 6. Final Processing =========
# Map label 0/1 to string for flow_from_dataframe
train_df["label"] = train_df["label"].map({0: "benign", 1: "malignant"})
test_df["label"] = test_df["label"].map({0: "benign", 1: "malignant"})

print("[INFO] Dataframes ready for training/testing.")


# ========= 4. Data Augmentation =========
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=keras.applications.resnet_v2.preprocess_input,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.1,
    brightness_range=[0.8, 1.2]
)

test_datagen = keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=keras.applications.resnet_v2.preprocess_input
)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col="image_path",
    y_col="label",
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True
)

test_generator = test_datagen.flow_from_dataframe(
    test_df,
    x_col="image_path",
    y_col="label",
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)

# ========= 5. Model =========
base_model = keras.applications.ResNet50V2(include_top=False, pooling='avg', input_shape=(224,224,3))
for layer in base_model.layers[-60:]:  # Unfreeze last ~40 layers
    if not isinstance(layer, keras.layers.BatchNormalization):
        layer.trainable = True

x = base_model.output
x = layers.Dense(1024, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4))(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(2, activation='softmax')(x)

model = keras.models.Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.25),
    metrics=["accuracy"]
)

# ========= 6. Callbacks =========
checkpoint_path = os.path.join(save_dir, "resnet50v2_best_model.h5")

checkpoint = ModelCheckpoint(
    checkpoint_path,
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
    mode='max'
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,
    patience=2,
    verbose=1,
    mode='max',
    min_lr=1e-5
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

callbacks_list = [checkpoint, reduce_lr, early_stop]

# ========= 7. Training =========
steps_per_epoch = len(train_generator)
validation_steps = len(test_generator)

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=test_generator,
    validation_steps=validation_steps,
    epochs=num_epochs,
    callbacks=callbacks_list,
    verbose=1
)

# ========= 8. Load Best Model and Evaluate =========
model.load_weights(checkpoint_path)

loss, accuracy = model.evaluate(test_generator, steps=validation_steps)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# ========= 9. Confusion Matrix =========
predictions = model.predict(test_generator, steps=validation_steps, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)

true_classes = test_generator.labels
class_labels = ["Benign", "Malignant"]

cm = confusion_matrix(true_classes, predicted_classes)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(true_classes, predicted_classes, target_names=class_labels))
print("\nROC AUC Score:\n", roc_auc_score(true_classes, predicted_classes))

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=45)
    plt.yticks(ticks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

plt.figure(figsize=(8,6))
plot_confusion_matrix(cm, class_labels)
plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
plt.close()

# ========= 10. Plot Training Curves =========
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))  # Update in case of EarlyStopping

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.savefig(os.path.join(save_dir, "training_curves.png"))
plt.close()

print(f"[INFO] Training completed and results saved to {save_dir}")