import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import joblib
import os

models_dir = r'E:\Projects\Travel_Recommender\models'

# تحميل ما تم حفظه من preprocessing
print("Loading preprocessed data...")
final_df = pd.read_csv(os.path.join(models_dir, 'final_df.csv'))
item_vectors = np.load(os.path.join(models_dir, 'item_vectors.npy'))
vectorizer = joblib.load(os.path.join(models_dir, 'tfidf_vectorizer.joblib'))

print(f"Loaded final_df shape: {final_df.shape}")
print(f"Loaded item_vectors shape: {item_vectors.shape}")

# ========================
# بناء وتدريب النموذج العصبي
# ========================
X = item_vectors
y = final_df['average_rating'].values / 5.0  # Normalize 1-5 → 0-1

model = Sequential([
    Dense(512, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Output 0-1
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
              loss='mse',
              metrics=['mae'])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
]

print("\nStarting neural network training...")
history = model.fit(
    X, y,
    epochs=50,
    batch_size=64,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# حفظ الموديل
model.save(os.path.join(models_dir, 'neural_recommend_model.keras'))
print("Neural model trained and saved successfully!")

# رسم Loss Curve
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training and Validation Loss Curve')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(models_dir, 'loss_curve.png'))
plt.show()

# تقرير نهائي
final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]
final_mae = history.history['mae'][-1]
print("\nTraining Summary:")
print(f"Final Training Loss (MSE): {final_train_loss:.6f}")
print(f"Final Validation Loss (MSE): {final_val_loss:.6f}")
print(f"Final MAE: {final_mae:.6f}")
print(f"Best epoch (early stopping): {len(history.history['loss'])}")
print("\nAll done! Model saved in:", os.path.join(models_dir, 'neural_recommend_model.keras'))
print("Loss curve saved as loss_curve.png")