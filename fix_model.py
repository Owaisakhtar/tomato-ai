import tensorflow as tf

# Load your old model
old_model = tf.keras.models.load_model("best_model.h5", compile=False)

# Rebuild model with proper Input
inputs = tf.keras.Input(shape=old_model.input_shape[1:])
outputs = old_model(inputs)

new_model = tf.keras.Model(inputs, outputs)

# Save fixed model
new_model.save("best_model_fixed.h5")

print("Model fixed and saved as best_model_fixed.h5")
