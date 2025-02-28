"""Transform module for diabetes prediction pipeline."""

import tensorflow as tf
import tensorflow_transform as tft

# Define feature columns
NUMERIC_FEATURES = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
CATEGORICAL_FEATURES = ['gender', 'smoking_history']
BINARY_FEATURES = ['hypertension', 'heart_disease']
LABEL_KEY = 'diabetes'

def preprocessing_fn(inputs):
    """Preprocessing function for TFX Transform component."""
    outputs = {}
    
    # Scale numeric features
    for feature in NUMERIC_FEATURES:
        outputs[feature] = tft.scale_to_z_score(inputs[feature])
    
    # Convert categorical features to indices
    for feature in CATEGORICAL_FEATURES:
        outputs[feature] = tft.compute_and_apply_vocabulary(
            inputs[feature], vocab_filename=feature)
    
    # Pass through binary features
    for feature in BINARY_FEATURES:
        outputs[feature] = tf.cast(inputs[feature], tf.float32)
    
    # Pass through the label
    outputs[LABEL_KEY] = tf.cast(inputs[LABEL_KEY], tf.float32)
    
    return outputs
