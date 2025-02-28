"""Trainer module for diabetes prediction pipeline."""

import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.keras import layers
from tfx.components.trainer.fn_args_utils import FnArgs

# Define feature columns
NUMERIC_FEATURES = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
CATEGORICAL_FEATURES = ['gender', 'smoking_history']
BINARY_FEATURES = ['hypertension', 'heart_disease']
LABEL_KEY = 'diabetes'

def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Returns a function that parses a serialized tf.Example."""
    
    model.tft_layer = tf_transform_output.transform_features_layer()
    
    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        """Returns the output to be used in the serving signature."""
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        return model(transformed_features)
    
    return serve_tf_examples_fn

def _input_fn(file_pattern, tf_transform_output, batch_size=32):
    """Generates features and label for training."""
    
    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy())
    
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=tf.data.TFRecordDataset,
        label_key=LABEL_KEY)
    
    return dataset

def _build_keras_model(tf_transform_output):
    """Creates a DNN Keras model for classifying diabetes."""
    
    # Get transformed feature specifications
    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy())
    transformed_feature_spec.pop(LABEL_KEY)
    
    # Create inputs for each feature
    inputs = {}
    encoded_features = []
    
    # Process numeric features
    for feature in NUMERIC_FEATURES:
        inputs[feature] = tf.keras.Input(shape=(1,), name=feature)
        encoded_features.append(inputs[feature])
    
    # Process categorical features
    for feature in CATEGORICAL_FEATURES:
        vocab_size = tf_transform_output.vocabulary_size_by_name(feature)
        inputs[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.int64)
        embedding_dim = min(50, (vocab_size + 1) // 2)
        embedding = layers.Embedding(vocab_size + 1, embedding_dim)(inputs[feature])
        embedding_flat = layers.Flatten()(embedding)
        encoded_features.append(embedding_flat)
    
    # Process binary features
    for feature in BINARY_FEATURES:
        inputs[feature] = tf.keras.Input(shape=(1,), name=feature)
        encoded_features.append(inputs[feature])
    
    # Combine all features
    x = layers.Concatenate()(encoded_features)
    
    # Build DNN layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation='relu')(x)
    
    # Output layer
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.AUC(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()
        ]
    )
    
    return model

def run_fn(fn_args: FnArgs):
    """Train the model based on given args."""
    
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    
    train_dataset = _input_fn(fn_args.train_files, tf_transform_output, 32)
    eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output, 32)
    
    model = _build_keras_model(tf_transform_output)
    
    # Define callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=5,
            restore_best_weights=True,
            mode='max'
        )
    ]
    
    # Train the model
    model.fit(
        train_dataset,
        validation_data=eval_dataset,
        epochs=20,
        callbacks=callbacks
    )
    
    # Save the model
    signatures = {
        'serving_default':
            _get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
                tf.TensorSpec(shape=[None], dtype=tf.string, name='examples'))
    }
    
    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
