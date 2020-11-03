import tensorflow as tf

def get_model(freeze=True):
    base_model = tf.keras.applications.MobileNetV2(weights='imagenet',
        include_top=False, input_shape=(224, 224, 3))
    
    head_model = base_model.output
    head_model = tf.keras.layers.AveragePooling2D(pool_size=(7, 7))(head_model)
    head_model = tf.keras.layers.Flatten()(head_model)
    head_model = tf.keras.layers.Dense(128, activation=tf.nn.relu)(head_model)
    head_model = tf.keras.layers.Dropout(0.5)(head_model)
    head_model = tf.keras.layers.Dense(2, activation=tf.nn.softmax)(head_model)

    model = tf.keras.Model(inputs=base_model.input, outputs=head_model)

    if freeze:
        for layer in model.layers:
            layer.trainable = False
    
    return model
