import tensorflow as tf

def initialize_unet():
    '''
    Initialize and return a U-Net model
    
    '''
    inputs = tf.keras.layers.Input(shape = (128, 128, 2))

    conv0 = tf.keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same')(inputs)
    pool0 = tf.keras.layers.MaxPool2D((2,2))(conv0)

    conv1 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same')(pool0)
    pool1 = tf.keras.layers.MaxPool2D((2,2))(conv1)

    conv2 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same')(pool1)
    pool2 = tf.keras.layers.MaxPool2D((2,2))(conv2)

    conv3 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same')(pool2)
    pool3 = tf.keras.layers.MaxPool2D((2,2))(conv3)

    conv4 = tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same')(pool3)
    pool4 = tf.keras.layers.MaxPool2D((2,2))(conv4)

    upsample0 = tf.keras.layers.UpSampling2D((2,2))(pool4)
    conv5 = tf.keras.layers.Conv2D(512, 1, activation = 'relu', padding = 'same')(upsample0)
    concat0 = tf.keras.layers.concatenate([conv5, conv4], axis = -1)
    conv6 = tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same')(concat0)

    upsample1 = tf.keras.layers.UpSampling2D((2,2))(conv6)
    conv7 = tf.keras.layers.Conv2D(256, 1, activation = 'relu', padding = 'same')(upsample1)
    concat1 = tf.keras.layers.concatenate([conv7, conv3], axis = -1)
    conv8 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same')(concat1)

    upsample2 = tf.keras.layers.UpSampling2D((2,2))(conv8)
    conv9 = tf.keras.layers.Conv2D(128, 1, activation = 'relu', padding = 'same')(upsample2)
    concat2 = tf.keras.layers.concatenate([conv9, conv2], axis = -1)
    conv10 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same')(concat2)

    upsample3 = tf.keras.layers.UpSampling2D((2,2))(conv10)
    conv11 = tf.keras.layers.Conv2D(64, 1, activation = 'relu', padding = 'same')(upsample3)
    concat3 = tf.keras.layers.concatenate([conv11, conv1], axis = -1)
    conv12 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same')(concat3)

    upsample4 = tf.keras.layers.UpSampling2D((2,2))(conv12)
    conv13 = tf.keras.layers.Conv2D(32, 1, activation = 'relu', padding = 'same')(upsample4)
    concat4 = tf.keras.layers.concatenate([conv13, conv0], axis = -1)
    conv14 = tf.keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same')(concat4)
    
    outputs = tf.keras.layers.Conv2D(3, 1)(conv14)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="u-net")
    
    model.summary()
    
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0008),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    return model