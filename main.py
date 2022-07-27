import tensorflow as tf

img_height = 35
img_width = 35
batch_size_train = 5
batch_size_test = 2

my_train = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/",
    labels='inferred',
    label_mode='int',
    class_names=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
    color_mode='grayscale',
    batch_size=batch_size_train,
    image_size=(img_height, img_width),
    shuffle=True,
    seed=123,
    validation_split=0.1,
    subset="training",
)

my_validation = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset_test/",
    labels='inferred',
    label_mode='int',
    class_names=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
    color_mode='grayscale',
    batch_size=batch_size_test,
    image_size=(img_height, img_width),
    shuffle=True,
    seed=123,
    validation_split=0.1,
    subset="validation",
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(img_height, img_width)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'],
              )

model.fit(my_train, epochs=10)

print("Evaluate: ")
model.evaluate(my_validation, verbose=2)

model.summary()

