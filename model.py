import tensorflow as tf
import numpy as np

def convert_img(img_path) -> np.ndarray:
    img = tf.keras.preprocessing.image.load_img(img_path,
                                                grayscale=True,
                                                color_mode="grayscale")
    output = tf.keras.preprocessing.image.img_to_array(img)
    output = tf.image.resize(output, size=(28,28))
    return output

def prepare_for_prediction(img_path):
    img = convert_img(img_path)
    img = tf.convert_to_tensor(img)

    img = tf.squeeze(img)

    return img

model = tf.keras.models.load_model('model.keras')

img = prepare_for_prediction("examples/Example_3.png")
input_value = tf.expand_dims(img, axis=0)

predict_value = model.predict(input_value)
print(predict_value)