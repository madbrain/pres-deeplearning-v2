import tensorflow as tf
from keras_flops import get_flops

# Model 1: 784 -> 10                     => 92% (  30k params)
# Model 2: 784 -> 625 + sigmoid -> 10    => 97% (1.90M params)
# Model 3: 784 -> (625 + relu) * 2 -> 10 => 98% (3.39M params)
# Model 4: 784 -> (conv2d + pooling) * 3 -> 625 + relu -> 10 => 99% (3.13M params)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  #tf.keras.layers.Dense(625, activation='sigmoid'),
  #tf.keras.layers.Dense(625, activation='relu'),
  #tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

# model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(28, 28, 1)),
#     tf.keras.layers.MaxPool2D(),
#     tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
#     tf.keras.layers.MaxPool2D(),
#     tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu'),
#     tf.keras.layers.MaxPool2D(),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(625, activation='relu'),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(10, activation='softmax')
# ])

print (get_flops(model, batch_size=1))
sys.exit(1)

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
model.summary()

import tensorflow_datasets as tfds

(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

model.fit(
    ds_train,
    epochs=6,
    validation_data=ds_test)
