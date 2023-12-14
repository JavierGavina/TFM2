import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv2DTranspose, Reshape, BatchNormalization, LeakyReLU, \
    Dropout, Input, Concatenate, Embedding
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import History, ReduceLROnPlateau

from PIL import Image

import matplotlib.pyplot as plt
import os
import sys

sys.path.append("..")

from src import dataloader
from src.constants import constants
from src.dataloader import labelEncoding, labelDecoding
from models.gans_utils import cGAN
from models.callbacks import LoggingCheckpointTraining, SaveImageTraining


def define_discriminator(input_shape=(28, 28, 1), n_classes=7):
    input_label = Input(shape=(1,), name="input_label")
    lab = Embedding(n_classes, 50)(input_label)
    lab = Dense(input_shape[0] * input_shape[1] * 1)(lab)
    lab = Reshape((input_shape[0], input_shape[1], 1))(lab)

    input_image = Input(shape=input_shape, name="input_image")
    combined = Concatenate(name="concatenate")([input_image, lab])

    x = Conv2D(64, (3, 3), strides=(2, 2), padding="same")(combined)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(128, (3, 3), strides=(2, 2), padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(256, (3, 3), strides=(2, 2), padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Flatten()(x)
    out = Dense(1, activation="sigmoid", name="out_layer")(x)

    model = Model([input_image, input_label], out, name="discriminator")
    return model


def define_generator(latent_dim=100, n_classes=7):
    input_label = Input(shape=(1,), name="input_label")
    lab = Embedding(n_classes, 50)(input_label)
    lab = Dense(7 * 7 * 128)(lab)
    lab = Reshape((7, 7, 128))(lab)

    input_latent = Input(shape=(latent_dim,), name="input_noise")
    x = Dense(7 * 7 * 128)(input_latent)
    x = LeakyReLU(alpha=0.2)(x)
    x = Reshape((7, 7, 128))(x)

    combined = Concatenate(name="concatenate")([x, lab])

    x = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same")(combined)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2DTranspose(64, (7, 7), strides=(1, 1), padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)

    out = Conv2D(1, (1, 1), activation="tanh", padding="same")(x)

    model = Model([input_latent, input_label], out, name="generator")
    return model

# def define_discriminator(input_shape=(28, 28, 1), n_classes=7):
#     input_label = Input(shape=(1,), name="input_label")
#     lab = Embedding(n_classes, 50)(input_label)
#     lab = Dense(input_shape[0] * input_shape[1] * 1)(lab)
#     input_image = Input(shape=input_shape, name="input_image")
#     flatten_image = Flatten()(input_image)
#     combined = Concatenate(name="concatenate")([flatten_image, lab])
#
#     x = Dense(512)(combined)
#     x = LeakyReLU(alpha=0.2)(x)
#
#     x = Dense(256)(x)
#     x = LeakyReLU(alpha=0.2)(x)
#
#     out = Dense(1, activation="sigmoid", name="out_layer")(x)
#     model = Model([input_image, input_label], out, name="discriminator")
#
#     return model
#
#
# def define_generator(latent_dim=100, n_classes=7):
#     input_label = Input(shape=(1,), name="input_label")
#     lab = Embedding(n_classes, 50)(input_label)
#     lab = Dense(4 * 4 * 64)(lab)
#
#     input_latent = Input(shape=(latent_dim,), name="input_noise")
#     x = Dense(4 * 4 * 64)(input_latent)
#     x = LeakyReLU(alpha=0.2)(x)
#
#     combined = Concatenate(name="concatenate")([x, lab])
#
#     x = Dense(7 * 7 * 32)(combined)
#     x = LeakyReLU(alpha=0.2)(x)
#
#     x = Dense(14 * 14 * 16)(x)
#     x = LeakyReLU(alpha=0.2)(x)
#
#     x = Dense(28 * 28 * 1, activation="tanh")(x)
#     out = Reshape((28, 28, 1))(x)
#
#     model = Model([input_latent, input_label], out, name="generator")
#     return model


if __name__ == "__main__":
    get_data = dataloader.DataLoader(data_dir=f"../{constants.data.train.FINAL_PATH}/groundtruth.csv",
                                     aps_list=constants.aps, batch_size=30, step_size=5,
                                     size_reference_point_map=28,
                                     return_axis_coords=False)
    # adaptamos los datos para el entrenamiento
    X, y, _ = get_data()
    y_encoded = labelEncoding(y)
    minimo, maximo = np.min(X), np.max(X)
    X_reescalado = 2 * (X - minimo) / (maximo - minimo) - 1  # Reescalado entre -1 y 1

    # Obtenemos el modelo
    discriminator = define_discriminator()
    generator = define_generator(latent_dim=100)
    path_arch = f"../{constants.outputs.architectures.arquitectures}"
    path_cgan_arch = f"../{constants.outputs.architectures.cgan_28}"
    # Guardamos las arquitecturas
    os.makedirs(path_arch, exist_ok=True)
    os.makedirs(path_cgan_arch, exist_ok=True)

    tf.keras.utils.plot_model(discriminator, show_shapes=True, show_layer_activations=True,
                              to_file=f"{path_cgan_arch}/conditional_discriminator_28x28.png", dpi=120)
    tf.keras.utils.plot_model(discriminator, show_shapes=True, show_layer_activations=True,
                              to_file=f"{path_cgan_arch}/conditional_generator_28x28.png", dpi=120)

    path_cgan_28 = f"../{constants.outputs.models.cgan_28}"
    os.makedirs(path_cgan_28, exist_ok=True)
    learning_rates = [0.00005, 0.0001, 0.0005, 0.001, 0.005]
    for lr in learning_rates:
        path = f"{path_cgan_28}/lr={lr}"
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            path_cgan_28_checkpoints = f"{path}/checkpoints"
            path_cgan_28_images = f"{path}/images"
            path_cgan_28_learning_curves = f"{path}/learning_curves"
            path_cgan_28_gif = f"{path}/gif"

            os.makedirs(path_cgan_28_checkpoints, exist_ok=True)
            os.makedirs(path_cgan_28_images, exist_ok=True)
            os.makedirs(path_cgan_28_learning_curves, exist_ok=True)
            os.makedirs(path_cgan_28_gif, exist_ok=True)

            # define the training dataset
            dataset = tf.data.Dataset.from_tensor_slices((X_reescalado, y_encoded)).shuffle(1000).batch(64)

            # define callbacks
            save_image = SaveImageTraining(X_reescalado, y_encoded, save_dir=path_cgan_28_images)
            save_model = LoggingCheckpointTraining(save_dir=path_cgan_28_checkpoints)
            hist = History()

            callbacks = [
                save_image,
                save_model,
                hist
            ]

            #  modelo cGAN
            discriminator = define_discriminator(input_shape=(28, 28, 1), n_classes=7)
            generator = define_generator(latent_dim=100)
            cgan = cGAN(generator, discriminator)

            #  compile model
            cgan.compile(
                Adam(learning_rate=lr, beta_1=0.5),
                Adam(learning_rate=lr, beta_1=0.5),
                tf.keras.losses.BinaryCrossentropy(),
                tf.keras.losses.BinaryCrossentropy()
            )

            # train model
            cgan.fit(dataset, epochs=250, callbacks=callbacks)

            # Curvas de aprendizaje
            plt.plot(hist.history["loss_d"], label="loss_d")
            plt.plot(hist.history["loss_g"], label="loss_g")
            plt.title("Curvas de aprendizaje")
            plt.legend()
            plt.savefig(f"{path_cgan_28_learning_curves}/learning_curves.png")
            plt.close()

            os.makedirs(path_cgan_28_gif, exist_ok=True)
            # Lista de nombres de archivo de imágenes PNG en la carpeta
            png_files = [f for f in os.listdir(path_cgan_28_images) if f.endswith('.png')]
            # Ordenar los nombres de archivo en orden alfabético
            png_files.sort()
            # Lista de objetos Image para cada imagen PNG
            image_list = [Image.open(os.path.join(path_cgan_28_images, f)) for f in png_files]
            # Guardar las imágenes como un archivo GIF animado
            image_list[0].save(f"{path_cgan_28_gif}/training_process.gif", save_all=True, append_images=image_list[1:],
                               duration=350,
                               loop=0)
