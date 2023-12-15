import tensorflow as tf
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import itertools

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv2DTranspose, Reshape, LeakyReLU, \
    Input, Concatenate, Embedding, ReLU
from tensorflow.keras.models import Model

from PIL import Image

import sys

sys.path.append("../..")

from src.utils.dataloader import DataLoader, labelEncoding
from src.utils.constants import constants
from src.models.callbacks import SaveImageTraining, LoggingCheckpointTraining


# Generador usa ReLu y no LeakyReLu
def build_generator(latent_dim=100, n_classes=7):
    input_label = Input(shape=(1,), name="input_label")
    lab = Embedding(n_classes, 50)(input_label)
    lab = Dense(7 * 7 * 128)(lab)
    lab = Reshape((7, 7, 128))(lab)

    input_latent = Input(shape=(latent_dim,), name="input_noise")
    x = Dense(7 * 7 * 128)(input_latent)
    x = ReLU()(x)
    x = Reshape((7, 7, 128))(x)

    combined = Concatenate(name="concatenate")([x, lab])

    x = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same")(combined)
    x = ReLU()(x)

    x = Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same")(x)
    x = ReLU()(x)

    x = Conv2DTranspose(64, (7, 7), strides=(1, 1), padding="same")(x)
    x = ReLU()(x)

    out = Conv2D(1, (1, 1), activation="tanh", padding="same")(x)

    model = Model([input_latent, input_label], out, name="generator")
    return model


# El crítico, si que usa LeakyReLu
# Sin activación en la última capa
def build_critic(input_shape=(28, 28, 1), n_classes=7):
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
    out = Dense(1, name="out_layer")(x)

    model = Model([input_image, input_label], out, name="discriminator")
    return model


class wCGAN(Model):
    def __init__(self, generator: tf.keras.models.Model, critic: tf.keras.models.Model,
                 clip_value: float = None, n_critic: int = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.generator = generator
        self.critic = critic
        self.clip_value = clip_value
        self.n_critic = n_critic
        self.opt_g = None
        self.opt_c = None

        if clip_value is None:
            self.clip_value = math.Inf

    def compile(self, opt_g, opt_c, *args, **kwargs):
        super().compile(*args, **kwargs)

        self.opt_g = opt_g
        self.opt_c = opt_c

    @tf.function
    def train_step(self, batch):
        batch_size = tf.shape(batch[0])[0]
        latent_dim = self.generator.input_shape[0][1]

        images_real, labels = batch
        labels = tf.expand_dims(labels, axis=-1)

        # # Generate images
        latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
        images_generated = self.generator([latent_vectors, labels], training=False)

        # Train the critic (n_critic times)
        for _ in range(self.n_critic):
            with tf.GradientTape() as c_tape:
                # Pass the real and fake images to the discriminator model
                yhat_real = self.critic([images_real, labels], training=True)
                yhat_fake = self.critic([images_generated, labels], training=True)

                # Add some noise to the TRUE outputs (crucial step)
                # noise_real = 0.15 * tf.random.uniform(tf.shape(yhat_real))
                # noise_fake = -0.15 * tf.random.uniform(tf.shape(yhat_fake))

                # Calculate loss
                total_loss_c = tf.reduce_mean(yhat_fake) - tf.reduce_mean(yhat_real)

                for layer in self.critic.trainable_variables:
                    layer.assign(tf.clip_by_value(layer, -self.clip_value, self.clip_value))

            # Apply backpropagation
            cgrad = c_tape.gradient(total_loss_c, self.critic.trainable_variables)
            self.opt_c.apply_gradients(zip(cgrad, self.critic.trainable_variables))

        # Train the generator
        with tf.GradientTape() as g_tape:
            # Generate images
            latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
            images_generated = self.generator([latent_vectors, labels], training=True)

            # Create the predicted labels
            predicted_labels = self.critic([images_generated, labels], training=False)

            # Calculate loss - trick to training to fake out the discriminator
            total_loss_g = -tf.reduce_mean(predicted_labels)

        # Apply backpropagation
        ggrad = g_tape.gradient(total_loss_g, self.generator.trainable_variables)
        self.opt_g.apply_gradients(zip(ggrad, self.generator.trainable_variables))

        return {"loss_c": total_loss_c, "loss_g": total_loss_g}


if __name__ == "__main__":
    model_architecture = "../outputs/model_architecture/wasserstein_cgan_28x28"
    path_wcgan = f"../{constants.outputs.models.wasserstein_cgan_28}"

    os.makedirs(path_wcgan, exist_ok=True)

    get_data = DataLoader(data_dir=f"../{constants.data.train.FINAL_PATH}/groundtruth.csv",
                          aps_list=constants.aps, batch_size=30, step_size=5,
                          size_reference_point_map=28, return_axis_coords=False)

    # adaptamos los datos para el entrenamiento
    X, y, _ = get_data()
    y_encoded = labelEncoding(y)
    minimo, maximo = np.min(X), np.max(X)
    X_reescalado = 2 * (X - minimo) / (maximo - minimo) - 1
    """
    Los parámetros en el estado del arte son:
        - learning_rate = 0.00005
        - n_critic = 5
        - clip_value = 0.01
        - batch_size = 64
        - optimizer = RMSprop(learning_rate=0.00005)
    """
    learning_rates = [0.00005, 0.0001, 0.0005, 0.001, 0.005]
    n_critics = [1, 2, 3, 4, 5]
    clip_values = [0.001, 0.005, 0.01, 0.05, 0.1]
    latent_dim = 100

    combinaciones = np.array(list(itertools.product(learning_rates, n_critics, clip_values)))
    np.random.shuffle(combinaciones)

    for i, combinacion in enumerate(combinaciones):
        lr, n_critic, clip_value = combinacion
        lr = float(lr)
        n_critic = int(n_critic)
        clip_value = float(clip_value)

        path_out = f"{path_wcgan}/lr={lr}/n_critic={n_critic}/clip_value={clip_value}"
        if not os.path.exists(path_out):
            os.makedirs(f"{path_wcgan}/lr={lr}", exist_ok=True)
            os.makedirs(f"{path_wcgan}/lr={lr}/n_critic={n_critic}", exist_ok=True)
            os.makedirs(path_out, exist_ok=True)

            path_out_wcgan_images = f"{path_out}/images"
            path_out_wcgan_checkpoints = f"{path_out}/checkpoints"
            path_out_wcgan_learning_curves = f"{path_out}/learning_curves"
            path_out_wcgan_gif = f"{path_out}/gif"

            os.makedirs(path_out_wcgan_images, exist_ok=True)
            os.makedirs(path_out_wcgan_checkpoints, exist_ok=True)
            os.makedirs(path_out_wcgan_learning_curves, exist_ok=True)
            os.makedirs(path_out_wcgan_gif, exist_ok=True)

            print(f"Combinación {i+1}/{len(combinaciones)}")
            print("="*90)
            print(f"lr={lr}, n_critic={n_critic}, clip_value={clip_value}")
            print("="*90)

            # constantes e hiperparámetros
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
            generator = build_generator(latent_dim=latent_dim, n_classes=7)
            critic = build_critic(input_shape=(28, 28, 1), n_classes=7)

            wgan = wCGAN(generator=generator, critic=critic, clip_value=clip_value, n_critic=n_critic)
            wgan.compile(opt_g=optimizer, opt_c=optimizer)

            # define the training dataset
            dataset = tf.data.Dataset.from_tensor_slices((X_reescalado, y_encoded)).shuffle(1000).batch(64)

            # define callbacks
            save_image = SaveImageTraining(X_reescalado, y_encoded, save_dir=path_out_wcgan_images)
            save_model = LoggingCheckpointTraining(save_dir=path_out_wcgan_checkpoints)
            hist = tf.keras.callbacks.History()

            callbacks = [
                save_image,
                save_model,
                hist
            ]

            # train the model
            wgan.fit(dataset, epochs=250, callbacks=callbacks)

            # Curvas de aprendizaje
            plt.plot(hist.history["loss_c"], label="loss_c")
            plt.plot(hist.history["loss_g"], label="loss_g")
            plt.title("Curvas de aprendizaje")
            plt.legend()
            plt.savefig(f"{path_out_wcgan_learning_curves}/learning_curves.png")
            plt.close()

            # save gif of generated images

            # Lista de nombres de archivo de imágenes PNG en la carpeta
            png_files = [f for f in os.listdir(path_out_wcgan_images) if f.endswith('.png')]
            # Ordenar los nombres de archivo en orden alfabético
            png_files.sort()
            # Lista de objetos Image para cada imagen PNG
            image_list = [Image.open(os.path.join(path_out_wcgan_images, f)) for f in png_files]
            # Guardar las imágenes como un archivo GIF animado
            image_list[0].save(f"{path_out_wcgan_gif}/training_process.gif", save_all=True, append_images=image_list[1:], duration=350,
                               loop=0)