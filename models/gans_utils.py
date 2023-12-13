import sys
import tensorflow as tf

sys.path.append("../")
from src.constants import constants
import numpy as np


def get_path_cgan(learning_rate=0.0005, epoch=250):
    path = f"{constants.outputs.models.cgan_28}/lr={learning_rate}/checkpoints/c_gan{epoch - 1}.h5"
    return path


def get_path_wcgan(learning_rate=0.00005, n_critic=5, clip_value=0.01, epoch=250):
    path = f"{constants.outputs.models.wasserstein_cgan_28}/lr={learning_rate}/n_critic={n_critic}/clip_value={clip_value}/checkpoints/c_gan{epoch - 1}.h5"
    return path


def get_path_wcgan_gp(learning_rate=0.0001, n_critic=5, gradient_penalty=10.0, epoch=250):
    path = f"{constants.outputs.models.wasserstein_cgan_gp_28}/lr={learning_rate}/n_critic={n_critic}/gradient_penalty={gradient_penalty}/checkpoints/c_gan{epoch - 1}.h5"
    return path


def incorpore_syntetic_data_to_real_data(real_data, syntetic_data):
    """
    Función que incorpora los datos sintéticos a los datos reales

    Parameters
    ----------
    real_data: np.array
        Los datos reales
    syntetic_data: np.array
        Los datos sintéticos
    samples_generated_per_ap: int
        El número de muestras sintéticas generadas por cada AP

    Returns
    -------
    rpmap_ext: np.array
        Los datos reales y sintéticos concatenados
    """
    if len(real_data.shape) == 4:
        real_data = real_data[:, :, :, 0]

    samples_per_ap = int(real_data.shape[0] / len(constants.aps))  # 223 muestras por 7 APs
    samples_generated_per_ap = int(syntetic_data.shape[0] / len(constants.aps))  # N muestras generated por 7 APs
    rpmap_ext = np.zeros((real_data.shape[0] + samples_generated_per_ap * len(constants.aps),
                          real_data.shape[1], real_data.shape[2]))  # Inicializamos el array de salida
    count_gen = 0  # Contador para ir añadiendo los datos sintéticos
    for n_ap, ap in enumerate(constants.aps):  # Iteramos por cada AP
        for batch_temporal in range(samples_per_ap):  # Iteramos por cada batch temporal
            id_row = n_ap * samples_per_ap + batch_temporal  # Obtenemos el id de la fila
            rpmap_ext[count_gen, :, :] = real_data[id_row, :, :]  # Añadimos los datos reales
            count_gen += 1  # Aumentamos el contador
        for id_gen in range(samples_generated_per_ap):  # Iteramos por cada muestra sintética
            rpmap_ext[count_gen, :, :] = syntetic_data[
                id_gen + samples_generated_per_ap * n_ap]  # Añadimos los datos sintéticos
            count_gen += 1  # Aumentamos el contador

    return rpmap_ext


class cGAN(tf.keras.models.Model):
    def __init__(self, generator, discriminator, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.generator = generator
        self.discriminator = discriminator

    def compile(self, opt_g, opt_d, loss_g, loss_d, *args, **kwargs):
        super().compile(*args, **kwargs)

        self.opt_g = opt_g
        self.opt_d = opt_d
        self.loss_g = loss_g
        self.loss_d = loss_d

    @tf.function
    def train_step(self, batch):
        batch_size = tf.shape(batch[0])[0]
        latent_dim = self.generator.input_shape[0][1]

        images_real, labels = batch

        labels = tf.expand_dims(labels, axis=-1)

        # Generate images
        latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
        images_generated = self.generator([latent_vectors, labels], training=False)

        # Train the discriminator
        with tf.GradientTape() as d_tape:
            # Pass the real and fake images to the discriminator model
            yhat_real = self.discriminator([images_real, labels], training=True)
            yhat_fake = self.discriminator([images_generated, labels], training=True)
            yhat_realfake = tf.concat([yhat_real, yhat_fake], axis=0)

            # Create labels for real and fakes images
            y_realfake = tf.concat([tf.zeros_like(yhat_real), tf.ones_like(yhat_fake)], axis=0)

            # Add some noise to the TRUE outputs (crucial step)
            noise_real = 0.15 * tf.random.uniform(tf.shape(yhat_real))
            noise_fake = -0.15 * tf.random.uniform(tf.shape(yhat_fake))
            y_realfake += tf.concat([noise_real, noise_fake], axis=0)

            # Calculate loss
            total_loss_d = self.loss_d(y_realfake, yhat_realfake)

        # Apply backpropagation
        dgrad = d_tape.gradient(total_loss_d, self.discriminator.trainable_variables)
        self.opt_d.apply_gradients(zip(dgrad, self.discriminator.trainable_variables))

        # Train the generator
        with tf.GradientTape() as g_tape:
            # Generate images
            latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
            images_generated = self.generator([latent_vectors, labels], training=True)

            # Create the predicted labels
            predicted_labels = self.discriminator([images_generated, labels], training=False)

            # Calculate loss - trick to training to fake out the discriminator
            total_loss_g = self.loss_g(tf.zeros_like(predicted_labels), predicted_labels)

        # Apply backpropagation
        ggrad = g_tape.gradient(total_loss_g, self.generator.trainable_variables)
        self.opt_g.apply_gradients(zip(ggrad, self.generator.trainable_variables))

        return {"loss_d": total_loss_d, "loss_g": total_loss_g}


class DataAugmentation:
    def __init__(self, path_to_generator):
        self.generator = self.get_model(path_to_generator)

    @staticmethod
    def get_model(path_to_generator):
        return tf.keras.models.load_model(path_to_generator)

    @staticmethod
    def reescale_output(output):
        # Hay casos en los que solo se predice un número constante (el algoritmo falla por completo)
        if np.std(output) == 0 and np.mean(output) == -1:
            return np.zeros(output.shape)

        if np.std(output) == 0 and np.mean(output) == 1:
            return np.ones(output.shape)

        maximum, minimum = np.max(output), np.min(output)
        return (output - minimum) / (maximum - minimum)

    def __call__(self, n_samples_per_label: int = 223):
        n_classes = len(constants.aps)
        noise_input = np.random.normal(0, 1, size=[n_samples_per_label * n_classes, 100])
        input_labels = np.repeat(np.arange(0, n_classes), n_samples_per_label).reshape(-1, 1)
        generated = self.generator.predict([noise_input, input_labels])[:, :, :, 0]
        generated = self.reescale_output(generated)
        return generated, input_labels
