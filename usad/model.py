import time

import numpy as np

from usad.utils import get_data, merge_data_to_csv
from usad.data import SlidingWindowDataset, SlidingWindowDataLoader

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers, Model

class Encoder(Model):
    def __init__(self, input_dims: int, z_dims: int, nn_size = None):
        super().__init__()
        if not nn_size:
            nn_size = (input_dims // 2, input_dims // 4)

        self.model = Sequential()
        for cur_size in nn_size:
            self.model.add(layers.Dense(cur_size, activation="relu"))

        self.model.add(layers.Dense(z_dims, activation="relu"))
        self._set_inputs(tf.TensorSpec([None, input_dims], tf.float32, name='inputs'))

    def call(self, x):
        z = self.model(x)
        return z

class Decoder(Model):
    def __init__(self, input_dims: int, z_dims: int, nn_size = None):
        super().__init__()

        if not nn_size:
            nn_size = (input_dims // 4, input_dims // 2)

        self.model =  Sequential()
        for cur_size in nn_size:
            self.model.add(layers.Dense(cur_size, activation="relu"))
        
        self.model.add(layers.Dense(input_dims, activation="sigmoid"))

    def call(self, z):
        w = self.model(z)
        return w
    
class USAD():
    def __init__(self, x_dims: int, max_epochs: int = 250, batch_size: int = 128,
                 encoder_nn_size = None, decoder_nn_size = None,
                 z_dims: int = 38, window_size: int = 10, valid_step_frep: int = 200):

        self._x_dims = x_dims
        self._max_epochs = max_epochs
        self._batch_size = batch_size
        self._encoder_nn_size = encoder_nn_size
        self._decoder_nn_size = decoder_nn_size
        self._z_dims = z_dims
        self._window_size = window_size
        self._input_dims = x_dims * window_size
        self._valid_step_freq = valid_step_frep
        self._step = 0

        self._shared_encoder = Encoder(input_dims=self._input_dims, z_dims=self._z_dims)
        self._decoder_G = Decoder(z_dims=self._z_dims, input_dims=self._input_dims)
        self._decoder_D = Decoder(z_dims=self._z_dims, input_dims=self._input_dims)

    def fit(self, values, valid_portion=0.2):
        n = int(len(values) * valid_portion)
        if n == 0:
            train_values, valid_values = values[:], values[-1:]
        else:
            train_values, valid_values = values[:-n], values[-n:]

        train_sliding_window = SlidingWindowDataLoader(
            SlidingWindowDataset(train_values, self._window_size)._strided_values,
            batch_size=self._batch_size,
            shuffle=True,
            drop_last=False
        )

        valid_sliding_window =  SlidingWindowDataLoader(
            SlidingWindowDataset(valid_values, self._window_size)._strided_values,
            batch_size=self._batch_size
        )

        train_loss1 = []
        train_loss2 = []

        optimizer_1 = tf.keras.optimizers.RMSprop(learning_rate=0.001)
        optimizer_2 = tf.keras.optimizers.RMSprop(learning_rate=0.001)


        train_time = 0
        valid_time = 0
       
        for epoch in range(1, self._max_epochs + 1):
            
            train_start = time.time()
            for step in range(train_sliding_window._total):

                with tf.GradientTape() as tape1:
                    x_batch_train = train_sliding_window.get_item(step)
                    
                    w = tf.reshape(x_batch_train,(-1, self._input_dims))

                    z = self._shared_encoder(w)
                    w_G = self._decoder_G(z)
                    w_D = self._decoder_D(z)
                    w_G_D = self._decoder_D(self._shared_encoder(w_G))
                    
                    loss1 = (1 / epoch) * tf.reduce_mean((w - w_G) ** 2) + (1 - 1 / epoch) * tf.reduce_mean((w - w_G_D) ** 2)

                with tf.GradientTape() as tape2:
                    
                    z = self._shared_encoder(w)
                    w_G = self._decoder_G(z)
                    w_D = self._decoder_D(z)
                    w3 = self._decoder_D(self._shared_encoder(w_G))

                    loss2 = (1 / epoch) * tf.reduce_mean((w - w_D) ** 2) - (1 - 1 / epoch) * tf.reduce_mean((w - w_G_D) ** 2)

                grad_ae1 = tape1.gradient(loss1, self._shared_encoder.trainable_variables + self._decoder_G.trainable_variables)
                grad_ae2 = tape2.gradient(loss2, self._shared_encoder.trainable_variables + self._decoder_D.trainable_variables)
                
                optimizer_1.apply_gradients(zip(grad_ae1, self._shared_encoder.trainable_variables + self._decoder_G.trainable_variables))
                optimizer_2.apply_gradients(zip(grad_ae2, self._shared_encoder.trainable_variables + self._decoder_D.trainable_variables))
            
            train_time += time.time() - train_start

            val_losses1 = []
            val_losses2 = []
            valid_start = time.time()

            for step in range(valid_sliding_window._total):
                x_batch_val = valid_sliding_window.get_item(step)

                w = tf.reshape(x_batch_val,(-1, self._input_dims))
                z = self._shared_encoder(w)
                w_G = self._decoder_G(z)
                w_D = self._decoder_D(z)
                w_G_D = self._decoder_D(self._shared_encoder(w_G))

                val_loss1 = 1 / epoch * tf.reduce_mean((w - w_G) ** 2) + (1 - 1 / epoch) * tf.reduce_mean(
                    (w - w_G_D) ** 2)
                val_loss2 = 1 / epoch * tf.reduce_mean((w - w_D) ** 2) - (1 - 1 / epoch) * tf.reduce_mean(
                    (w - w_G_D) ** 2)

                val_losses1.append(val_loss1.numpy())
                val_losses2.append(val_loss2.numpy())
            
            valid_time += time.time() - valid_start
            
            val1_loss = np.mean(val_losses1)
            val2_loss = np.mean(val_losses2)

            print(f'epoch {epoch} val1_loss: {val1_loss}, val2_loss: {val2_loss}')
        print()

        return {
            'train_time': train_time,
            'valid_time': valid_time
        }

    def predict(self, values, alpha=1, beta=0, on_dim=False):
        collect_scores = []
        test_sliding_window = SlidingWindowDataLoader(
            SlidingWindowDataset(values, self._window_size)._strided_values,
            batch_size=self._batch_size
        )
        
        for step in range(test_sliding_window._total):
            w = test_sliding_window.get_item(step)
            w = tf.reshape(w, (-1, self._input_dims))

            z = self._shared_encoder(w)
            w_G = self._decoder_G(z)
            w_D = self._decoder_D(z)
            w_G_D = self._decoder_D(self._shared_encoder(w_G))

            batch_scores = alpha *((w - w_G) ** 2) + beta * ((w - w_G_D) ** 2)
            batch_scores = tf.reshape(batch_scores, (-1, self._window_size, self._x_dims))

            if not on_dim:
                batch_scores = np.sum(batch_scores, axis=2)

            if not collect_scores:
                collect_scores.extend(batch_scores[0])
                collect_scores.extend(batch_scores[1:, -1])
            else:
                collect_scores.extend(batch_scores[:, -1])

        return collect_scores

    def reconstruct(self, values):
        collect_G = []
        collect_G_D = []
        test_sliding_window = SlidingWindowDataLoader(
            SlidingWindowDataset(values, self._window_size)._strided_values,
            batch_size=self._batch_size
        )

        n = 0
        for step in range(test_sliding_window._total):
            w = test_sliding_window.get_item(step)
            w = tf.reshape(w,(-1, self._input_dims))

            z = self._shared_encoder(w)
            w_G = self._decoder_G(z)
            w_D = self._decoder_D(z)
            w_G_D = self._decoder_D(self._shared_encoder(w_G))

            batch_G = tf.reshape(w_G, (-1, self._window_size, self._x_dims))
            batch_G_D = tf.reshape(w_G_D, (-1, self._window_size, self._x_dims))
            
            if not collect_G:
                collect_G.extend(batch_G[0])
                collect_G.extend(batch_G[1:, -1])
            else:
                collect_G.extend(batch_G[:, -1])


            if not collect_G_D:
                collect_G_D.extend(batch_G_D[0])
                collect_G_D.extend(batch_G_D[1:, -1])
            else:
                collect_G_D.extend(batch_G_D[:, -1])
        
        return collect_G, collect_G_D

    def save(self, shared_encoder_path, decoder_G_path, decoder_D_path):
        self._shared_encoder.save_weights(shared_encoder_path)
        self._decoder_G.save_weights(decoder_G_path)
        self._decoder_D.save_weights(decoder_D_path)

    def restore(self, shared_encoder_path, decoder_G_path, decoder_D_path):
        self._shared_encoder.load_weights(shared_encoder_path)
        self._decoder_G.load_weights(decoder_G_path)
        self._decoder_D.load_weights(decoder_D_path)


# class USADNoShare():
#     def __init__(self, x_dims: int, max_epochs: int = 250, batch_size: int = 128,
#                  encoder_nn_size = None, decoder_nn_size = None,
#                  z_dims: int = 38, window_size: int = 10, valid_step_frep: int = 200):

#         self._x_dims = x_dims
#         self._max_epochs = max_epochs
#         self._batch_size = batch_size
#         self._encoder_nn_size = encoder_nn_size
#         self._decoder_nn_size = decoder_nn_size
#         self._z_dims = z_dims
#         self._window_size = window_size
#         self._input_dims = x_dims * window_size
#         self._valid_step_freq = valid_step_frep
#         self._step = 0

#         self._encoder_G = Encoder(input_dims=self._input_dims, z_dims=self._z_dims)
#         self._encoder_D = Encoder(input_dims=self._input_dims, z_dims=self._z_dims)
#         self._decoder_G = Decoder(z_dims=self._z_dims, input_dims=self._input_dims)
#         self._decoder_D = Decoder(z_dims=self._z_dims, input_dims=self._input_dims)


#     def fit(self, values, valid_portion=0.2):
#         n = int(len(values) * valid_portion)
#         if n == 0:
#             train_values, valid_values = values[:], values[-1:]
#         else:
#             train_values, valid_values = values[:-n], values[-n:]

#         train_sliding_window = SlidingWindowDataLoader(
#             SlidingWindowDataset(train_values, self._window_size)._strided_values,
#             batch_size=self._batch_size,
#             shuffle=True,
#             drop_last=False
#         )

#         valid_sliding_window =  SlidingWindowDataLoader(
#             SlidingWindowDataset(valid_values, self._window_size)._strided_values,
#             batch_size=self._batch_size
#         )

#         train_loss1 = []
#         train_loss2 = []

#         optimizer_1 = tf.keras.optimizers.RMSprop(learning_rate=0.001)
#         optimizer_2 = tf.keras.optimizers.RMSprop(learning_rate=0.001)
       

#         for epoch in range(1, self._max_epochs+1):
#             print(epoch)
#             for step in range(train_sliding_window._total):

#                 with tf.GradientTape() as tape3:
#                     x_batch_train = train_sliding_window.get_item(step)
                    
#                     w = tf.reshape(x_batch_train,(-1, self._input_dims))

#                     z1 = self._encoder_G(w)
#                     z2 = self._encoder_D(w)
#                     w1 = self._decoder_G(z1)
#                     w2 = self._decoder_D(z2)
#                     w3 = self._decoder_D(self._encoder_D(w1))
                    
#                     loss1 = (1 / epoch) * tf.reduce_mean((w - w1) ** 2) + (1 - 1 / epoch) * tf.reduce_mean((w - w3) ** 2)

#                 with tf.GradientTape() as tape4:
                    
#                     z1 = self._encoder_G(w)
#                     z2 = self._encoder_D(w)
#                     w1 = self._decoder_G(z1)
#                     w2 = self._decoder_D(z2)
#                     w3 = self._decoder_D(self._encoder_D(w1))
                    
#                     loss2 = (1 / epoch) * tf.reduce_mean((w - w2) ** 2) - (1 - 1 / epoch) * tf.reduce_mean((w - w3) ** 2)

#                 grad_ae1 = tape3.gradient(loss1, self._encoder_G.trainable_variables+self._decoder_G.trainable_variables)
#                 grad_ae2 = tape4.gradient(loss2, self._encoder_D.trainable_variables+self._decoder_D.trainable_variables)
                
#                 optimizer_1.apply_gradients(zip(grad_ae1, self._encoder_G.trainable_variables+self._decoder_G.trainable_variables))
#                 optimizer_2.apply_gradients(zip(grad_ae2, self._encoder_D.trainable_variables+self._decoder_D.trainable_variables))

#             val_losses1 = []
#             val_losses2 = []
#             for step in range(valid_sliding_window._total):
#                 x_batch_val = valid_sliding_window.get_item(step)

#                 w = tf.reshape(x_batch_val,(-1, self._input_dims))
#                 w_G, w_D, w_G_D  = self.model(w)

#                 val_loss1 = 1 / epoch * tf.reduce_mean((w - w_G) ** 2) + (1 - 1 / epoch) * tf.reduce_mean(
#                     (w - w_G_D) ** 2)
#                 val_loss2 = 1 / epoch * tf.reduce_mean((w - w_D) ** 2) - (1 - 1 / epoch) * tf.reduce_mean(
#                     (w - w_G_D) ** 2)

#                 val_losses1.append(val_loss1.numpy())
#                 val_losses2.append(val_loss2.numpy())

#                 print("\rValidation","."*(step%10), end="")

#             print("")
#             val1_loss = np.mean(val_losses1)
#             val2_loss = np.mean(val_losses2)

#             print("val1_loss: ", val1_loss, "val2_loss: ", val2_loss)



#     def predict(self, values, alpha=1, beta=0, on_dim=False):
#         collect_scores = []
#         test_sliding_window = SlidingWindowDataLoader(
#             SlidingWindowDataset(values, self._window_size)._strided_values,
#             batch_size=self._batch_size
#         )
        
#         for step in range(test_sliding_window._total):
#             w = test_sliding_window.get_item(step)
#             w = tf.reshape(w, (-1, self._input_dims))

#             z1 = self._encoder_G(w)
#             z2 = self._encoder_D(w)
#             w1 = self._decoder_G(z1)
#             w2 = self._decoder_D(z2)
#             w3 = self._decoder_D(self._encoder_D(w1))

#             batch_scores = alpha *((w - w1) ** 2) + beta * ((w - w3) ** 2)

#             batch_scores = tf.reshape(batch_scores, (-1, self._window_size, self._x_dims))

#             if not on_dim:
#                 batch_scores = np.sum(batch_scores, axis=2)

#             if not collect_scores:
#                 collect_scores.extend(batch_scores[0])
#                 collect_scores.extend(batch_scores[1:, -1])
#             else:
#                 collect_scores.extend(batch_scores[:, -1])

#         return collect_scores

#     def reconstruct(self, values):
#         collect_G = []
#         collect_G_D = []
#         test_sliding_window = SlidingWindowDataLoader(
#             SlidingWindowDataset(values, self._window_size)._strided_values,
#             batch_size=self._batch_size
#         )

#         n = 0
#         for step in range(test_sliding_window._total):
#             w = test_sliding_window.get_item(step)
#             w = tf.reshape(w,(-1, self._input_dims))

#             w_G = self._AE_G(w)
#             w_G_D = self._AE_D(w_G)
#             batch_G = tf.reshape(w_G, (-1, self._window_size, self._x_dims))
#             batch_G_D = tf.reshape(w_G_D, (-1, self._window_size, self._x_dims))
#             print(batch_G.shape, batch_G_D.shape)

#             n += batch_G.shape[0]
            
#             if not collect_G:
#                 collect_G.extend(batch_G[0])
#                 collect_G.extend(batch_G[1:, -1])
#             else:
#                 collect_G.extend(batch_G[:, -1])


#             if not collect_G_D:
#                 collect_G_D.extend(batch_G_D[0])
#                 collect_G_D.extend(batch_G_D[1:, -1])
#             else:
#                 collect_G_D.extend(batch_G_D[:, -1])

#         print('final', n)

        
#         return collect_G, collect_G_D


    
#     def save(self, shared_encoder_path, decoder_G_path, decoder_D_path):
#         self._encoder_G.save_weights(shared_encoder_path+'G')
#         self._encoder_D.save_weights(shared_encoder_path+'D')
#         self._decoder_G.save_weights(decoder_G_path)
#         self._decoder_D.save_weights(decoder_D_path)

#     def restore(self, shared_encoder_path, decoder_G_path, decoder_D_path):
#         self._encoder_G.load_weights(shared_encoder_path+'G')
#         self._encoder_D.load_weights(shared_encoder_path+'D')
#         self._decoder_G.load_weights(decoder_G_path)
#         self._decoder_D.load_weights(decoder_D_path)
