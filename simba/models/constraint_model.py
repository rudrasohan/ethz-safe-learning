import numpy as np
import tensorflow as tf
from simba.models.mlp_ensemble import BaseLayer, EpochLearningRateSchedule
from simba.infrastructure.logging_utils import logger

class ConstraintMlp(tf.keras.Model):
    def __init__(self,
                 inputs_dim,
                 outputs_dim,
                 n_layers,
                 units,
                 activation,
                 dropout_rate):
        super().__init__()
        self.inputs_dim = inputs_dim
        self.forward = tf.keras.Sequential(
            [tf.keras.layers.InputLayer(input_shape=(inputs_dim,))] +
            [BaseLayer(units, activation, dropout_rate) for _ in range(n_layers)]
        )
        self.head = tf.keras.layers.Dense(outputs_dim)
        #activation=lambda t: tf.math.softplus(t) + 1e-4
        self.output_dim = outputs_dim

    @tf.function(
        input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.float32),
                         tf.TensorSpec(shape=(), dtype=tf.bool)]
    )
    def call(self, inputs, training=None):
        #import pdb; pdb.set_trace()
        x = self.forward(inputs, training)
        return self.head(x)

class ConstraintModel(tf.Module):
    def __init__(self,
                 inputs_dim,
                 outputs_dim,
                 cost_function,
                 batch_size,
                 validation_split,
                 learning_rate,
                 learning_rate_schedule,
                 training_steps,
                 mlp_params,
                 c_level,
                 train_epochs):
        super().__init__()
        self.inputs_dim = inputs_dim
        self.outputs_dim = outputs_dim
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.training_steps = training_steps
        self.cost_function = cost_function
        self.mlp_params = mlp_params
        self.c_level = c_level
        self.constraint_model = ConstraintMlp(inputs_dim=self.inputs_dim, outputs_dim=self.outputs_dim, **self.mlp_params) 
        self.optimizer = tf.keras.optimizers.Adam(
            EpochLearningRateSchedule(learning_rate, training_steps, train_epochs)
            if learning_rate_schedule else learning_rate,
            clipvalue=1.0,
            epsilon=1e-5)

    def build(self):
        pass

    def forward(self, states, training=tf.constant(False)):
        g = self.constraint_model(states, training)
        return g
    
    @tf.function
    def training_step(self, prev_states, actions, states):
        loss = 0.0
        ensemble_trainable_variables = []
        with tf.GradientTape() as tape:
            #import pdb; pdb.set_trace()
            C_0 = self.cost_function(prev_states, actions)
            targets = self.cost_function(states, actions)
            g = self.forward(states, tf.constant(True))
            g = tf.reshape(g, actions.shape)
            #import pdb; pdb.set_trace()
            y_preds = tf.reduce_sum(tf.multiply(g, actions), 1)
            loss = tf.keras.losses.MSE((targets-C_0), y_preds)
            grads = tape.gradient(loss, self.constraint_model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.constraint_model.trainable_variables))
        return loss

    @tf.function
    def validation_step(self, prev_states, actions, states):
        C_0 = self.cost_function(prev_states, actions)
        targets = self.cost_function(states, actions)
        g = self.forward(states, tf.constant(False))
        g = tf.reshape(g, actions.shape)
        y_preds = tf.reduce_sum(tf.multiply(g, actions), 1)
        loss = tf.keras.losses.MSE((targets-C_0), y_preds)
        return loss

    def split_train_validate(self, obs, acts, next_obs):
        indices = np.random.permutation(obs.shape[0])
        num_val = int(obs.shape[0] * self.validation_split)
        train_idx, val_idx = indices[num_val:], indices[:num_val]
        return obs[train_idx, ...], acts[train_idx, ...], next_obs[train_idx,...],\
         obs[val_idx, ...], acts[val_idx, ...], next_obs[val_idx, ...]

    def fit(self, obs, acts, next_obs):
        assert obs.shape[0] == acts.shape[0] == next_obs.shape[0], "Obs batch size ({})" 
        "Acts batch size ({}) Next Obs batch size ({})".format(obs.shape[0], acts.shape[0], next_obs.shape[0])
        assert np.isfinite(obs).all() and np.isfinite(acts).all() and np.isfinite(next_obs).all(), \
            "Training data is not finite."
        losses = np.empty((self.training_steps,))
        train_o, train_a, train_no, validate_o, validate_a, validate_no = self.split_train_validate(obs, acts, next_obs)
        n_batches = int(np.ceil(train_o.shape[0] / self.batch_size))
        step = 0
        while step < self.training_steps:
            #import pdb; pdb.set_trace()
            o_batches = np.array_split(train_o, n_batches, axis=0)
            a_batches = np.array_split(train_a, n_batches, axis=0)
            no_batches = np.array_split(train_no, n_batches, axis=0)
            for o_batch, a_batch, no_batch in zip(o_batches, a_batches, no_batches):
                loss = self.training_step(tf.constant(o_batch),
                                          tf.constant(a_batch),
                                          tf.constant(no_batch))
                losses[step] = loss
                step += 1
                if step % int(self.training_steps / 10) == 0:
                    validation_loss = self.validation_step(validate_o, validate_a, validate_no).numpy()
                    logger.debug(
                        "Step {} | Const Training Loss {} | Const Validation Loss {}".format(step, loss, validation_loss))
                if step == self.training_steps:
                    break
        return losses

    @tf.function
    def __call__(self, C_0, states, actions, *args, **kwargs):
        #import pdb; pdb.set_trace()
        if len(states.shape) < 2:
            states = tf.cast(tf.expand_dims(states, 0), dtype=tf.float32)
            actions = tf.cast(tf.expand_dims(actions, 0), dtype=tf.float32)
        # else:
        #     import pdb; pdb.set_trace()
        g = self.constraint_model(states, training=tf.constant(False))
        g = tf.reshape(g, actions.shape)
        gg = tf.reduce_sum(tf.multiply(g, g), 1)
        ga = tf.reduce_sum(tf.multiply(g, actions), 1)
        div = tf.divide((ga + C_0 - self.c_level), gg)
        lamda = tf.maximum(div, 0.0)
        lb = tf.reshape(lamda, (-1, 1))
        lc = tf.concat([lb, lb], 1)
        return tf.squeeze((actions - lc * g)) 
