import tensorflow as tf
import numpy as np
from simba.infrastructure.common import standardize_name
from simba.models import BaseModel, MlpEnsemble


class TransitionModel(BaseModel):
    def __init__(self,
                 model,
                 c_model,
                 observation_space,
                 action_space,
                 scale_features,
                 sampling_propagation,
                 **kwargs):
        super().__init__(observation_space.shape[0] + action_space.shape[0],
                         observation_space.shape[0])
        self.model_scope = model
        self.model = eval(standardize_name(model))(
            inputs_dim=self.inputs_dim,
            outputs_dim=self.outputs_dim,
            **kwargs)
        self.c_model = c_model
        self.observation_space = observation_space
        self.action_space = action_space
        self.scale_features = scale_features
        self.sampling_propagation = sampling_propagation
        self.observation_space_dim = observation_space.shape[0]
        self.action_space_dim = action_space.shape[0]
        self.inputs_min = tf.concat([observation_space.low, action_space.low], axis=0)
        self.inputs_max = tf.concat([observation_space.high, action_space.high], axis=0)

    def build(self):
        self.model.build()

    def fit(self, inputs, targets):
        self._fit_statistics(inputs)
        observations = inputs[:, :self.observation_space_dim]
        next_observations = targets
        return self.model.fit(
            self.scale(tf.constant(inputs, dtype=tf.float32)).numpy(),
            (next_observations - observations).astype(np.float32))

    def _fit_statistics(self, inputs):
        if not self.scale_features:
            return
        high = np.concatenate([self.observation_space.high, self.action_space.high])
        low = np.concatenate([self.observation_space.low, self.action_space.low])
        self.inputs_min = tf.constant(
            np.where(np.isfinite(low), low, inputs.min(axis=0)), dtype=tf.float32)
        self.inputs_max = tf.constant(
            np.where(np.isfinite(high), high, inputs.max(axis=0)), dtype=tf.float32)

    def predict(self, inputs):
        return self.simulate_trajectories(
            current_state=inputs[..., :self.observation_space_dim],
            action_sequences=np.expand_dims(inputs[..., -self.action_space_dim:], axis=1)
        )

    def simulate_trajectories(self, current_state, action_sequences):
        sequences = self.unfold_sequences(
            tf.convert_to_tensor(current_state, dtype=tf.float32),
            tf.convert_to_tensor(action_sequences, dtype=tf.float32)
        )
        import pdb; pdb.set_trace()
        return sequences.numpy()

    @tf.function
    def unfold_sequences(self, s_0, action_sequences):
        horizon = action_sequences.shape[1]
        trajectories = tf.TensorArray(tf.float32, size=horizon + 1)
        s_t = s_0
        for t in tf.range(horizon):
            trajectories = trajectories.write(t, s_t)
            a_t = action_sequences[:, t, ...]
            if self.c_model is not None:
                #print("CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC")
                c_0 = self.c_model.cost_function(s_t, None)
                #print("############################")
                a_t = self.c_model(c_0, s_t, a_t)
            s_t_a_t_scaled = self.scale(tf.concat([s_t, a_t], axis=1))
            # The model predicts s_t_1 - s_t hence we add here the previous state.
            mus, sigmas, d_s_t = self.model(s_t_a_t_scaled)
            s_t += d_s_t if self.sampling_propagation else mus
        trajectories = trajectories.write(horizon, s_t)
        return tf.transpose(trajectories.stack(), [1, 0, 2])

    @tf.function(
        input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)]
    )
    def scale(self, inputs):
        if not self.scale_features:
            return inputs
        delta = self.inputs_max - self.inputs_min
        delta = tf.where(tf.less(delta, 1e-5), 1.01, delta)
        return (inputs - self.inputs_min) / delta

    def save(self):
        pass

    def load(self):
        pass
