import tensorflow as tf
from simba.policies.mpc_policy import MpcPolicy
from simba.infrastructure.logging_utils import logger


class RandomShootingMpc(MpcPolicy):
    def __init__(self,
                 model,
                 c_model,
                 environment,
                 horizon,
                 n_samples,
                 particles):
        super().__init__(
            model,
            c_model,
            environment,
            horizon,
            n_samples,
            particles
        )

    def generate_action(self, state):
        action = self.do_generate_action(tf.constant(state, dtype=tf.float32))
        return action.numpy()

    @tf.function
    def do_generate_action(self, state):
        lb, ub, mu, sigma = self.sampling_params
        action_sequences = tf.random.uniform((self.n_samples, self.horizon, self.action_space.shape[0]), lb, ub)
        action_sequences_batches = tf.tile(action_sequences, (self.particles, 1, 1))
        trajectories = self.model.unfold_sequences(
            tf.broadcast_to(state, (action_sequences_batches.shape[0], state.shape[0])), action_sequences_batches
        )
        scores = self.compute_objective(trajectories, action_sequences_batches)
        #scores = self.objective(cumulative_rewards)
        best_trajectory_id = tf.argmax(scores)
        return action_sequences[best_trajectory_id, 0, :]


