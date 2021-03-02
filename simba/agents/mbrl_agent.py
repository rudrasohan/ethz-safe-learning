import numpy as np
from simba.infrastructure.common import standardize_name
from simba.infrastructure.logging_utils import logger
from simba.agents import BaseAgent
from simba.policies import CemMpc, SafeCemMpc, RandomShootingMpc, RandomMpc
from simba.models.transition_model import TransitionModel
from simba.models.constraint_model import ConstraintModel


class MbrlAgent(BaseAgent):
    def __init__(self,
                 environment,
                 warmup_timesteps,
                 train_batch_size,
                 train_interaction_steps,
                 episode_length,
                 replay_buffer_size,
                 **kwargs
                 ):
        super().__init__(
            replay_buffer_size,
            **kwargs)
        self.observation_space_dim = environment.observation_space.shape[0]
        self.actions_space_dim = environment.action_space.shape[0]
        self.train_batch_size = train_batch_size
        self.train_interaction_steps = train_interaction_steps
        self.episode_length = episode_length
        self.warmup_policy = self._make_policy('random_mpc', kwargs['policy_params'], environment)
        self.warmup_timesteps = warmup_timesteps
        self.total_warmup_timesteps_so_far = 0
        assert all(key in kwargs.keys() for key in ('policy', 'policy_params', 'model', 'model_params')), \
            "Did not specify a policy or a model."
        kwargs['model_params']['scale_features'] = kwargs['scale_features']
        self.active_constraint = False
        self.c_model = None
        if kwargs['c_params'] is not None:
            self.active_constraint = True
            self.c_model = self._make_constraint_model(environment.get_cost, kwargs.pop('c_params'))

        self.model = self._make_model(kwargs.pop('model'), kwargs.pop('model_params'), environment,
                                      kwargs.pop('sampling_propagation'))
        self.policy = self._make_policy(kwargs.pop('policy'), kwargs.pop('policy_params'), environment)
        

    @property
    def warm(self):
        return self.total_warmup_timesteps_so_far >= self.warmup_timesteps

    def update(self, c_update=False):
        observations, actions, next_observations, _, _, infos = \
            self.replay_buffer.sample_recent_data(self.train_batch_size)
        goal_mets = np.array(list(map(lambda info: info.get('goal_met', False), infos)))
        # We masked transitions where the goal was met since they are non-continuous what extremely destabilizes
        # the learning of p(s_t_1 | s_t, a_t)
        masked_observations, masked_actions, masked_next_observations = \
            observations[~goal_mets, ...], actions[~goal_mets, ...], next_observations[~goal_mets, ...]
        observations_with_actions = np.concatenate([
            masked_observations,
            masked_actions], axis=1
        )
        self.model.fit(observations_with_actions, masked_next_observations)
        if self.active_constraint and c_update:
            self.c_model.fit(observations, actions, next_observations)

    def _interact(self, environment):
        if not self.warm:
            samples, timesteps_this_batch = self.sample_trajectories(
                environment,
                self.warmup_policy,
                self.warmup_timesteps,
                self.episode_length
            )
            self.total_warmup_timesteps_so_far += timesteps_this_batch
        else:
            samples, timesteps_this_batch = self.sample_trajectories(
                environment,
                self.policy,
                self.train_interaction_steps,
                self.episode_length
            )
        return samples, timesteps_this_batch

    def _build(self):
        self.model.build()
        self.policy.build()
        if self.active_constraint:
            self.c_model.build()
        logger.info("Done building Mbrl agent computational graph.")

    def _load(self):
        raise NotImplementedError

    def report(self,
               environment,
               eval_interaction_steps,
               eval_episode_length):
        logger.info("Evaluating policy.")
        evaluation_trajectories, _ = self.sample_trajectories(
            environment,
            self.policy,
            eval_interaction_steps,
            eval_episode_length)
        eval_return_values = np.array([trajectory['reward'].sum() for
                                       trajectory in evaluation_trajectories])
        trajectories_infos = [trajectory['info'] for trajectory in evaluation_trajectories]
        sum_costs = np.asarray([sum(list(map(lambda info: info.get('cost', 0.0), trajectory)))
                                for trajectory in trajectories_infos])
        self.training_report.update(dict(
            eval_rl_objective=eval_return_values.mean(),
            sum_rewards_stddev=eval_return_values.std(),
            eval_mean_sum_costs=sum_costs.mean(),
        ))
        return self.training_report

    def _make_policy(self, policy, policy_params, environment):
        eval_policy = eval(standardize_name(policy))
        policy_params['environment'] = environment
        if eval_policy == RandomMpc:
            return RandomMpc(policy_params['environment'].action_space)
        if policy_params is None:
            return eval((standardize_name(policy)))()
        return eval((standardize_name(policy)))(model=self.model, **policy_params)

    def _make_model(self, model, model_params, environment, sampling_propagation):
        return TransitionModel(
            model=model,
            c_model=self.c_model,
            observation_space=environment.observation_space,
            action_space=environment.action_space,
            sampling_propagation=sampling_propagation,
            **model_params)

    def _make_constraint_model(self, cost_function, c_params):
        return ConstraintModel(
            inputs_dim=self.observation_space_dim,
            outputs_dim=self.actions_space_dim,
            cost_function=cost_function,
            **c_params
        )
