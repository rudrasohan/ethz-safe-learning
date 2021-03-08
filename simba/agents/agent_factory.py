from simba.infrastructure.common import standardize_name
import simba.agents as agents


def make_agent(config, environment):
    agent_name = config['options']['agent']
    assert agent_name in config['agents'], "Specified agent does not exist."
    agent = eval('agents.' + standardize_name(agent_name))
    agent_params = config['agents'][agent_name]
    base_agent_params = config['agents']['agent']
    policy = agent_params['policy']
    assert policy in config['policies'], "Specified policy does not exist."
    policy_params = config['policies'][policy]
    model = agent_params['model']
    constraint_model = None#agent_params['constraint']
    assert model in config['models'], "Specified model does not exist."
    model_params = config['models'][model]
    c_params = None
    if constraint_model is not None:
        assert constraint_model in config['models'], "Specified constraint does not exist"
        c_params = config['models']['constraint_model']
        c_params['train_epochs'] = config['options']['train_iterations']
    if agent is agents.MbrlAgent:
        assert len(environment.action_space.shape) == 1 and \
               len(environment.observation_space.shape) == 1, "No support for non-flat action/observation spaces."
        kwargs = {**agent_params, **base_agent_params, 'policy_params': policy_params,
                  'model_params': model_params, 'c_params': c_params}
        config['models']['mlp_ensemble']['train_epochs'] = config['options']['train_iterations']
        return agents.MbrlAgent(environment=environment,
                                **kwargs)
