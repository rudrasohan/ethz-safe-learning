def main():
    import os
    import argparse
    import logging
    import time
    import numpy as np
    import tensorflow as tf
    from config.config import load_config_or_die, pretty_print
    from simba.infrastructure.logging_utils import init_loggging
    from simba.infrastructure.common import dump_string, get_git_hash
    from simba.agents.agent_factory import make_agent
    from simba.environment_utils.environment_factory import make_environment
    from simba.infrastructure.trainer import RLTrainer
    parser = argparse.ArgumentParser()
    log_dir_suffix = time.strftime("%d-%m-%Y_%H-%M-%S")
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--log_dir', type=str, default='experiments')
    parser.add_argument('--log_level', type=str, default='INFO')
    parser.add_argument('--config_dir', type=str, required=True)
    parser.add_argument('--config_basename', type=str, required=True)
    parser.add_argument('--cuda_device', type=str)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    if args.cuda_device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
    experiment_log_dir = args.log_dir + '/' + args.name + '_' + log_dir_suffix
    os.makedirs(experiment_log_dir, exist_ok=True)
    init_loggging(args.log_level)
    params = load_config_or_die(args.config_dir, args.config_basename)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    logging.getLogger('simba').info("Startning a training session with parameters:\n" +
                                    pretty_print(params))
    dump_string(pretty_print(params) + '\n' +
                'git hash: ' + get_git_hash(), experiment_log_dir + '/params.txt')
    #import pdb; pdb.set_trace()
    env = make_environment(params)
    agent = make_agent(params, env)
    trainer_options = params['options'].pop('trainer_options')
    trainer_options['training_logger_params'].update({
        'log_dir': (experiment_log_dir + '/training_data')
    })
    trainer = RLTrainer(agent=agent,
                        environemnt=env,
                        **trainer_options)
    trainer.train(params['options'].pop('train_iterations'))


if __name__ == '__main__':
    main()
