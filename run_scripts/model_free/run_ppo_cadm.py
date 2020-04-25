from cadm.logger import logger
from cadm.envs.normalized_env import normalize
from cadm.model_free import ppo_cadm, policies
from cadm.samplers import vectorized_env_executor
from cadm.utils.utils import ClassEncoder
from cadm.dynamics.mlp_cadm_ensemble_cem_dynamics import MLPEnsembleCEMDynamicsModel
from cadm.samplers.model_sample_processor import ModelSampleProcessor
from cadm.envs.config import get_environment_config

import tensorflow as tf
import json
import os
import gym
import argparse

def run_experiment(config):
    env, config = get_environment_config(config)

    # Save final config after editing config with respect to each environment.
    EXP_NAME = config['save_name']
    EXP_NAME += 'hidden_' + str(config['dim_hidden']) + '_lr_' + str(config['learning_rate'])
    EXP_NAME += '_horizon_' + str(config['horizon']) + '_seed_' + str(config['seed'])

    exp_dir = os.getcwd() + '/data/' + EXP_NAME + '/' + config.get('exp_name', '')
    logger.configure(dir=exp_dir, format_strs=['stdout', 'log', 'csv'], snapshot_mode='last', only_test=config['only_test_flag'])
    json.dump(config, open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)

    tf_config = tf.compat.v1.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    with tf.compat.v1.Session(config=tf_config):
        dynamics_model = MLPEnsembleCEMDynamicsModel(
            name="dyn_model",
            env=env,
            learning_rate=config['learning_rate'],
            hidden_sizes=config['hidden_sizes'],
            valid_split_ratio=config['valid_split_ratio'],
            rolling_average_persitency=config['rolling_average_persitency'],
            hidden_nonlinearity=config['hidden_nonlinearity'],
            batch_size=config['batch_size'],
            normalize_input=config['normalize_flag'],
            n_forwards=config['horizon'],
            n_candidates=config['n_candidates'],
            ensemble_size=config['ensemble_size'],
            n_particles=config['n_particles'],
            use_cem=config['use_cem'],
            deterministic=config['deterministic'],
            weight_decays=config['weight_decays'],
            weight_decay_coeff=config['weight_decay_coeff'],
            cp_hidden_sizes=config['context_hidden_sizes'],
            context_weight_decays=config['context_weight_decays'],
            context_out_dim=config['context_out_dim'],
            context_hidden_nonlinearity=config['context_hidden_nonlinearity'],
            history_length=config['history_length'],
            future_length=config['future_length'],
            state_diff=config['state_diff'],
            back_coeff=config['back_coeff'],
        )

        policy = policies.MlpCPPolicy
        ppo_cadm.learn(policy=policy,
                       dynamics_model=dynamics_model,
                       load_path=config['load_path'],
                       env=env,
                       nsteps=config['num_steps'],
                       ent_coef=config['entropy_coeff'],
                       lr=lambda f : f * config['learning_rate'],
                       vf_coef=config['vf_coef'],
                       max_grad_norm=config['max_grad_norm'],
                       gamma=config['gamma'],
                       lam=0.95,
                       log_interval=1,
                       nminibatches=config['num_minibatches'],
                       noptepochs=config['ppo_epochs'],
                       cliprange=lambda f : f * 0.2,
                       save_interval=config['save_interval'],
                       n_parallel=config['n_parallel'],
                       num_rollouts=config['num_rollouts'],
                       max_path_length=config['max_path_length'],
                       seed=config['seed'],
                       hidden_size=config['hidden_size'],
                       test_range=config['test_range'],
                       num_test=config['num_test'],
                       total_test=config['total_test'],
                       test_interval=config['test_interval'],
                       env_flag=config['dataset'],
                       total_timesteps=config['total_timesteps'],
                       normalize_flag=config['normalize_flag'],
                       no_test_flag=config['no_test_flag'],
                       only_test_flag=config['only_test_flag'],
                       state_diff=config['state_diff'],
                       history_length=config['history_length'],
                       cp_dim_output=config['context_out_dim'],
                       n_layers=config['n_layers'],)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='context conditional ppo')
    parser.add_argument('--save_name', default='PPO_CADM/', help="experiments name")
    parser.add_argument('--seed', type=int, default=0, help='random_seed')
    parser.add_argument('--dataset', default='length', help='dataset flag [length, mass, force]')
    parser.add_argument('--hidden_size', type=int, default=64, help='size of hidden feature')
    parser.add_argument('--model_hidden_size', type=int, default=200, help='size of model hidden feature')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning_rate')
    parser.add_argument('--num_steps', type=int, default=200, help='number of steps')
    parser.add_argument('--entropy_coeff', type=float, default=0.01, help='entropy coefficient')
    parser.add_argument('--num_minibatches', type=int, default=4, help='ppo minibatches')
    parser.add_argument('--ppo_epochs', type=int, default=8, help='ppo epochs')
    parser.add_argument('--total_test', type=int, default=20, help='# of test')
    parser.add_argument('--normalize_flag', action='store_true', help='flag to normalize')
    parser.add_argument('--num_rollouts', type=int, default=10, help='# of parallel training envs')
    parser.add_argument('--no_test_flag', action='store_true', help='flag to disable test')
    parser.add_argument('--only_test_flag', action='store_true', help='flag to enable only test')
    parser.add_argument('--load_path', type=str, default='', help='dynamics model load path')

    parser.add_argument('--horizon', type=int, default=30, help='horrizon for planning')
    parser.add_argument('--ensemble_size', type=int, default=5, help='size of ensembles')
    parser.add_argument('--n_particles', type=int, default=20, help='size of particles in trajectory sampling')
    parser.add_argument('--policy_type', type=str, default='RS', help='Policy Type')
    parser.add_argument('--deterministic_flag', type=int, default=0, help='flag to use deterministic dynamics model')
    parser.add_argument('--state_diff', type=int, default=1, help='flag to use delta state')
    parser.add_argument('--context_out_dim', type=int, default=10, help='dimension of context vector')
    parser.add_argument('--n_candidate', type=int, default=1000, help='candidate for planning')
    parser.add_argument('--back_coeff', type=float, default=0.5, help='coefficient for backward recon')
    parser.add_argument('--history_length', type=int, default=10, help='length of history')
    parser.add_argument('--future_length', type=int, default=10, help='length of future prediction')

    parser.add_argument('--n_layers', type=int, default=2, help='n_layers')

    args = parser.parse_args()

    if args.normalize_flag:
        args.save_name = "/NORMALIZED/" + args.save_name
    else:
        args.save_name = "/RAW/" + args.save_name
    
    if args.dataset == 'cartpole':
        args.save_name = "/CARTPOLE/" + args.save_name
    elif args.dataset == 'pendulum':
        args.save_name = "/PENDULUM/" + args.save_name
    elif args.dataset == 'halfcheetah':
        args.save_name = "/HALFCHEETAH/" + args.save_name
    elif args.dataset == 'cripple_halfcheetah':
        args.save_name = "/CRIPPLE_HALFCHEETAH/" + args.save_name
    elif args.dataset == 'ant':
        args.save_name = "/ANT/" + args.save_name
    elif args.dataset == 'slim_humanoid':
        args.save_name = "/SLIM_HUMANOID/" + args.save_name
    else:
        raise ValueError(args.dataset)
 
    if args.deterministic_flag == 0:
        args.save_name += "PROB/"
    else:
        args.save_name += "DET/"

    if args.policy_type in ['RS', 'CEM']:
        args.save_name += "{}/".format(args.policy_type)
        args.save_name += "CAND_{}/".format(args.n_candidate)
    else:
        raise ValueError(args.policy_type)

    args.save_name += "H_{}/".format(args.history_length)
    args.save_name += "F_{}/".format(args.future_length)
    args.save_name += "BACK_COEFF_{}/".format(args.back_coeff)
    if args.state_diff:
        args.save_name += "DIFF/"
    else:
        args.save_name += "WHOLE/"

    args.save_name += '/num_rollouts_' + str(args.num_rollouts) + '/'
    args.save_name += '/normal_ppo/'

    config = {
            # PPO
            'num_steps': args.num_steps,
            'entropy_coeff': args.entropy_coeff,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'gamma': 0.99,
            'lam': 0.95,
            'num_minibatches': args.num_minibatches,
            'ppo_epochs': args.ppo_epochs,
            'hidden_size': args.hidden_size,
            'total_timesteps': 5000000,
            'normalize_flag': args.normalize_flag,
            
            # etc
            'save_name': args.save_name,
            'dataset': args.dataset,
            'n_parallel': 5,
            'num_rollouts': args.num_rollouts,
            'total_test': 20,
            'log_interval': 1,
            'save_interval': int(10000 / args.num_steps),
            'test_interval': int(1000 / args.num_steps),
            'seed': args.seed,
            'no_test_flag': args.no_test_flag,
            'only_test_flag': args.only_test_flag,

            # Training
            'learning_rate': args.lr,
            'batch_size': 128,
            'valid_split_ratio': 0.1,
            'rolling_average_persitency': 0.99,

            # Dynamics Model Hyperparameters
            'dim_hidden': args.model_hidden_size,
            'hidden_sizes': (args.model_hidden_size,) * 4,
            'hidden_nonlinearity': 'swish',
            'deterministic': (args.deterministic_flag > 0),
            'weight_decays': (0.000025, 0.00005, 0.000075, 0.000075, 0.0001),
            'weight_decay_coeff': 1.0,
            'load_path': args.load_path,

            # PE-TS Hyperparameters
            'ensemble_size': args.ensemble_size,
            'n_particles': args.n_particles,

            # CaDM Hyperparameters
            'context_hidden_sizes': (256, 128, 64),
            'context_weight_decays': (0.000025, 0.00005, 0.000075),
            'context_out_dim': args.context_out_dim,
            'context_hidden_nonlinearity': 'relu',
            'history_length': args.history_length,
            'future_length': args.future_length,
            'state_diff': args.state_diff,
            'back_coeff': args.back_coeff,

            # Policy
            'n_candidates': args.n_candidate,
            'horizon': args.horizon,

            # Policy - CEM Hyperparameters
            'use_cem': args.policy_type == 'CEM',

            # PPO Architecture
            'n_layers': args.n_layers,
            }

    run_experiment(config)