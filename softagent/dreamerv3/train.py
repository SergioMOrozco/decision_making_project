import os
from envs.env import Env
import dreamerv3
from dreamerv3 import embodied
from embodied.envs import from_gym


def update_env_kwargs(vv):
    new_vv = vv.copy()
    for v in vv:
        if v.startswith('env_kwargs_'):
            arg_name = v[len('env_kwargs_'):]
            new_vv['env_kwargs'][arg_name] = vv[v]
            del new_vv[v]
    return new_vv


def run_task(arg_vv, log_dir, exp_name):
    if arg_vv['algorithm'] == 'planet':
        from planet.config import DEFAULT_PARAMS
    else:
        raise NotImplementedError

    # See configs.yaml for all options.
    config = embodied.Config(dreamerv3.configs['defaults'])
    config = config.update(dreamerv3.configs['medium'])
    config = config.update({
      'logdir': '~/logdir/run1',
      'run.train_ratio': 64,
      'run.log_every': 30,  # Seconds
      'batch_size': 16,
      'jax.prealloc': False,
      'encoder.mlp_keys': '$^',
      'decoder.mlp_keys': '$^',
      'encoder.cnn_keys': 'image',
      'decoder.cnn_keys': 'image',
      # 'jax.platform': 'cpu',
    })
    config = embodied.Flags(config).parse()

    logdir = embodied.Path(config.logdir)
    step = embodied.Counter()
    logger = embodied.Logger(step, [
      embodied.logger.TerminalOutput(),
      embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
      embodied.logger.TensorBoardOutput(logdir),
      # embodied.logger.WandBOutput(logdir.name, config),
      # embodied.logger.MLFlowOutput(logdir.name),
    ])

    vv = DEFAULT_PARAMS
    vv.update(**arg_vv)
    vv = update_env_kwargs(vv)
    vv['max_episode_length'] = vv['env_kwargs']['horizon']

    env = Env(vv['env_name'], vv['symbolic_env'], vv['seed'], vv['max_episode_length'], vv['action_repeat'], vv['bit_depth'], vv['image_dim'],
              env_kwargs=vv['env_kwargs'])

    env = from_gym.FromGym(env, obs_key='image')  # Or obs_key='vector'.
    env = dreamerv3.wrap_env(env, config)
    env = embodied.BatchEnv([env], parallel=False)
    agent = dreamerv3.Agent(env.observation_space, env.act_space, step, config)

    replay = embodied.replay.Uniform(
      config.batch_length, config.replay_size, logdir / 'replay')

    args = embodied.Config(
      **config.run, logdir=config.logdir,
      batch_steps=config.batch_size * config.batch_length)

    embodied.run.train(agent, env, replay, logger, args)
