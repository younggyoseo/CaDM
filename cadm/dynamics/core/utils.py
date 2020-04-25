import tensorflow as tf
import numpy as np
from baselines.common.distributions import make_pdtype

def create_plus_ensemble_cem_mlp(output_dim,
                                 hidden_sizes,
                                 hidden_nonlinearity,
                                 output_nonlinearity,
                                 input_obs_dim=None,
                                 input_act_dim=None,
                                 input_obs_var=None,
                                 input_act_var=None,
                                 n_forwards=1,
                                 ensemble_size=5,
                                 weight_decays=None,
                                 reward_fn=None,
                                 n_candidates=None,
                                 norm_obs_mean_var=None,
                                 norm_obs_std_var=None,
                                 norm_act_mean_var=None,
                                 norm_act_std_var=None,
                                 norm_delta_mean_var=None,
                                 norm_delta_std_var=None,
                                 norm_cp_act_mean_var=None,
                                 norm_cp_act_std_var=None,
                                 n_particles=20,
                                 bs_input_obs_var=None,
                                 bs_input_act_var=None,
                                 discrete=False,
                                 cem_init_mean_var=None,
                                 cem_init_var_var=None,
                                 obs_preproc_fn=None,
                                 obs_postproc_fn=None,
                                 deterministic=False,
                                 w_init=tf.contrib.layers.xavier_initializer(),
                                 b_init=tf.zeros_initializer(),
                                 reuse=False,
                                 ):

    proc_obs_dim = int(obs_preproc_fn(bs_input_obs_var).shape[-1])
    hidden_sizes = [proc_obs_dim+input_act_dim] + list(hidden_sizes)

    layers = []
    l2_regs = []
    for idx in range(len(hidden_sizes) - 1):
        layer, l2_reg = create_dense_layer(name='hidden_%d' % idx,
                                           ensemble_size=ensemble_size,
                                           input_dim=hidden_sizes[idx],
                                           output_dim=hidden_sizes[idx+1],
                                           activation=hidden_nonlinearity,
                                           weight_decay=weight_decays[idx])
        layers.append(layer)
        l2_regs.append(l2_reg)
    
    mu_layer, mu_l2_reg = create_dense_layer(name='output_mu',
                                             ensemble_size=ensemble_size,
                                             input_dim=hidden_sizes[-1],
                                             output_dim=output_dim,
                                             activation=output_nonlinearity,
                                             weight_decay=weight_decays[-1])
    logvar_layer, logvar_l2_reg = create_dense_layer(name='output_logvar',
                                                     ensemble_size=ensemble_size,
                                                     input_dim=hidden_sizes[-1],
                                                     output_dim=output_dim,
                                                     activation=output_nonlinearity,
                                                     weight_decay=weight_decays[-1])
    layers += [mu_layer, logvar_layer]
    l2_regs += [mu_l2_reg, logvar_l2_reg]

    max_logvar = tf.Variable(np.ones([1, output_dim])/2., dtype=tf.float32, name="max_log_var")
    min_logvar = tf.Variable(-np.ones([1, output_dim])*10, dtype=tf.float32, name="min_log_var")

    def forward(xx):
        for layer in layers[:-2]:
            xx = layer(xx)
        mu = layers[-2](xx)
        logvar = layers[-1](xx)

        denormalized_mu = denormalize(mu, norm_delta_mean_var, norm_delta_std_var)

        if deterministic:
            xx = denormalized_mu
        else:
            logvar = max_logvar - tf.nn.softplus(max_logvar - logvar)
            logvar = min_logvar + tf.nn.softplus(logvar - min_logvar)

            denormalized_logvar = logvar + 2 * tf.math.log(norm_delta_std_var)
            denormalized_std = tf.exp(denormalized_logvar / 2.0)

            xx = denormalized_mu + tf.random.normal(tf.shape(denormalized_mu)) * denormalized_std

        return xx, mu, logvar

    bs_input_proc_obs_var = obs_preproc_fn(bs_input_obs_var)
    bs_normalized_input_obs = normalize(bs_input_proc_obs_var, norm_obs_mean_var, norm_obs_std_var)
    bs_normalized_input_act = normalize(bs_input_act_var, norm_act_mean_var, norm_act_std_var)

    x = tf.concat([bs_normalized_input_obs, bs_normalized_input_act], 2)
    output_var, mu, logvar = forward(x)

    '''build inference graph for gpu inference'''
    '''episodic trajectory sampling(TS-inf) will be used'''

    n = n_candidates
    p = n_particles
    m = tf.shape(input_obs_var)[0]
    h = n_forwards
    obs_dim = input_obs_dim
    act_dim = input_act_dim

    num_elites = 50
    num_cem_iters = 5
    alpha = 0.1

    lower_bound = -1.0
    upper_bound = 1.0

    if cem_init_mean_var is not None:
        ############################
        ### CROSS ENTROPY METHOD ###
        ############################
        print("=" * 80)
        print("CROSS ENTROPY METHOD")
        print("=" * 80)
        mean = cem_init_mean_var # (m, h, act_dim)
        var = cem_init_var_var

        # input_obs_var: (m, obs_dim)

        for _ in range(num_cem_iters):
            lb_dist, ub_dist = mean - lower_bound, upper_bound - mean
            constrained_var = tf.minimum(tf.minimum(tf.square(lb_dist / 2), tf.square(ub_dist / 2)), var)
            repeated_mean = tf.tile(mean[:, None, :, :], [1, n, 1, 1]) # (m, n, h, act_dim)
            repeated_var = tf.tile(constrained_var[:, None, :, :], [1, n, 1, 1]) # (m, n, h, act_dim)
            actions = tf.random.truncated_normal([m, n, h, act_dim], repeated_mean, tf.sqrt(repeated_var))

            returns = 0
            observation = tf.tile(tf.reshape(input_obs_var, [m, 1, 1, obs_dim]), [1, n, p, 1]) # (m, n, p, obs_dim)

            for t in range(h):
                action = actions[:, :, t] # [m, n, act_dim]
                normalized_act = normalize(action, norm_act_mean_var, norm_act_std_var) # [m, n, act_dim]
                normalized_act = tf.tile(normalized_act[:, :, None, :], [1, 1, p, 1]) # [m, n, p, act_dim]
                normalized_act = tf.reshape(
                    tf.transpose(normalized_act, [2, 0, 1, 3]),
                    [ensemble_size, int(p/ensemble_size) * m * n, act_dim]
                ) # [ensemble_size, p/ensemble_size * m * n, act_dim]

                proc_observation = obs_preproc_fn(observation)
                normalized_proc_obs = normalize(proc_observation, norm_obs_mean_var, norm_obs_std_var) # [m, n, p, proc_obs_dim]
                normalized_proc_obs = tf.reshape(
                    tf.transpose(normalized_proc_obs, [2, 0, 1, 3]),
                    [ensemble_size, int(p/ensemble_size) * m * n, proc_obs_dim]
                ) # [ensemble_size, p/ensemble_size * m * n, proc_obs_dim ]

                x = tf.concat([normalized_proc_obs, normalized_act], 2) # [ensemble_size, p/ensemble_size * m * n, proc_obs_dim + act_dim]

                delta, *_ = forward(x)
                delta = tf.reshape(delta, [p, m, n, obs_dim])
                delta = tf.transpose(delta, [1, 2, 0, 3])

                next_observation = obs_postproc_fn(observation, delta) # [m, n, p, obs_dim]
                repeated_action = tf.tile(action[:, :, None, :], [1, 1, p, 1])
                
                reward = reward_fn(observation, repeated_action, next_observation)

                returns += reward # [m, n, p]
                observation = next_observation
            
            returns = tf.reduce_mean(returns, axis=2)
            _, elites_idx = tf.nn.top_k(returns, k=num_elites, sorted=True) # [m, num_elites]
            elites_idx += tf.range(0, m*n, n)[:, None]
            flat_elites_idx = tf.reshape(elites_idx, [m * num_elites]) # [m * num_elites]
            flat_actions = tf.reshape(actions, [m * n, h, act_dim])
            flat_elites = tf.gather(flat_actions, flat_elites_idx) # [m * num_elites, h, act_dim]
            elites = tf.reshape(flat_elites, [m, num_elites, h, act_dim])

            new_mean = tf.reduce_mean(elites, axis=1) # [m, h, act_dim]
            new_var = tf.reduce_mean(tf.square(elites - new_mean[:, None, :, :]), axis=1)

            mean = mean * alpha + (1 - alpha) * new_mean
            var = var * alpha + (1 - alpha) * new_var
        
        optimal_action_var = mean 

    else:
        #######################
        ### RANDOM SHOOTING ###
        ####################### 
        print("=" * 80)
        print("RANDOM SHOOTING")
        print("=" * 80)

        if discrete:
            action = tf.random.uniform([m, n, h], maxval=act_dim, dtype=tf.int32)
            normalized_action = tf.one_hot(action, act_dim)
        else:
            action = tf.random.uniform([m, n, h, act_dim], -1, 1)
            normalized_action = normalize(action, norm_act_mean_var, norm_act_std_var) # [m, n, h, act_dim]
        returns = 0

        observation = input_obs_var # [m, obs_dim]

        for t in range(h):
            if t == 0:
                cand_action = action[:, :, 0] # [m, n, act_dim]
                observation = tf.tile(tf.reshape(observation, [m, 1, 1, obs_dim]), [1, n, p, 1]) # [m, n, p, obs_dim]
            
            proc_observation = obs_preproc_fn(observation)
            normalized_proc_obs = normalize(proc_observation, norm_obs_mean_var, norm_obs_std_var) # [m, n, p, proc_obs_dim]
            normalized_proc_obs = tf.reshape(
                tf.transpose(normalized_proc_obs, [2, 0, 1, 3]),
                [ensemble_size, int(p/ensemble_size) * m * n, proc_obs_dim]
            ) # [ensemble_size, p/ensemble_size * m * n, proc_obs_dim]

            normalized_act = normalized_action[:, :, t] # [m, n, act_dim]
            normalized_act = tf.tile(normalized_act[:, :, None, :], [1, 1, p, 1]) # [m, n, p, act_dim]
            normalized_act = tf.reshape(
                tf.transpose(normalized_act, [2, 0, 1, 3]),
                [ensemble_size, int(p/ensemble_size) * m * n, act_dim]
            ) # [ensemble_size, p/ensemble_size * m * n, act_dim]

            x = tf.concat([normalized_proc_obs, normalized_act], 2) # [ensemble_size, p/ensemble_size * m * n, proc_obs_dim + act_dim]

            delta, *_ = forward(x)
            delta = tf.reshape(delta, [p, m, n, obs_dim])
            delta = tf.transpose(delta, [1, 2, 0, 3]) # [m, n, p, obs_dim]

            next_observation = obs_postproc_fn(observation, delta) # [m, n, p, obs_dim]
            if discrete:
                repeated_action = tf.tile(action[:, :, t][:, :, None], [1, 1, p]) # [m, n, p]
            else:
                repeated_action = tf.tile(action[:, :, t][:, :, None, :], [1, 1, p, 1]) # [m, n, p, act_dim]

            reward = reward_fn(observation, repeated_action, next_observation)
            returns += reward # [m, n, p]
            observation = next_observation

        returns = tf.reduce_mean(returns, axis=2) # [m, n]
        max_return_idxs = tf.argmax(returns, 1, output_type=tf.int32)
        max_return_idxs = max_return_idxs + tf.range(0, m*n, n) # [m,]
        if discrete:
            flat_cand_action = tf.reshape(cand_action, [m*n])
        else:
            flat_cand_action = tf.reshape(cand_action, [m*n, act_dim])
        optimal_action_var = tf.gather(flat_cand_action, max_return_idxs) # [m, act_dim]

    return input_obs_var, input_act_var, output_var, optimal_action_var, mu, logvar, max_logvar, min_logvar, l2_regs


def create_plus_cadm_ensemble_cem_mlp(output_dim,
                                      hidden_sizes,
                                      hidden_nonlinearity,
                                      output_nonlinearity,
                                      input_obs_dim=None,
                                      input_act_dim=None,
                                      input_obs_var=None,
                                      input_act_var=None,
                                      input_cp_obs_var=None,
                                      input_cp_act_var=None,
                                      n_forwards=1,
                                      ensemble_size=5,
                                      weight_decays=None,
                                      reward_fn=None,
                                      n_candidates=None,
                                      norm_obs_mean_var=None,
                                      norm_obs_std_var=None,
                                      norm_act_mean_var=None,
                                      norm_act_std_var=None,
                                      norm_delta_mean_var=None,
                                      norm_delta_std_var=None,
                                      norm_cp_obs_mean_var=None,
                                      norm_cp_obs_std_var=None,
                                      norm_cp_act_mean_var=None,
                                      norm_cp_act_std_var=None,
                                      norm_back_delta_mean_var=None,
                                      norm_back_delta_std_var=None,
                                      n_particles=20,
                                      bs_input_obs_var=None,
                                      bs_input_act_var=None,
                                      bs_input_cp_var=None,
                                      cp_output_dim=None,
                                      discrete=False,
                                      use_discrete_context=False,
                                      context_out_class=None,
                                      use_stacking=False,
                                      history_length=None,
                                      cem_init_mean_var=None,
                                      cem_init_var_var=None,
                                      obs_preproc_fn=None,
                                      obs_postproc_fn=None,
                                      deterministic=False,
                                      w_init=tf.contrib.layers.xavier_initializer(),
                                      b_init=tf.zeros_initializer(),
                                      reuse=False,
                                      build_policy_graph=False,
                                      cp_forward=None,
                                      ):

    proc_obs_dim = int(obs_preproc_fn(bs_input_obs_var).shape[-1])
    if bs_input_cp_var is None:
        cp_output_dim = 0
    else:
        if use_discrete_context:
            cp_output_dim = cp_output_dim * context_out_class
        elif use_stacking:
            cp_output_dim = (input_obs_dim + input_act_dim) * history_length

    hidden_sizes = [proc_obs_dim+input_act_dim+cp_output_dim] + list(hidden_sizes)

    layers = []
    l2_regs = []
    for idx in range(len(hidden_sizes) - 1):
        layer, l2_reg = create_dense_layer(name='hidden_%d' % idx,
                                           ensemble_size=ensemble_size,
                                           input_dim=hidden_sizes[idx],
                                           output_dim=hidden_sizes[idx+1],
                                           activation=hidden_nonlinearity,
                                           weight_decay=weight_decays[idx])
        layers.append(layer)
        l2_regs.append(l2_reg)
    
    mu_layer, mu_l2_reg = create_dense_layer(name='output_mu',
                                             ensemble_size=ensemble_size,
                                             input_dim=hidden_sizes[-1],
                                             output_dim=output_dim,
                                             activation=output_nonlinearity,
                                             weight_decay=weight_decays[-1])
    logvar_layer, logvar_l2_reg = create_dense_layer(name='output_logvar',
                                                     ensemble_size=ensemble_size,
                                                     input_dim=hidden_sizes[-1],
                                                     output_dim=output_dim,
                                                     activation=output_nonlinearity,
                                                     weight_decay=weight_decays[-1])
    layers += [mu_layer, logvar_layer]
    l2_regs += [mu_l2_reg, logvar_l2_reg]

    max_logvar = tf.Variable(np.ones([1, output_dim])/2., dtype=tf.float32, name="max_logvar")
    min_logvar = tf.Variable(-np.ones([1, output_dim])*10, dtype=tf.float32, name="min_logvar")

    def forward(xx, ablation=False):
        for layer in layers[:-2]:
            xx = layer(xx)
        ablation_last_layer = xx
        mu = layers[-2](xx)
        logvar = layers[-1](xx)

        if norm_delta_mean_var is not None:
            denormalized_mu = denormalize(mu, norm_delta_mean_var, norm_delta_std_var)
        else:
            denormalized_mu = denormalize(mu, norm_back_delta_mean_var, norm_back_delta_std_var)

        if deterministic:
            xx = denormalized_mu
        else:
            logvar = max_logvar - tf.nn.softplus(max_logvar - logvar)
            logvar = min_logvar + tf.nn.softplus(logvar - min_logvar)

            if norm_delta_mean_var is not None:
                denormalized_logvar = logvar + 2 * tf.math.log(norm_delta_std_var)
            else:
                denormalized_logvar = logvar + 2 * tf.math.log(norm_back_delta_std_var)
            denormalized_std = tf.exp(denormalized_logvar / 2.0)

            xx = denormalized_mu + tf.random.normal(tf.shape(denormalized_mu)) * denormalized_std

        if ablation:
            return xx, mu, logvar, ablation_last_layer
        else:
            return xx, mu, logvar

    bs_input_proc_obs_var = obs_preproc_fn(bs_input_obs_var)
    bs_normalized_input_obs = normalize(bs_input_proc_obs_var, norm_obs_mean_var, norm_obs_std_var)
    bs_normalized_input_act = normalize(bs_input_act_var, norm_act_mean_var, norm_act_std_var)

    if bs_input_cp_var is not None:
        x = tf.concat([bs_normalized_input_obs, bs_normalized_input_act, bs_input_cp_var], 2)
    else:
        x = tf.concat([bs_normalized_input_obs, bs_normalized_input_act], 2)
    output_var, mu, logvar, ablation_last_layer = forward(x, ablation=True)

    '''build inference graph for gpu inference'''
    '''episodic trajectory sampling(TS-inf) will be used'''
    n = n_candidates
    p = n_particles
    m = tf.shape(input_obs_var)[0]
    h = n_forwards
    obs_dim = input_obs_dim
    act_dim = input_act_dim

    num_elites = 50
    num_cem_iters = 5
    alpha = 0.1

    lower_bound = -1.0
    upper_bound = 1.0

    if build_policy_graph:
        # Manually forward cp_input_obs and cp_input_act for cp output for inference
        if bs_input_cp_var is not None:
            bs_input_cp_obs_var = tf.tile(input_cp_obs_var[None, :, :], (ensemble_size, 1, 1)) # (ensemble_size, m, obs_dim*history_length)
            bs_input_cp_act_var = tf.tile(input_cp_act_var[None, :, :], (ensemble_size, 1, 1)) # (ensemble_size, m, act_dim*history_length)
            bs_normalized_input_cp_obs = normalize(bs_input_cp_obs_var, norm_cp_obs_mean_var, norm_cp_obs_std_var)
            bs_normalized_input_cp_act = normalize(bs_input_cp_act_var, norm_cp_act_mean_var, norm_cp_act_std_var)
            bs_normalized_cp_x = tf.concat([bs_normalized_input_cp_obs, bs_normalized_input_cp_act], axis=-1)
            bs_input_cp_var = cp_forward(bs_normalized_cp_x, inference=True) # (ensemble_size, m, cp_output_dim)
            inference_cp_var = bs_input_cp_var
        else:
            inference_cp_var = None

        if cem_init_mean_var is not None:
            ############################
            ### CROSS ENTROPY METHOD ###
            ############################
            print("=" * 80)
            print("CROSS ENTROPY METHOD")
            print("=" * 80)
            mean = cem_init_mean_var # (m, h, act_dim)
            var = cem_init_var_var

            # input_obs_var: (m, obs_dim)
            # bs_input_cp_var: (ensemble_size, m, cp_output_dim)

            for _ in range(num_cem_iters):
                lb_dist, ub_dist = mean - lower_bound, upper_bound - mean
                constrained_var = tf.minimum(tf.minimum(tf.square(lb_dist / 2), tf.square(ub_dist / 2)), var)
                repeated_mean = tf.tile(mean[:, None, :, :], [1, n, 1, 1]) # (m, n, h, act_dim)
                repeated_var = tf.tile(constrained_var[:, None, :, :], [1, n, 1, 1]) # (m, n, h, act_dim)
                actions = tf.random.truncated_normal([m, n, h, act_dim], repeated_mean, tf.sqrt(repeated_var))

                returns = 0
                observation = tf.tile(tf.reshape(input_obs_var, [m, 1, 1, obs_dim]), [1, n, p, 1]) # (m, n, p, obs_dim)
                if bs_input_cp_var is not None:
                    bs_input_cp_var = tf.transpose(bs_input_cp_var, (1, 0, 2)) # (m, ensemble_size, cp_output_dim)
                    context = tf.tile(tf.reshape(bs_input_cp_var, [m, 1, ensemble_size, cp_output_dim]), [1, n, int(p/ensemble_size), 1]) # (m, n, p, cp_output_dim)
                    reshaped_context = tf.reshape(
                        tf.transpose(context, [2, 0, 1, 3]),
                        [ensemble_size, int(p/ensemble_size) * m * n, cp_output_dim]
                    ) # [ensemble_size, p/ensemble_size * m * n, cp_output_dim]

                for t in range(h):
                    action = actions[:, :, t] # [m, n, act_dim]
                    normalized_act = normalize(action, norm_act_mean_var, norm_act_std_var) # [m, n, act_dim]
                    normalized_act = tf.tile(normalized_act[:, :, None, :], [1, 1, p, 1]) # [m, n, p, act_dim]
                    normalized_act = tf.reshape(
                        tf.transpose(normalized_act, [2, 0, 1, 3]),
                        [ensemble_size, int(p/ensemble_size) * m * n, act_dim]
                    ) # [ensemble_size, p/ensemble_size * m * n, act_dim]

                    proc_observation = obs_preproc_fn(observation)
                    normalized_proc_obs = normalize(proc_observation, norm_obs_mean_var, norm_obs_std_var) # [m, n, p, proc_obs_dim]
                    normalized_proc_obs = tf.reshape(
                        tf.transpose(normalized_proc_obs, [2, 0, 1, 3]),
                        [ensemble_size, int(p/ensemble_size) * m * n, proc_obs_dim]
                    ) # [ensemble_size, p/ensemble_size * m * n, proc_obs_dim]

                    if bs_input_cp_var is not None:
                        x = tf.concat([normalized_proc_obs, normalized_act, reshaped_context], 2) # [ensemble_size, p/ensemble_size * m * n, proc_obs_dim + act_dim + cp_output_Dim]
                    else:
                        x = tf.concat([normalized_proc_obs, normalized_act], 2) # [ensemble_size, p/ensemble_size * m * n, proc_obs_dim + act_dim]

                    delta, *_ = forward(x)
                    delta = tf.reshape(delta, [p, m, n, obs_dim])
                    delta = tf.transpose(delta, [1, 2, 0, 3])

                    next_observation = obs_postproc_fn(observation, delta) # [m, n, p, obs_dim]
                    repeated_action = tf.tile(action[:, :, None, :], [1, 1, p, 1])
                    
                    reward = reward_fn(observation, repeated_action, next_observation)

                    returns += reward # [m, n, p]
                    observation = next_observation
                
                returns = tf.reduce_mean(returns, axis=2)
                _, elites_idx = tf.nn.top_k(returns, k=num_elites, sorted=True) # [m, num_elites]
                elites_idx += tf.range(0, m*n, n)[:, None]
                flat_elites_idx = tf.reshape(elites_idx, [m * num_elites]) # [m * num_elites]
                flat_actions = tf.reshape(actions, [m * n, h, act_dim])
                flat_elites = tf.gather(flat_actions, flat_elites_idx) # [m * num_elites, h, act_dim]
                elites = tf.reshape(flat_elites, [m, num_elites, h, act_dim])

                new_mean = tf.reduce_mean(elites, axis=1) # [m, h, act_dim]
                new_var = tf.reduce_mean(tf.square(elites - new_mean[:, None, :, :]), axis=1)

                mean = mean * alpha + (1 - alpha) * new_mean
                var = var * alpha + (1 - alpha) * new_var
            
            optimal_action_var = mean 

        else:
            #######################
            ### RANDOM SHOOTING ###
            ####################### 
            print("=" * 80)
            print("RANDOM SHOOTING")
            print("=" * 80)

            if discrete:
                action = tf.random.uniform([m, n, h], maxval=act_dim, dtype=tf.int32)
                normalized_action = tf.one_hot(action, act_dim) # [m, n, h, act_dim]
            else:
                action = tf.random.uniform([m, n, h, act_dim], -1, 1)
                normalized_action = normalize(action, norm_act_mean_var, norm_act_std_var) # [m, n, h, act_dim]
            returns = 0

            observation = input_obs_var # [m, obs_dim]

            for t in range(h):
                if t == 0:
                    cand_action = action[:, :, 0] # [m, n, act_dim]
                    observation = tf.tile(tf.reshape(observation, [m, 1, 1, obs_dim]), [1, n, p, 1]) # [m, n, p, obs_dim]
                    if bs_input_cp_var is not None:
                        bs_input_cp_var = tf.transpose(bs_input_cp_var, (1, 0, 2)) # (m, ensemble_size, cp_output_dim)
                        context = tf.tile(tf.reshape(bs_input_cp_var, [m, 1, ensemble_size, cp_output_dim]), [1, n, int(p/ensemble_size), 1]) # (m, n, p, cp_output_dim)
                        reshaped_context = tf.reshape(
                            tf.transpose(context, [2, 0, 1, 3]),
                            [ensemble_size, int(p/ensemble_size) * m * n, cp_output_dim]
                        ) # [ensemble_size, p/ensemble_size * m * n, cp_output_dim]

                
                proc_observation = obs_preproc_fn(observation)
                normalized_proc_obs = normalize(proc_observation, norm_obs_mean_var, norm_obs_std_var) # [m, n, p, proc_obs_dim]
                normalized_proc_obs = tf.reshape(
                    tf.transpose(normalized_proc_obs, [2, 0, 1, 3]),
                    [ensemble_size, int(p/ensemble_size) * m * n, proc_obs_dim]
                ) # [ensemble_size, p/ensemble_size * m * n, proc_obs_dim]

                normalized_act = normalized_action[:, :, t] # [m, n, act_dim]
                normalized_act = tf.tile(normalized_act[:, :, None, :], [1, 1, p, 1]) # [m, n, p, act_dim]
                normalized_act = tf.reshape(
                    tf.transpose(normalized_act, [2, 0, 1, 3]),
                    [ensemble_size, int(p/ensemble_size) * m * n, act_dim]
                ) # [ensemble_size, p/ensemble_size * m * n, act_dim]

                if bs_input_cp_var is not None:
                    x = tf.concat([normalized_proc_obs, normalized_act, reshaped_context], 2) # [ensemble_size, p/ensemble_size * m * n, proc_obs_dim + act_dim + cp_output_Dim]
                else:
                    x = tf.concat([normalized_proc_obs, normalized_act], 2) # [ensemble_size, p/ensemble_size * m * n, proc_obs_dim + act_dim]

                delta, *_ = forward(x)
                delta = tf.reshape(delta, [p, m, n, obs_dim])
                delta = tf.transpose(delta, [1, 2, 0, 3]) # [m, n, p, obs_dim]

                next_observation = obs_postproc_fn(observation, delta) # [m, n, p, obs_dim]
                if discrete:
                    repeated_action = tf.tile(action[:, :, t][:, :, None], [1, 1, p]) # [m, n, p]
                else:
                    repeated_action = tf.tile(action[:, :, t][:, :, None, :], [1, 1, p, 1]) # [m, n, p, act_dim]

                reward = reward_fn(observation, repeated_action, next_observation)
                returns += reward # [m, n, p]
                observation = next_observation

            returns = tf.reduce_mean(returns, axis=2) # [m, n]
            max_return_idxs = tf.argmax(returns, 1, output_type=tf.int32)
            max_return_idxs = max_return_idxs + tf.range(0, m*n, n) # [m,]
            if discrete:
                flat_cand_action = tf.reshape(cand_action, [m*n])
            else:
                flat_cand_action = tf.reshape(cand_action, [m*n, act_dim])
            optimal_action_var = tf.gather(flat_cand_action, max_return_idxs) # [m, act_dim]
    else:
        optimal_action_var = None
        inference_cp_var = None

    return input_obs_var, input_act_var, output_var, optimal_action_var, mu, logvar, max_logvar, min_logvar, l2_regs, inference_cp_var, ablation_last_layer


def create_ensemble_pure_context_predictor(context_hidden_sizes,
                                           context_hidden_nonlinearity,
                                           output_nonlinearity,
                                           ensemble_size=None,
                                           cp_input_dim=None,
                                           context_weight_decays=None,
                                           bs_input_cp_obs_var=None,
                                           bs_input_cp_act_var=None,
                                           norm_cp_obs_mean_var=None,
                                           norm_cp_obs_std_var=None,
                                           norm_cp_act_mean_var=None,
                                           norm_cp_act_std_var=None,
                                           cp_output_dim=0,
                                           reuse=False,
                                           ):
    #################################
    ### CONTINUOUS CONTEXT VECTOR ###
    #################################
    print("=" * 80)
    print("CONTINUOUS CONTEXT VECTOR")
    print("=" * 80)

    context_hidden_sizes = [cp_input_dim] + list(context_hidden_sizes)

    layers = []
    l2_regs = []
    for idx in range(len(context_hidden_sizes) - 1):
        layer, l2_reg = create_dense_layer(name='cp_hidden_%d' % idx,
                                           ensemble_size=ensemble_size,
                                           input_dim=context_hidden_sizes[idx],
                                           output_dim=context_hidden_sizes[idx+1],
                                           activation=context_hidden_nonlinearity,
                                           weight_decay=context_weight_decays[idx])
        layers.append(layer)
        l2_regs.append(l2_reg)
    
    layer, l2_reg = create_dense_layer(name='cp_output',
                                       ensemble_size=ensemble_size,
                                       input_dim=context_hidden_sizes[-1],
                                       output_dim=cp_output_dim,
                                       activation=output_nonlinearity,
                                       weight_decay=context_weight_decays[-1])
    layers += [layer]
    l2_regs += [l2_reg]
    
    def forward(xx, inference=False):
        for layer in layers:
            xx = layer(xx)
        return xx
    
    bs_normalized_input_cp_obs = normalize(bs_input_cp_obs_var, norm_cp_obs_mean_var, norm_cp_obs_std_var)
    bs_normalized_input_cp_act = normalize(bs_input_cp_act_var, norm_cp_act_mean_var, norm_cp_act_std_var)
    bs_normalized_cp_x = tf.concat([bs_normalized_input_cp_obs, bs_normalized_input_cp_act], axis=-1)
    bs_cp_output_var = forward(bs_normalized_cp_x)

    return bs_cp_output_var, l2_regs, forward


def normalize(data_array, mean, std):
    return (data_array - mean) / (std + 1e-10)


def denormalize(data_array, mean, std):
    return data_array * (std + 1e-10) + mean


def create_dense_layer(name, ensemble_size, input_dim, output_dim, activation, weight_decay=0.0):
    weight = tf.compat.v1.get_variable("{}_weight".format(name),
                                shape=[ensemble_size, input_dim, output_dim],
                                initializer=tf.truncated_normal_initializer(stddev=1/(2*np.sqrt(input_dim))))
    bias = tf.compat.v1.get_variable("{}_bias".format(name),
                            shape=[ensemble_size, 1, output_dim],
                            initializer=tf.constant_initializer(0.0))
    l2_reg = tf.multiply(weight_decay, tf.nn.l2_loss(weight), name='{}_l2_reg'.format(name))
    def _thunk(input_tensor):
        out = tf.matmul(input_tensor, weight) + bias
        out = activation(out)
        return out
    return _thunk, l2_reg
