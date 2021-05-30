from global_params import*
import tensorflow.compat.v1 as tf
import gym
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from mcts_nodes import*
from IPython.display import clear_output
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from models import*
import os
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.disable_v2_behavior()


def play_game(cur_game_num, mu_model, all_models, cur_gen):

    train_pics = np.zeros((max_number_of_rounds_per_game,) + observation_dim)
    train_actions = np.zeros((max_number_of_rounds_per_game, n_actions))
    train_op_actions = np.zeros(max_number_of_rounds_per_game)
    train_action_value = np.zeros((max_number_of_rounds_per_game, 1))
    rewards_array = np.zeros(max_number_of_rounds_per_game)
    last_obs_actions = np.zeros(observation_last_actions_num)
    t_counter = 0
    env = gym.make("CartPole-v0")
    state_imgs_array = np.zeros((max_number_of_rounds_per_game, ) + state_dim[:-1])
    root_observation = env.reset()
    env.step(1)
    state_imgs_array[t_counter] = env.unwrapped.state
    observation_img = get_observation(state_imgs_array, last_obs_actions, t_counter)
    root = Root(observation_img, all_models)
    done = 0
    total_reward = 0
    while not done:

        if t_counter > max_number_of_rounds_per_game:
            break

        # PLANNING MOVE:
        plan_mcts(root, planning_depth)

        # PLAY BEST MOVE:
        action, best_child, p_choose, action_value = root.play_move(cur_gen)
        new_s, r, done, info = env.step(action)
        state_imgs_array[t_counter + 1] = env.unwrapped.state
        if len(last_obs_actions) > 0:
            last_obs_actions[0:(observation_last_actions_num - 1)] = last_obs_actions[1:observation_last_actions_num]
            last_obs_actions[observation_last_actions_num - 1] = action

        observation_img = get_observation(state_imgs_array, last_obs_actions, t_counter + 1)

        # ADD DATA TO REPLAY BUFFER:
        if is_to_render:
            env.render()
        if done:
            r = -2
        total_reward += r
        rewards_array[t_counter] = r

        train_pics[t_counter] = root.observation
        train_actions[t_counter] = p_choose
        train_action_value[t_counter] = action_value
        train_op_actions[t_counter] = action
        t_counter += 1

        if done:
            print("=====================================")
            print("Finished game " + str(cur_game_num) + " with reward = ", total_reward)
            print("=====================================")
            break
        # print('game: ' + str(cur_game_num))
        # print('action: ' + str(action))
        # print('reward: ' + str(r))
        # print('step: ' + str(t_counter))

        root = Root.from_node(best_child, observation_img)

    env.close()
    train_pics = train_pics[:t_counter]
    train_actions = train_actions[:t_counter]
    train_action_value = train_action_value[:t_counter]
    train_op_actions = train_op_actions[:t_counter]
    rewards_array = rewards_array[:t_counter]  # need to add discounting factor
    train_value = np.cumsum(rewards_array[::-1])[::-1]
    return train_pics, train_actions, train_value, train_action_value, total_reward, rewards_array, train_op_actions


def play_games_and_save_data(gen_num, mu_model, all_models):
    game_start = 0
    avg_reward = 0
    reg_model = create_reg_model(k_hypothetical_steps, all_models)
    game_reward_array = []
    if num_games_to_play <= game_start:
        return 0

    for k in range(game_start, num_games_to_play):
        train_pics, train_actions, train_value, train_action_value, total_reward, train_rewards, train_op_actions = \
            play_game(k, mu_model, all_models, gen_num)

        while len(train_pics) < k_hypothetical_steps:
            train_pics, train_actions, train_value, train_action_value, total_reward, train_rewards, train_op_actions\
                = play_game(k, mu_model, all_models, gen_num)

        train_input1, train_input2 = create_proper_inputs2(train_pics, train_op_actions)
        predicted_values = reg_model.predict([train_input1, train_input2])[1][:, 0, 0]
        predicted_values = np.concatenate((predicted_values, np.tile(predicted_values[-1], (td_steps,))))
        td_rewards = np.concatenate((train_rewards, np.tile(0, (td_steps,))))  # was train_rewards[-1]. since the same value is expected then the reward should stay 0
        train_value = create_train_values_by_bootstrap(predicted_values, td_rewards)
        train_value[np.where(train_value >= support_vector_size)] = support_vector_size - 1
        train_policy_stack, train_value_stack, train_reward_stack = create_proper_outputs2(train_op_actions, np.expand_dims(train_value, axis=-1),
                                                                                          np.expand_dims(train_rewards, axis=-1), train_actions)

        train_value_stack = scale_output(train_value_stack)
        train_value_stack = categorize_output(train_value_stack)
        train_reward_stack = categorize_output(train_reward_stack)
        if not(os.path.exists(generation_folder_path + str(gen_num))):
            os.mkdir(generation_folder_path + str(gen_num))
        np.save(generation_folder_path + str(gen_num) + '\\' + 'train_input1_' + str(k), train_input1)
        np.save(generation_folder_path + str(gen_num) + '\\' + 'train_input2_' + str(k), train_input2)
        np.save(generation_folder_path + str(gen_num) + '\\' + 'train_policy_stack' + str(k), train_policy_stack)
        np.save(generation_folder_path + str(gen_num) + '\\' + 'train_value_stack' + str(k), train_value_stack)
        np.save(generation_folder_path + str(gen_num) + '\\' + 'train_reward_stack' + str(k), train_reward_stack)

        clear_output()
        avg_reward += total_reward
        game_reward_array.append(total_reward)
    avg_reward = avg_reward / (num_games_to_play - game_start)

    del train_pics, train_value, train_actions, train_rewards, train_op_actions, train_action_value, total_reward, \
        train_input1, train_input2, train_policy_stack, train_value_stack, train_reward_stack

    return avg_reward, game_reward_array


def load_from_cur_gen_and_prev(gen_num, game_reward_array):

    train_input1 = np.zeros((100000, ) + observation_dim)
    train_input2 = np.zeros((100000, k_hypothetical_steps) + hidden_state_dim[:-1] + (1, ))
    train_policy_stack = np.zeros((100000, ) + (n_actions, k_hypothetical_steps))
    #  for regression on value:
    #  train_value_stack = np.zeros((100000, ) + (1, k_hypothetical_steps))
    #  for classification on value:
    train_value_stack = np.zeros((100000, ) + (support_vector_size * 2 + 1, k_hypothetical_steps))
    #train_reward_stack = np.zeros((100000, ) + (1, k_hypothetical_steps))
    train_reward_stack = np.zeros((100000, ) + (support_vector_size * 2 + 1, k_hypothetical_steps))

    t_count = 0
    for k in tqdm(range(0, num_games_to_play)):
        if gen_num <= 100 and k >= 50:
            break
        # if game_reward_array[k] < np.median(game_reward_array):
        #     continue
        cur_train_input1 = np.load(generation_folder_path + str(gen_num) + '\\' + 'train_input1_' + str(k) + '.npy')
        cur_train_input2 = np.load(generation_folder_path + str(gen_num) + '\\' + 'train_input2_' + str(k) + '.npy')
        cur_policy_stack = np.load(generation_folder_path + str(gen_num) + '\\' + 'train_policy_stack' + str(k) + '.npy')
        cur_value_stack = np.load(generation_folder_path + str(gen_num) + '\\' + 'train_value_stack' + str(k) + '.npy')
        cur_reward_stack = np.load(generation_folder_path + str(gen_num) + '\\' + 'train_reward_stack' + str(k) + '.npy')

        train_input1[t_count:(t_count + len(cur_train_input1))] = cur_train_input1
        train_input2[t_count:(t_count + len(cur_train_input2))] = cur_train_input2
        train_policy_stack[t_count:(t_count + len(cur_policy_stack))] = cur_policy_stack
        train_value_stack[t_count:(t_count + len(cur_value_stack))] = cur_value_stack
        train_reward_stack[t_count:(t_count + len(cur_reward_stack))] = cur_reward_stack
        t_count += len(cur_train_input1)

    # "Sliding window"
    cur_g = gen_num - 1
    while cur_g >= 0 and t_count < max_training_examples:
        for k in tqdm(range(0, int(num_games_to_play * history_window_coefficient))):
            if cur_g <= 100 and k >= 50:
                break
            cur_train_input1 = np.load(
                generation_folder_path + str(cur_g) + '\\' + 'train_input1_' + str(k) + '.npy')
            cur_train_input2 = np.load(
                generation_folder_path + str(cur_g) + '\\' + 'train_input2_' + str(k) + '.npy')
            cur_policy_stack = np.load(
                generation_folder_path + str(cur_g) + '\\' + 'train_policy_stack' + str(k) + '.npy')
            cur_value_stack = np.load(
                generation_folder_path + str(cur_g) + '\\' + 'train_value_stack' + str(k) + '.npy')
            cur_reward_stack = np.load(
                generation_folder_path + str(cur_g) + '\\' + 'train_reward_stack' + str(k) + '.npy')

            train_input1[t_count:(t_count + len(cur_train_input1))] = cur_train_input1
            train_input2[t_count:(t_count + len(cur_train_input2))] = cur_train_input2
            train_policy_stack[t_count:(t_count + len(cur_policy_stack))] = cur_policy_stack
            train_value_stack[t_count:(t_count + len(cur_value_stack))] = cur_value_stack
            train_reward_stack[t_count:(t_count + len(cur_reward_stack))] = cur_reward_stack
            t_count += len(cur_train_input1)
        break
        cur_g -= 1

    train_input1 = train_input1[:t_count]
    train_input2 = train_input2[:t_count]
    train_policy_stack = train_policy_stack[:t_count]
    train_value_stack = train_value_stack[:t_count]
    train_reward_stack = train_reward_stack[:t_count]
    return train_input1, train_input2, train_policy_stack, train_value_stack, train_reward_stack


def play_and_train_generations(start_generation):

    f = create_prediction_models()
    g = create_dynamics_models()
    h = create_hidden_state_model()
    all_models = {}
    all_models['f'] = f
    all_models['g'] = g
    all_models['h'] = h
    reg_f = get_reg_f(all_models)
    all_models['reg_f'] = reg_f
    reg_g = get_reg_g(all_models)
    all_models['reg_g'] = reg_g
    mu_model = create_mu_model(k=k_hypothetical_steps, all_models=all_models)
    saver = tf.train.Saver()

    avg_rewards = []
    for cur_gen in range(start_generation, num_generations):

        if cur_gen >= 1:
            mu_model.load_weights(model_folder_path + "model_gen_" + str(cur_gen - 1))

        if train_and_play:  # and cur_gen != start_generation:
            avg_reward, game_reward_array = play_games_and_save_data(cur_gen, mu_model, all_models)
            avg_rewards.append(avg_reward)
            print('Finished generation with average reward: ' + str(avg_reward))

        print('===============================================')
        print('GENERATION ' + str(cur_gen) + ' FINISHED PLAYING')
        print('===============================================')

        # loading data:
        print('loading data....')
        train_input1, train_input2, train_policy_stack, train_value_stack, train_reward_stack = \
            load_from_cur_gen_and_prev(cur_gen, game_reward_array)

        print('Loading Finished...')
        print('training size: ' + str(len(train_input1)))

        # formatting outputs:
        # print('Formating outputs...')
        # train_value_stack = scale_output(train_value_stack)
        # train_value_stack = categorize_output(train_value_stack)
        # train_reward_stack = scale_output(train_reward_stack)
        # train_reward_stack = categorize_output(train_reward_stack)

        # splitting data:
        print('Splitting data...')
        train_input1, val_input1, train_input2, val_input2, train_policy_stack, val_policy_stack, train_value_stack,\
            val_value_stack, train_reward_stack, val_reward_stack = train_test_split(
                train_input1, train_input2, train_policy_stack, train_value_stack, train_reward_stack)

        # Training agent:
        print('Training agent...')
        ada = Adam(lr=adam_lr/(np.floor(cur_gen/4) + 1))
        es = EarlyStopping(patience=100)
        #es = EarlyStopping(patience=3000, monitor='loss')
        mu_model.compile(optimizer=ada,
                         loss={'policy_vectors': custom_cat_loss, 'value_vectors': custom_cat_loss,
                               'immediate_reward_vectors': custom_cat_loss})

        history = mu_model.fit([train_input1, train_input2], [train_policy_stack, train_value_stack, train_reward_stack]
                               , validation_data=([val_input1, val_input2]
                                                  , [val_policy_stack, val_value_stack, val_reward_stack])
                               , batch_size=batch_size, epochs=num_epochs, callbacks=[es])

        # history = mu_model.fit([train_input1, train_input2], [train_policy_stack, train_value_stack, train_reward_stack]
        #                        , batch_size=batch_size, epochs=1000, callbacks=[es])

        print('Training complete')

        if is_to_plot:
            plt.figure()
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            if not(os.path.exists(save_res_dir)):
                os.mkdir(save_res_dir)
            if len(avg_rewards) != 0:
                plt.savefig(save_res_dir + 'training_gen' + str(cur_gen) + '_' + str(int(avg_rewards[-1])) + '.png')
                plt.close()
            else:
                plt.savefig(save_res_dir + 'training_gen' + str(cur_gen) + '.png')
                plt.close()

        if not (os.path.exists(model_folder_path)):
            os.mkdir(model_folder_path)

        mu_model.save_weights(model_folder_path + "model_gen_" + str(cur_gen))
        del train_input1, train_input2, train_policy_stack, train_value_stack, train_reward_stack


play_and_train_generations(0)





