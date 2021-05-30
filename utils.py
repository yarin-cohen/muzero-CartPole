import numpy as np
import cv2
from global_params import*


def get_pics_org(node, env):
    img_shape = env.img_shape
    pic = np.zeros((img_shape[0], img_shape[1], img_shape[2], num_frames))
    for k in range(num_frames):
        pic[:, :, :, k] = env.pics[node.snapshot]
        if node.parent is None:
            break
        node = node.parent

    return pic


def get_observation(state_imgs_array, actions_array, cur_ind):
    # observation of the environment is defined before-hand from list of latest states and latest actions
    # how many states backwards for one observation? observation_last_states_num
    # how many actions backwards for one observation? observation_last_actions_num
    pic = np.zeros(observation_dim[:-1] + (observation_last_states_num, ))

    for k in range(observation_last_states_num):
        if cur_ind - k < 0:
            break
        pic[:, k] = state_imgs_array[cur_ind - k, :]

    new_pic = pic
    actions_plane = np.zeros(state_dim_h[:-1] + (observation_last_actions_num, ))
    count = 0
    if cur_ind != 0:
        cur_ind -= 1

    for k in range(observation_last_actions_num):
        cur_action_plane = 1/n_actions * actions_array[k] * np.ones((state_dim_h[:-1]))
        actions_plane[:, count] = cur_action_plane
        count += 1

    final_pic = np.concatenate((new_pic, actions_plane), axis=1)
    return np.expand_dims(final_pic, axis=0)


def format_pics(train_pics):
    if len(train_pics.shape) == 4:
        new_train_pics = np.zeros((1, 64, 64, num_frames))
    else:
        new_train_pics = np.zeros((len(train_pics), state_dim[0], state_dim[1], state_dim[2]))
    for k in (range(len(new_train_pics))):
        for f in range(num_frames):
            if len(train_pics.shape) == 4:
                img = train_pics[:, :, :, f]
            else:
                img = train_pics[k, :, :, :, f]
            crop_image = img[34:-16, :, :]
            gray_image = crop_image.mean(-1, keepdims=True)
            resize_image = cv2.resize(gray_image, (64, 64))
            new_train_pics[k, :, :, f] = np.asarray(resize_image[..., np.newaxis] / 255.0, dtype='float32')[:, :, 0]

    return new_train_pics


def flip_horizontal(train_pics, train_actions, train_value, train_action_value):
    # flipping images to augment the data.
    new_train_pics = np.zeros(train_pics.shape)
    new_train_actions = np.zeros(train_actions.shape)
    new_train_value = np.zeros(train_value.shape)
    new_train_action_value = np.zeros(train_action_value.shape)
    print('augmenting data...')
    for k in range(len(train_pics)):
        new_train_pics[k, :, :, :] = cv2.flip(train_pics[k, :, :, :], 1)
        new_train_actions[k, :] = train_actions[k, :]
        t = new_train_actions[k, 2]
        new_train_actions[k, 2] = new_train_actions[k, 3]
        new_train_actions[k, 3] = t
        new_train_value[k] = train_value[k]
        new_train_action_value[k] = train_action_value[k]
    return new_train_pics, new_train_actions, new_train_value, new_train_action_value


def create_proper_inputs(train_pics, train_actions):
    #new_input1 = np.zeros((len(train_pics) - k_hypothetical_steps, ) + state_dim_h)
    new_input1 = train_pics[:(len(train_pics) - k_hypothetical_steps + 1)]
    new_input2 = np.zeros((len(train_pics) - k_hypothetical_steps + 1, k_hypothetical_steps) + hidden_state_dim[:-1] + (1, ))
    for k in range(0, len(train_pics) - k_hypothetical_steps + 1):
        #  encoding actions only, for concatination with hidden state for function g
        next_actions = train_actions[k:(k + k_hypothetical_steps)]
        action_planes = np.zeros((k_hypothetical_steps, ) + hidden_state_dim[:-1] + (1, ))
        for i in range(k_hypothetical_steps):
            action_planes[i, :, 0] = 1 / n_actions * next_actions[i] * np.ones((hidden_state_dim[0]))
        new_input2[k] = action_planes
    return new_input1, new_input2


#  with absorbing states
def create_proper_inputs2(train_pics, train_actions):
    #  padding
    pad_pics = np.tile(train_pics[-1], (k_hypothetical_steps, ) + tuple(map(tuple, np.ones((1, len(train_pics.shape) - 1), dtype=int)))[0])
    train_pics = np.concatenate((train_pics, pad_pics), axis=0)
    pad_acts = np.tile(train_actions[-1], (k_hypothetical_steps, ) + tuple(map(tuple, np.ones((1, len(train_actions.shape) - 1), dtype=int)))[0])
    train_actions = np.concatenate((train_actions, pad_acts), axis=0)

    new_input1 = train_pics[:(len(train_pics) - k_hypothetical_steps)]
    new_input2 = np.zeros((len(train_pics) - k_hypothetical_steps, k_hypothetical_steps) + hidden_state_dim[:-1] + (1, ))
    for k in range(0, len(train_pics) - k_hypothetical_steps):
        #  encoding actions only, for concatination with hidden state for function g
        next_actions = train_actions[k:(k + k_hypothetical_steps)]
        action_planes = np.zeros((k_hypothetical_steps, ) + hidden_state_dim[:-1] + (1, ))
        for i in range(k_hypothetical_steps):
            action_planes[i, :, 0] = 1 / n_actions * next_actions[i] * np.ones((hidden_state_dim[0]))
        new_input2[k] = action_planes
    return new_input1, new_input2


def create_proper_outputs(train_op_actions, train_value, train_rewards, train_actions):
    policy_stack = np.zeros((len(train_op_actions) - k_hypothetical_steps + 1, n_actions, k_hypothetical_steps))
    value_stack = np.zeros((len(train_op_actions) - k_hypothetical_steps + 1, 1, k_hypothetical_steps))
    reward_stack = np.zeros((len(train_op_actions) - k_hypothetical_steps + 1, 1, k_hypothetical_steps))

    for k in range(k_hypothetical_steps, len(train_op_actions) + 1):
        cur_policy_stack = np.zeros((n_actions, k_hypothetical_steps))
        for j in range(k_hypothetical_steps):
            cur_policy_stack[:, j] = train_actions[k - k_hypothetical_steps + j, :]
        policy_stack[k - k_hypothetical_steps] = cur_policy_stack
        value_stack[k - k_hypothetical_steps] = np.transpose(train_value[(k - k_hypothetical_steps):k, :])
        reward_stack[k - k_hypothetical_steps] = np.transpose(train_rewards[(k - k_hypothetical_steps):k, :])

    return policy_stack, value_stack, reward_stack


#  with absorbing states
def create_proper_outputs2(train_op_actions, train_value, train_rewards, train_actions):
    pad_op_acts = np.tile(train_op_actions[-1], (k_hypothetical_steps,) +
                          tuple(map(tuple, np.ones((1, len(train_op_actions.shape) - 1), dtype=int)))[0])
    pad_values = np.tile(train_value[-1], (k_hypothetical_steps,) +
                         tuple(map(tuple, np.ones((1, len(train_value.shape) - 1), dtype=int)))[0])
    pad_rewards = np.tile(train_rewards[-1], (k_hypothetical_steps,) +
                          tuple(map(tuple, np.ones((1, len(train_rewards.shape) - 1), dtype=int)))[0])
    pad_acts = np.tile(train_actions[-1], (k_hypothetical_steps,) +
                       tuple(map(tuple, np.ones((1, len(train_actions.shape) - 1), dtype=int)))[0])

    train_op_actions = np.concatenate((train_op_actions, pad_op_acts), axis=0)
    train_value = np.concatenate((train_value, pad_values), axis=0)
    train_rewards = np.concatenate((train_rewards, pad_rewards), axis=0)
    train_actions = np.concatenate((train_actions, pad_acts), axis=0)

    policy_stack = np.zeros((len(train_op_actions) - k_hypothetical_steps + 1, n_actions, k_hypothetical_steps))
    value_stack = np.zeros((len(train_op_actions) - k_hypothetical_steps + 1, 1, k_hypothetical_steps))
    reward_stack = np.zeros((len(train_op_actions) - k_hypothetical_steps + 1, 1, k_hypothetical_steps))

    for k in range(k_hypothetical_steps, len(train_op_actions) + 1):
        cur_policy_stack = np.zeros((n_actions, k_hypothetical_steps))
        for j in range(k_hypothetical_steps):
            cur_policy_stack[:, j] = train_actions[k - k_hypothetical_steps + j, :]
        policy_stack[k - k_hypothetical_steps] = cur_policy_stack
        value_stack[k - k_hypothetical_steps] = np.transpose(train_value[(k - k_hypothetical_steps):k, :])
        reward_stack[k - k_hypothetical_steps] = np.transpose(train_rewards[(k - k_hypothetical_steps):k, :])

    return policy_stack[:-1], value_stack[:-1], reward_stack[:-1]


def create_train_values_by_bootstrap(predicted_values, td_rewards):
    td_values = np.zeros(len(predicted_values) - td_steps)
    for k in range(len(td_values)):
        td_values[k] = np.sum(td_rewards[k:(k + td_steps)]) + predicted_values[k + td_steps]

    return td_values


def scale_output(x):
    """tensor scaling according to Appendix F of the article. this happens on the network side"""
    new_x = np.zeros(x.shape)
    for k in range(len(x)):
        new_x[k] = np.sign(x[k]) * (np.sqrt(np.abs(x[k]) + 1) - 1 + scale_eps * x[k])
    return new_x


def categorize_output(x):
    """tensor categorizing to use categorical crossentropy in calculating the value and reward, instead of MSE-App F"""
    vec = np.zeros((len(x), support_vector_size * 2 + 1, k_hypothetical_steps))
    for k in range(len(x)):
        for j in range(k_hypothetical_steps):
            ind1 = int(np.floor((x[k, 0, j])))
            remainder = x[k, 0, j] - ind1
            ind1 = ind1 + support_vector_size
            ind2 = ind1 + 1
            cur_vec = np.zeros((support_vector_size * 2 + 1,))
            cur_vec[ind1] = 1 - remainder
            cur_vec[ind2] = remainder
            vec[k, :, j] = cur_vec
            assert np.all(cur_vec >= 0)
            assert np.sum(cur_vec) == 1

    return vec


def calc_crossentropy(y1, y2):
    losses = []
    for label, pred in zip(y1, y2):
        pred /= pred.sum(axis=-1, keepdims=True)
        losses.append(np.sum(label * -np.log(pred), axis=-1, keepdims=False))
    return losses