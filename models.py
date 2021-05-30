import tensorflow.compat.v1 as tf
from tensorflow.keras.layers import Flatten, Dense, Input, Concatenate, Reshape
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from global_params import*
import numpy as np
# from mcts_nodes import*
tf.disable_v2_behavior()


def actualize_tensor(x):
    # s = tf.constant([0])
    # for k in range(len(support_vector_size * 2)):
    #     s = s + (k - support_vector_size)*x[k]
    indxs = tf.expand_dims(tf.range(-300, 301, delta=1), axis=0)
    indxs = tf.cast(indxs, tf.float32)
    s = tf.reduce_sum(tf.math.multiply(indxs, x), axis=1)
    s = tf.expand_dims(s, axis=1)
    return s


def unscale_tensor(x):
    a = tf.math.sqrt(1 + 4*scale_eps*(1 + scale_eps + tf.math.abs(x))) - 1
    return tf.math.sign(x)*((a/(2*scale_eps)) ** 2 - 1)


def create_hidden_state_model():
    """ this function returns a model to compute the representation function- function h in the paper. """
    inp = Input(shape=observation_dim)
    x = Flatten()(inp)
    x = Dense(64, activation='relu')(x)
    x = Dense(hidden_state_dim_flattened, activation='tanh')(x)
    x = Reshape(hidden_state_dim)(x)
    min_t = tf.math.reduce_min(x)
    max_t = tf.math.reduce_max(x)
    #x = (x - min_t)/(max_t - min_t)
    model = Model(inputs=inp, outputs=x)
    return model


def create_prediction_models():
    """ this function returns a model to compute the prediction function - function f in the paper"""
    inp = Input(shape=hidden_state_dim)
    x = Flatten()(inp)
    x = Dense(64, activation='swish')(x)

    policy_output_layer = Dense(16, activation='relu')(x)
    policy_output_layer = Dense(n_actions, activation='softmax')(policy_output_layer)
    value_output_layer = Dense(64, activation='linear', kernel_regularizer=regularizers.l2(c))(x)
    #value_output_layer = Dense(1)(value_output_layer)
    value_output_layer = Dense(support_vector_size * 2 + 1, activation='softmax',
                               kernel_regularizer=regularizers.l2(c), kernel_initializer=tf.keras.initializers.Zeros)(value_output_layer)
    model = Model(inputs=inp, outputs=[policy_output_layer, value_output_layer])
    return model


def create_dynamics_models():
    """ this function returns a model to compute the dynamics function - function g in the paper"""
    # the action is concatenated to another plane on the hidden state
    inp1 = Input(shape=hidden_state_dim)
    inp2 = Input(shape=hidden_state_dim[:-1] + (1, ))
    inp = Concatenate(axis=-1)([inp1, inp2])
    x = Flatten()(inp)
    x = Dense(64, activation='swish')(x)

    x1 = Dense(hidden_state_dim_flattened, activation='tanh')(x)
    x1 = Reshape(hidden_state_dim)(x1)

    #x2 = Dense(1)(x)

    reward_output_layer = Dense(support_vector_size * 2 + 1, activation='softmax',
                                kernel_regularizer=regularizers.l2(c), kernel_initializer=tf.keras.initializers.Zeros)(
        x)
    model = Model(inputs=[inp1, inp2], outputs=[x1, reward_output_layer])
    return model


def create_mu_model(k, all_models):
    """creates one mu model that will be trained end-to-end with f, h, and g from the article. will use recurrent
    activation of every model for K hypothetical steps (from the article k = 5)"""
    h = all_models['h']
    g = all_models['g']
    f = all_models['f']
    inp1 = Input(h.input_shape[1:])  #the first observation
    # ASSUMING INPUT 2 is all k actions already encoded each to 2 dim planes !!
    inp2 = Input((k, ) + hidden_state_dim[:-1] + (1, ))  # the k actions that were performed encoded into planes the size of the hidden state
    p = []
    v = []
    r = []
    s0 = h(inp1)
    cur_s = s0
    # p0, v0 = f(s0)
    for i in range(1, k + 1):
        cur_p, cur_v = f(cur_s)
        cur_a = inp2[:, i - 1, :, :]
        #input_g = Concatenate(axis=2)([cur_s, cur_a])
        cur_s, cur_r = g([cur_s, cur_a])
        p.append(tf.expand_dims(cur_p, axis=2))
        v.append(tf.expand_dims(cur_v, axis=2))
        r.append(tf.expand_dims(cur_r, axis=2))

    output_p = Concatenate(axis=2, name='policy_vectors')(p)
    output_v = Concatenate(axis=2, name='value_vectors')(v)
    output_r = Concatenate(axis=2, name='immediate_reward_vectors')(r)

    model = Model(inputs=[inp1, inp2], outputs=[output_p, output_v, output_r])
    return model


def create_reg_model(k, all_models):
    """creates one mu model that will be trained end-to-end with f, h, and g from the article. will use recurrent
    activation of every model for K hypothetical steps (from the article k = 5)"""
    h = all_models['h']
    g = all_models['g']
    f = all_models['reg_f']
    inp1 = Input(h.input_shape[1:])  #the first observation
    # ASSUMING INPUT 2 is all k actions already encoded each to 2 dim planes !!
    inp2 = Input((k, ) + hidden_state_dim[:-1] + (1, ))  # the k actions that were performed encoded into planes the size of the hidden state
    p = []
    v = []
    r = []
    s0 = h(inp1)
    cur_s = s0
    # p0, v0 = f(s0)
    for i in range(1, k + 1):
        cur_p, cur_v = f(cur_s)
        cur_a = inp2[:, i - 1, :, :]
        #input_g = Concatenate(axis=2)([cur_s, cur_a])
        cur_s, cur_r = g([cur_s, cur_a])
        p.append(tf.expand_dims(cur_p, axis=2))
        v.append(tf.expand_dims(cur_v, axis=2))
        r.append(tf.expand_dims(cur_r, axis=2))

    output_p = Concatenate(axis=2, name='policy_vectors')(p)
    output_v = Concatenate(axis=2, name='value_vectors')(v)
    output_r = Concatenate(axis=2, name='immediate_reward_vectors')(r)

    model = Model(inputs=[inp1, inp2], outputs=[output_p, output_v, output_r])
    return model


def get_reg_f(all_models):
    f = all_models['f']
    inp = Input(f.input_shape[1:])
    p_out, v_out = f(inp)
    actual_value = actualize_tensor(v_out)
    actual_value = unscale_tensor(actual_value)
    model = Model(inputs=[inp], outputs=[p_out, actual_value])
    return model


def get_reg_g(all_models):
    g = all_models['g']
    inp1 = Input(g.input_shape[0][1:])
    inp2 = Input(g.input_shape[1][1:])
    s_out, r_out = g([inp1, inp2])
    actual_reward = actualize_tensor(r_out)
    model = Model(inputs=[inp1, inp2], outputs=[s_out, actual_reward])
    return model


def custom_cat_loss(y_true, y_pred):
    # assuming y_true and pred are of length of k_hypothetical_steps
    all_loss = 0
    for k in range(k_hypothetical_steps):
        cur_true = y_true[:, :, k]
        cur_pred = y_pred[:, :, k]
        #all_loss += tf.reduce_mean(categorical_crossentropy(cur_true, cur_pred))
        all_loss += (categorical_crossentropy(cur_true, cur_pred))

    return 1/k_hypothetical_steps*all_loss


def custom_reg_loss(y_true, y_pred):
    all_loss = 0
    for k in range(k_hypothetical_steps):
        cur_true = y_true[:, :, k]
        cur_pred = y_pred[:, :, k]
        all_loss += tf.math.sqrt(tf.reduce_sum((cur_true - cur_pred) ** 2))

    return 1/k_hypothetical_steps*all_loss


def custom_cat_loss2(y_true, y_pred):
    # assuming y_true and pred are of length of k_hypothetical_steps
    all_loss = 0
    for k in range(k_hypothetical_steps):
        cur_true = y_true[:, :, k]
        cur_pred = y_pred[:, :, k]
        #all_loss += tf.reduce_mean(categorical_crossentropy(cur_true, cur_pred))
        all_loss += (categorical_crossentropy(cur_true, cur_pred))

    return 1/k_hypothetical_steps*all_loss*10
