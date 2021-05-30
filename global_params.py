n_actions         = 2
c                 = 0 #1e-3 #1e-2 #was 1e-2 up to training model7 #regularizer constant
dirichlet_alpha   = [0.25, 0.25]#[4, 4, 4, 4]
dir_noise_weight  = 0.25 #0.5
adam_lr           = 5e-3 #1e-3
#move_temp_initial = 0.01 #0.8
move_temps         = [0.7, 0.7, 0.7, 0.4]
training_steps_temps = [10, 20, 30, 40]
state_dim = (4, 1)#(64, 64, 4)
observation_last_states_num = 1
observation_last_actions_num = 0
num_frames        = observation_last_states_num
state_dim_h = (4, observation_last_states_num + observation_last_actions_num)#(64, 64, 9)  #4 is the state dim, 5 = 3 (last states) + 3 (last actions)
hidden_state_dim = state_dim#(4, 2)#(7, 7, 6)
k_hypothetical_steps = 5
img_shape = (64, 64, 1)
org_img_shape = (210, 160, 3)
c_shrink = 0.4
a_shrink = 10
c_puct1 = 1.25
c_puct2 = 19652
support_vector_size = 300
scale_eps = 0.001


train_and_play = True
is_to_render = False
history_window_coefficient = 1
max_number_of_rounds_per_game = 5000
num_games_to_play = 50#50
planning_depth = 11#11
num_generations = 200
num_evaluation_games = 5
is_to_augment_data = 0
num_epochs = 700#5#200
batch_size = 128
test_size = 0.1
is_to_plot = 1
save_res_dir = 'generation plots path - enter here'
generation_folder_path = 'training data path - enter here'
model_folder_path = 'model weights path - enter here'


observation_dim = (4, observation_last_states_num + observation_last_actions_num)
hidden_state_dim = (4, 1)

observation_dim_flattened = 1
for k in range(len(observation_dim)):
    observation_dim_flattened *= observation_dim[k]
hidden_state_dim_flattened = 1
for k in range(len(hidden_state_dim)):
    hidden_state_dim_flattened *= hidden_state_dim[k]
max_training_examples = 3000
discount_factor = 1
td_steps = 10
