# MuZero - Cartpole
This repository is a python implementation of
[DeepMind's "MuZero" reinforcement learning algorithm](https://arxiv.org/pdf/1911.08265.pdf). This solution was
implemented for Openai's CartPole environment. This implementation can be used to solve other environments, provided you
changed the implementation of proper models.

# Overview:
* Using Tensorflow v2 but with v2 behavior disabled
* Using openai's gym
* Using cv2

# Implementation details:
## Model
* The file `models.py` contains every definition and implementation of the models required.
* Implements the 3 core models from the paper:
    * The hidden state model - h. s_0 = h(o_1, ..., o_t)
    * The dynamics model - g.  r_k, s_k = g(s_km1, a_k)
    * The prediction model - f. p_k, v_k = f(s_k)
* A "Mu model" is built and  defined by using the 3 models recurrently. The model is built using a dictionary
containing all 3 models.

## MCTS
* The file `mcts_nodes.py` implements a class of Monte Carlo search tree nodes
* Defining and initializing a node requires all 3 core models as inputs
* This class implements every core functionality that is detailed in the original paper.

## Playing and Training
* The file `train_and_play.py` implements the entire loop of playing the game and training the model until conversion.
* Creating the correct models, building a Mu model, and starting to train for X generations. 
* Every generation:
  * Loads the latest weight file for all models
  * Plays the game for X iterations
  * Collects and saves proper data
  * Trains on data - Experience Replay.
  * Saves training plots.
* It's important to note that both values and rewards as outputs from the relevant models are encoded and normalized
exactly as detailed in the original paper.

# How To Use:

* Open `global_params.py`. Enter paths to save relevant training data and plots.
* Change any other desired parameters.
* Run `train_and_play.py`.

This implementation can work with other environments, with the proper adjustments:
* Change the environment in `play_game()` under `train_and_play.py`.
* Write a relevant implementation for all 3 core models.
* Change relevant parameters under `global_params.py` to adjust new state and hidden state dimensions.
