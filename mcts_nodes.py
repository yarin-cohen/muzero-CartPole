from utils import*
import numpy as np
from global_params import*
from tqdm import tqdm, trange



class Node:
    """ a tree node for MCTS """

    # metadata:
    parent = None  # parent Node
    value_sum = 0.  # sum of state values from all visits (numerator)
    times_visited = 0  # counter of visits (denominator)
    network_value_sum = 0  # W in the article
    min_tree_value = 1000000000
    max_tree_value = -1000000000

    def __init__(self, parent, action, all_models):
        """
        Creates and empty node with no children.
        Does so by commiting an action and recording outcome.

        :param parent: parent Node
        :param action: action to commit from parent Node

        """

        self.parent = parent
        self.action = action
        self.children = set()  # set of child nodes
        # get action outcome and save it
        #res = env.get_result(parent.hidden_state, action)
        #self.snapshot, self.observation, self.immediate_reward, self.is_done, _ = res
        self.all_models = all_models
        #state_pics = get_pics(self, env)
        #self.state = format_pics(state_pics)

        if self.parent is not None:
            self.prob = self.parent.prior_probs[action]
        else:
            self.prob = 1

        self.prior_probs = None
        self.action_value = 0  # Q in the article
        new_state, new_reward = self.get_hidden_state()
        self.hidden_state = new_state
        self.immediate_reward = new_reward[0]

    def get_hidden_state(self):
        g = self.all_models['reg_g']
        action_plane = self.action/n_actions * np.ones((1, ) + hidden_state_dim[:-1] + (1, ))  # normalizing actions
        hidden_state_plane = self.parent.hidden_state  # dims of hidden state was already expanded in axis=0
        new_state, new_reward = g.predict([hidden_state_plane, action_plane])
        return new_state, new_reward

    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        return self.parent is None

    def get_mean_value(self):
        return self.value_sum / self.times_visited if self.times_visited != 0 else 0

    def ucb_score(self, scale=1 / np.sqrt(2), max_value=1e100):
        """
        Computes ucb1 upper bound using current value and visit counts for node and it's parent.

        :param scale: Multiplies upper bound by that. From hoeffding inequality, assumes reward range to be [0,scale].
        :param max_value: a value that represents infinity (for unvisited nodes)

        """

        if self.times_visited == 0:
            return max_value

        # compute ucb-1 additive component (to be added to mean value)
        # hint: you can use self.parent.times_visited for N times node was considered,
        # and self.times_visited for n times it was visited

        if self.parent is not None:
            parent_visits = self.parent.times_visited
        else:
            parent_visits = 0
        U = np.sqrt(2 * np.log(self.parent.times_visited) / self.times_visited)

        return self.get_mean_value() + scale * U

    def poly_upper_conf_score(self, c_puct=10):  # selecting by the PUCT algorithm

        # U = c_puct * self.prob * np.sqrt(np.sum([child.times_visited for child in self.children])) / (
        #             1 + self.times_visited)

        if self.parent is None:
            U = 0
        else:
            child_visits = [child.times_visited for child in self.parent.children]
            #U = c_puct * self.prob * np.sqrt(np.sum(child_visits)) / (1 + self.times_visited)
            U = self.prob * np.sqrt(np.sum(child_visits)) / (1 + self.times_visited) * (
                        c_puct1 + np.log((np.sum(child_visits) + c_puct2 + 1) / c_puct2))

        if Node.max_tree_value != Node.min_tree_value:
            normalized_action_value = (self.action_value - Node.min_tree_value) / (
                        Node.max_tree_value - Node.min_tree_value)
        else:
            normalized_action_value = self.action_value

        return normalized_action_value + U

    # MCTS steps

    def select_best_leaf(self):
        """
        Picks the leaf with highest priority to expand
        Does so by recursively picking nodes with best UCB-1 score until it reaches the leaf.

        """
        if self.is_leaf():
            return self

        # best_child = <YOUR CODE: select best child node in terms of node.ucb_score()>
        children = list(self.children)
        children_puct = [node.poly_upper_conf_score() for node in children]
        ii = np.argmax(children_puct)
        best_child = children[ii]  # <select best child node in terms of node.ucb_score()>

        return best_child.select_best_leaf()

    def expand(self):
        """
        Expands the current node by creating all possible child nodes.
        Then returns one of those children.

        input:
            sess                - tensorflow v1 session
            policy_output_layer - a tensor of the policy output layer
            value_output_layer  - a tensorf of the value output layer
            state_t             - a placeholder of the current None state

        all inputs are used for running the current session with the current network weights
        """

        f = self.all_models['reg_f']
        prior_probs, network_value_sum = f.predict(self.hidden_state) #hidden state is already expanded dims in axis=0
        self.prior_probs = prior_probs[0]  # remove expand_dims
        self.network_value_sum = network_value_sum[0]  # remove expand_dims

        self.prior_probs = (1 - dir_noise_weight) * self.prior_probs + dir_noise_weight * np.random.dirichlet(
            alpha=dirichlet_alpha)

        # when we expand we automatically need to update visit count, since the propagation will be from the parent of
        # the expanded node and up.
        self.times_visited += 1
        # in theory, the expansion should init Q to be 0, and updated only on propagate. here we update the first action
        # value in the expansion, and carry the rest on propagate. in this calc G_k is zero (paper, article (3) page 12)

        self.action_value = ((self.times_visited - 1) * self.network_value_sum[0]) / self.times_visited
        if Node.min_tree_value > self.action_value:
            Node.min_tree_value = self.action_value
        if Node.max_tree_value < self.action_value:
            Node.max_tree_value = self.action_value

        for action in range(n_actions):
            self.children.add(Node(self, action, self.all_models))

        return self.select_best_leaf()

    def propagate(self, child_network_return, child_network_reward):
        """
        Uses child value (sum of rewards) to update parents recursively.
        """
        # compute node value - Q, like article (3) page 12 in muzero paper
        G_k = discount_factor * child_network_return + child_network_reward
        self.network_value_sum = G_k #is this true? page 12 article (3)
        self.times_visited += 1
        self.action_value = ((self.times_visited - 1)*self.action_value + G_k)/self.times_visited

        if Node.min_tree_value > self.action_value:
            Node.min_tree_value = self.action_value
        if Node.max_tree_value < self.action_value:
            Node.max_tree_value = self.action_value

        # propagate upwards
        if not self.is_root():
            self.parent.propagate(G_k, self.immediate_reward)  # TODO: Figure out if child_value or network value sum

    def safe_delete(self):
        """safe delete to prevent memory leak in some python versions"""
        del self.parent
        for child in self.children:
            child.safe_delete()
            del child

    def pop_pic(self):
        if len(self.children) == 0:
            self.env.pics.pop(self.snapshot)
        else:
            children = self.children
            self.env.pics.pop(self.snapshot)
            for child in children:
                child.pop_pic()
                del child.state

    def find_root(self):
        if self.parent is None:
            if hasattr(self, 'state'):
                del self.state
            return self
        else:
            if hasattr(self, 'state'):
                del self.state
            for child in self.children:
                if hasattr(child, 'state'):
                    del child.state
            return self.parent.find_root()


class Root(Node):
    def __init__(self, observation, all_models):
        """
        creates special node that acts like tree root
        :hidden_state: snapshot (from h) to start planning from
        :observation: last environment observation
        """

        self.parent = self.action = None
        self.children = set()  # set of child nodes

        self.value_sum = 0.  # sum of state values from all visits (numerator)
        self.times_visited = 0  # counter of visits (denominator)
        self.network_value_sum = 0
        # root: load snapshot and observation

        #self.hidden_state = snapshot
        self.observation = observation
        h = all_models['h']
        hidden_state = h.predict(self.observation) #dims of observation already expanded for predict in format pic func
        self.hidden_state = hidden_state
        self.all_models = all_models
        self.immediate_reward = 0
        #state_pics = get_pics(self, env)
        # print(state_pics.shape)
        #self.state = format_pics(state_pics)
        #self.env = env

    @staticmethod
    def from_node(node, observation):
        """initializes node as root"""
        root = Root(observation, node.all_models)
        #starting tree FROM SCRATCH every time we get a new observation
        copied_fields = []
        for field in copied_fields:
            setattr(root, field, getattr(node, field))

        return root

    def play_move(self, gen_num):
        # if num_move < 30:
        #     temp = move_temp_initial
        # else:
        #     temp = move_temp
        if gen_num < training_steps_temps[0]:
            temp = move_temps[0]
        elif gen_num < training_steps_temps[1]:
            temp = move_temps[1]
        elif gen_num < training_steps_temps[2]:
            temp = move_temps[2]
        else:
            temp = move_temps[3]
        p = np.zeros((n_actions))
        # count = 0
        total_n = np.sum([child.times_visited ** (1 / temp) for child in self.children])
        c_list = {}
        c_val_list = {}
        if total_n == 0:
            print('default action')  # if we got here it means that every child has is_done marked as true!!!
            cc = list(self.children)
            return cc[0].action, cc[0], p, cc[0].action_value  # p and ac_val are nonesense but it doesn't matter since the child we're choosing is is_done

        for child in self.children:
            p[child.action] = (child.times_visited ** (1 / temp)) / total_n
            c_list[child.action] = child
            c_val_list[child.action] = child.action_value

        move = np.random.choice(np.arange(n_actions), p=p)
        return move, c_list[move], p, c_val_list[move]


def plan_mcts(root, n_iters=1600):
    """
    builds tree with monte-carlo tree search for n_iters iterations
    :param root: tree node to plan from
    :param sess: current tensorflow session
    :param policy_output_layer: tensor for policy vector
    :param value_output_layer: tensor for the value of a specific state
    :param state_t: placeholder for the current state
    :param n_iters: how many select-expand-simulate-propagete loops to make
    """
    for _ in range(n_iters):
        node = root.select_best_leaf()

        leaf = node.expand()
        g_return = node.network_value_sum
        r_reward = node.immediate_reward
        if node.parent is not None:
            node.parent.propagate(g_return, r_reward)