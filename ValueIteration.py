import numpy as np

class ValueIteration:
    def __init__(self, reward_function, transition_model, gamma) -> None:
        self.num_states = transition_model.shape[0]
        self.num_actions = transition_model.shape[2]
        self.reward_function = np.nan_to_num(reward_function)
        self.transition_model = transition_model
        self.gamma = gamma
        self.values = np.zeros(self.transition_model.shape[1])
        self.policy = None

    def get_policy(self, q_table):
        pi = np.ones(12) * -1
        for s in range(0, 12):
            v_list = np.zeros(2)
            for a in ['stand', 'swing']:
                pass

    def one_iteration(self):
        delta = 0
        for s in range(self.num_states):
            temp = self.values[s]
            v_list = np.zeros(self.num_actions)
            for a in range(self.num_actions):
                new_value = 0.0
                for j in range(s, self.transition_model.shape[1]):
                    p = self.transition_model[s,j,a]
                    reward = self.reward_function(s, a, j)
                    new_value += p*(reward + self.values[j])
                v_list[a] = new_value
            self.values[s] = max(v_list)
            delta = max(delta, abs(temp - self.values[s]))
        return delta

    def train(self):
        epoch = 0
        delta = np.Inf
        epsilon = np.finfo(float).eps
        while delta >= epsilon:
            epoch += 1
            delta = self.one_iteration()
        print(f"Converged in {epoch} epochs!")
        print(self.values)
        # policy = self.get_policy(q_table)
        return None