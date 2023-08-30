import numpy as np

class ValueIteration:
    def __init__(self, reward_function, transition_model, gamma) -> None:
        pass

    def get_policy(self, q_table):
        pi = np.ones(12) * -1
        for s in range(0, 12):
            v_list = np.zeros(2)
            for a in ['stand', 'swing']:
                pass

    def value_iteration(self, P):
        q_table = {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0,
            9: 0,
            10: 0 ,
            11: 0 ,
            12: 0 ,
            13: 0 ,
            14: 0 ,
            15: 0 ,
            16: 0 ,
            17: 0 ,
        }
        delta = np.Inf
        epsilon = np.finfo(float).eps
        while delta >= epsilon:
            for i in range(0, 12): # TODO: replace range with len of var
                v = q_table[i]
                swing_value = 0
                stand_value = 0
                for j in range(i, 18): # TODO: replace range with len of var
                    p_stand = P[(P['state']==i) & (P['next_state']==j)].stand
                    p_swing = P[(P['state']==i) & (P['next_state']==j)].swing
                    if not p_stand.empty:
                        reward = self.reward_fn(i,"stand",j)
                        stand_value += p_stand.item()*(reward + q_table[j])
                    if not p_swing.empty:
                        reward = self.reward_fn(i,"swing",j)
                        swing_value += p_swing.item()*(reward + q_table[j])
                q_table[i] = max(swing_value, stand_value)
                delta = min(delta, np.abs(v - q_table[i]))
        print("Converged!")
        print(q_table)
        policy = self.get_policy(q_table)
        return policy