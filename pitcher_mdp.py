from pybaseball import statcast_pitcher
from pybaseball import playerid_lookup
import pandas as pd
import numpy as np

states = {
    (0,0): 0,
    (1,0): 1,
    (2,0): 2,
    (3,0): 3,
    (0,1): 4,
    (0,2): 5,
    (1,1): 6,
    (1,2): 7,
    (2,1): 8,
    (2,2): 9,
    (3,1): 10,
    (3,2): 11,
    "Out": 12, 
    "Single": 13, 
    "Double": 14, 
    "Triple": 15, 
    "HR": 16, 
    "Walk": 17
}

pitch_count_labels = [
    "(0,0)",
    "(1,0)",
    "(2,0)",
    "(3,0)",
    "(0,1)",
    "(0,2)",
    "(1,1)",
    "(1,2)",
    "(2,1)",
    "(2,2)",
    "(3,1)",
    "(3,2)",
    "Out", 
    "Single",
    "Double",
    "Triple",
    "HR",
    "Walk"
]

class PitcherMDP:
    def __init__(self, fname, lname, start_dt, end_dt) -> None:
        self.fname = fname
        self.lname = lname
        self.start_dt = start_dt
        self.end_dt = end_dt
        self.year = int(start_dt[:4])
        id = playerid_lookup(lname,fname).key_mlbam.item() # TODO: handle missing player
        self.data = statcast_pitcher(start_dt, end_dt, player_id = id)

    def get_markov_chain(self, no_actions=False):
        # Data prep
        df = self.data
        df['year'] = pd.DatetimeIndex(df['game_date']).year  # separate year from date
        df['month'] = pd.DatetimeIndex(df['game_date']).month
        df_season = df[(df['year']==self.year) & (df['month'] >= 4)] # TODO: handle different start months
        df_season = df_season.filter(items=['balls',
                                'strikes',
                                'events',
                                'description',
                                'game_date'])
        df_season = df_season.iloc[::-1] # reverse order from earliest to latest
        non_terminal_states = list(zip(df_season.balls, df_season.strikes))
        df_season['non_terminal_states'] = non_terminal_states
        df_season = df_season.reset_index()

        # Retrieve pitch and action transitions sequences
        pitch_seq = []
        action_seq = []
        walk_flag = False
        for i, row in df_season.iterrows():
            event = row['events']
            # no outcome, add non-terminal state
            if str(event) == "nan" or str(event) == "field_error" or str(event) == "fielders_choice":
                pitch_seq.append(states[row['non_terminal_states']])
            # if Out, add terminal state
            elif (str(event) == "strikeout" or 
                    str(event) == "field_out" or 
                    str(event) == "grounded_into_double_play" or 
                    str(event) == "fielders_choice_out" or 
                    str(event) == "sac_fly" or 
                    str(event) == "force_out" or 
                    str(event) == "sac_bunt" or 
                    str(event) == "caught_stealing_2b" or 
                    str(event) == "double_play"):
                pitch_seq.append(states["Out"])
            elif (str(event) == "single"):
                pitch_seq.append(states["Single"])
            elif (str(event) == "double"):
                pitch_seq.append(states["Double"])
            elif (str(event) == "triple"):
                pitch_seq.append(states["Triple"])
            elif (str(event) == "home_run"):
                pitch_seq.append(states["HR"])
            elif (str(event) == "walk"):
                pitch_seq.append(states[row['non_terminal_states']])
                pitch_seq.append(states["Walk"])
                walk_flag = True
            # if swing
            if (row['description'] == 'foul' or 
                row['description'] == 'swinging_strike' or 
                row['description'] == 'foul_tip' or
                row['description'] == 'swinging_strike_blocked' or
                row['description'] == 'foul_bunt' or 
                row['description'] == 'missed_bunt' or 
                row['description'] == 'hit_into_play'):
                action_seq.append("swing")
            # elif stand
            elif (row['description'] == 'called_strike' or 
                row['description'] == 'ball' or 
                row['description'] == 'blocked_ball'):
                action_seq.append("stand")
                if walk_flag:
                    action_seq.append("stand")
                    walk_flag = False
        if no_actions:
            data = {'state': pitch_seq}
            df = pd.DataFrame(data)
            df['next_state'] = df['state'].shift(-1)
            df = df[:-1]
            transition_counts = df.groupby(['state', 'next_state']).size().unstack(fill_value=0)
            M = transition_counts.div(transition_counts.sum(axis=1), axis=0)
        else: 
            data = {'state': pitch_seq,
                    'action': action_seq}
            df = pd.DataFrame(data)
            df['next_state'] = df['state'].shift(-1)
            df['next_label'] = df['action'].shift(-1)
            df = df[:-1]
            transition_counts = df.groupby(['state', 'next_state', 'action']).size().unstack(fill_value=0)
            M = transition_counts.div(transition_counts.sum(axis=1), axis=0).reset_index()
        return M

    def reward_fn(self, i, action, j):
        if i in list(range(0,12)):
            # if out
            if j == 12 or j in list(range(0,12)):
                return 0
            # if walk
            elif j == 17 and action == "stand":
                return 1
            # if single
            elif j == 13 and action == "swing":
                return 2
            elif j == 14 and action == "swing":
                return 3
            elif j == 15 and action == "swing":
                return 4
            elif j == 16 and action == "swing":
                return 5
            else:
                return 0
        else:
            return 0

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
            for i in range(0, 12):
                v = q_table[i]
                swing_value = 0
                stand_value = 0
                for j in range(i, 18):
                    print(f"(i,j): {(i,j)}")
                    p_stand = P[(P['state']==i) & (P['next_state']==j)].stand
                    p_swing = P[(P['state']==i) & (P['next_state']==j)].swing
                    if not p_stand.empty:
                        reward = self.reward_fn(i,"stand",j)
                        print(f"reward when stand: {reward}")
                        stand_value += p_stand.item()*(reward + q_table[j])
                    if not p_swing.empty:
                        reward = self.reward_fn(i,"swing",j)
                        print(f"reward when swing: {reward}")
                        swing_value += p_swing.item()*(reward + q_table[j])
                q_table[i] = max(swing_value, stand_value)
                delta = min(delta, np.abs(v - q_table[i]))
                print(f"J(i): {np.abs(v - q_table[i])}")
                print(f"delta: {delta}")
        print("Converged!")
        print(q_table)