from pybaseball import statcast_pitcher
from pybaseball import playerid_lookup
import pandas as pd
import numpy as np
import math

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

    def get_markov_chain(self):
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
            if (row['description'] == 'foul' or 
                    row['description'] == 'swinging_strike' or 
                    row['description'] == 'foul_tip' or
                    row['description'] == 'swinging_strike_blocked' or
                    row['description'] == 'foul_bunt' or 
                    row['description'] == 'missed_bunt' or 
                    row['description'] == 'hit_into_play'):
                action_seq.append(1)
            elif (row['description'] == 'called_strike' or 
                    row['description'] == 'ball' or 
                    row['description'] == 'blocked_ball' or
                    row['description'] == 'hit_by_pitch' or
                    row['description'] == 'pitchout'):
                action_seq.append(0)
                if walk_flag:
                    action_seq.append(0)
                    walk_flag = False

        data = {'state': pitch_seq,
                'action': action_seq}
        df = pd.DataFrame.from_dict(data, orient='index')
        df = df.transpose()
        df['next_state'] = df['state'].shift(-1)
        M = np.zeros((12, 18, 2))
        for i, row in df.iterrows():
            if math.isnan(row['state']):
                continue
            current_state = int(row['state'])
            action = int(row['action'])
            if math.isnan(row['next_state']) or current_state >= 12:
                continue
            next_state = int(row['next_state'])
            M[current_state, next_state, action] += 1

        for row in M[:,:,0]:
            n = sum(row)
            if n > 0:
                row[:] = [f/sum(row) for f in row]

        for row in M[:,:,1]:
            n = sum(row)
            if n > 0:
                row[:] = [f/sum(row) for f in row]
        return M

    def reward_fn(self, i, action, j):
        if i in list(range(0,12)):
            # if out
            if j == 12 or j in list(range(0,12)):
                return 0
            elif j == 17 and action == 0: # walk
                return 1
            elif j == 13 and action == 1: # single
                return 2
            elif j == 14 and action == 1: # double
                return 3
            elif j == 15 and action == 1: # triple
                return 4
            elif j == 16 and action == 1: # HR
                return 5
            else:
                return 0
        else:
            return 0
