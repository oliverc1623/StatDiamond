import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
pd.set_option('display.max_columns', None)
from IPython.display import display

from pybaseball import statcast_pitcher
from pybaseball import playerid_lookup

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

def get_pitch_sequence(df_game):
    pitch_seq = []
    for i, row in df_game.iterrows():
        event = row['events']
        if str(event) == "nan":
            pitch_seq.append(states[row['non_terminal_states']])
    return pitch_seq

def count_transitions(pitch_seq):
    n = 1+ max(pitch_seq) #number of states
    M = [[0]*n for _ in range(n)]
    for (i,j) in zip(pitch_seq,pitch_seq[1:]):
        M[i][j] += 1
    return np.array(M)

def get_probabilities(m):
    for row in m:
        s = sum(row)
        if s > 0:
            row[:] = [f/s for f in row]
    return m

def main(args):
    # import data and select season
    df = pd.read_csv(args.pitcher_file)
    df['year'] = pd.DatetimeIndex(df['game_date']).year  # separate year from date
    df['month'] = pd.DatetimeIndex(df['game_date']).month
    df_season = df[(df['year']==args.year) & (df['month'] >= 4)]

    # select certain columns
    df_simple = df_season.filter(items=['balls',
                             'strikes',
                             'events',
                             'description',
                             'game_date'])
    df_simple = df_simple.iloc[::-1] # reverse order from earliest to latest
    non_terminal_states = list(zip(df_simple.balls, df_simple.strikes))
    df_simple['non_terminal_states'] = non_terminal_states
    df_simple = df_simple.reset_index()

    # Make a dictionary for each game
    unique_games = df_simple.game_date.unique()
    DataFrameDict = {elem : pd.DataFrame() for elem in unique_games}

    for key in DataFrameDict.keys():
        DataFrameDict[key] = df_simple[:][df_simple.game_date == key]

    all_transition_matrices = []
    for game_date, game_df in DataFrameDict.items():
        pitch_sequence = get_pitch_sequence(game_df)
        m = count_transitions(pitch_sequence)
        pad_len = 12 - len(m)
        m = np.pad(m, (0, pad_len), mode="constant")
        all_transition_matrices.append(m)
    all_transition_matrices = np.array(all_transition_matrices)
    M = sum(all_transition_matrices)
    M = get_probabilities(M.tolist())
    # for row in M: print(' '.join('{0:.2f}'.format(x) for x in row))
    M = np.triu(np.array(M))

    M = pd.DataFrame(M)
    M = M.rename(columns={0:"(0,0)",
                    1:"(1,0)",
                    2:"(2,0)",
                    3:"(3,0)",
                    4:"(0,1)",
                    5:"(0,2)",
                    6:"(1,1)",
                    7:"(1,2)",
                    8:"(2,1)",
                    9:"(2,2)",
                    10:"(3,1)",
                    11:"(3,2)"})
    h = sns.heatmap(M, xticklabels=M.columns, yticklabels=M.columns)
    if args.save:
        fig = h.get_figure()
        fig.savefig(f'figures/{args.pitcher_file[5:-4]}_markov_matrix.pdf')

if __name__=="__main__":
    parser = argparse.ArgumentParser(
                    prog='PitcherToCSV',
                    description='Returns pitch data for an individual pitcher as a csv')
    parser.add_argument('-f', '--pitcher_file', required=True)      # option that takes a value
    parser.add_argument('-y', '--year', required=True, type=int)
    parser.add_argument('-s', '--save', action="store_true")
    main(parser.parse_args())