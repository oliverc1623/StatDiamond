import argparse
from pybaseball import statcast_pitcher
from pybaseball import playerid_lookup
import pandas as pd
pd.set_option('display.max_columns', None)
from IPython.display import display
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
    
    # compute list of pitch transitions
    transitions = []
    for i, row in df_simple.iterrows():
        event = row['events']
        if str(event) == "nan":
            transitions.append(states[row['non_terminal_states']])

    """
    Taken from https://stackoverflow.com/questions/46657221/generating-markov-transition-matrix-in-python
    """
    def transition_matrix(transitions):
        n = 1+ max(transitions) #number of states

        M = [[0]*n for _ in range(n)]

        for (i,j) in zip(transitions,transitions[1:]):
            M[i][j] += 1

        #now convert to probabilities:
        for row in M:
            s = sum(row)
            if s > 0:
                row[:] = [f/s for f in row]
        return M

    m = transition_matrix(transitions)
    for row in m: print(' '.join('{0:.2f}'.format(x) for x in row))


if __name__=="__main__":
    parser = argparse.ArgumentParser(
                    prog='PitcherToCSV',
                    description='Returns pitch data for an individual pitcher as a csv')
    parser.add_argument('-f', '--pitcher_file', required=True)      # option that takes a value
    parser.add_argument('-y', '--year', required=True, type=int)
    main(parser.parse_args())