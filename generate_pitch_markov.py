import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})
import pandas as pd
pd.set_option('display.max_columns', None)


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

def get_pitch_sequence(df_game):
    pitch_seq = []
    action_seq = []
    walk_flag = False
    for i, row in df_game.iterrows():
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
    return pitch_seq, action_seq

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

def get_markov_chain(data, simple=False):
    pitch_seq, action_seq = get_pitch_sequence(data) # TODO: utilized action_seq
    if simple:
        data = {'state': pitch_seq}
        df = pd.DataFrame(data)
        df['next_state'] = df['state'].shift(-1)
        df = df[:-1]
        transition_counts = df.groupby(['state', 'next_state']).size().unstack(fill_value=0)
        transition_probabilities = transition_counts.div(transition_counts.sum(axis=1), axis=0)
    else: 
        data = {'state': pitch_seq,
                'action': action_seq}
        df = pd.DataFrame(data)
        df['next_state'] = df['state'].shift(-1)
        df['next_label'] = df['action'].shift(-1)
        df = df[:-1]
        transition_counts = df.groupby(['state', 'next_state', 'action']).size().unstack(fill_value=0)
        transition_probabilities = transition_counts.div(transition_counts.sum(axis=1), axis=0)
    return transition_probabilities

def main(args):
    # import data and select season
    df = pd.read_csv(args.pitcher_file)
    df['year'] = pd.DatetimeIndex(df['game_date']).year  # separate year from date
    df['month'] = pd.DatetimeIndex(df['game_date']).month
    df_season = df[(df['year']==args.year) & (df['month'] >= 4)]

    # Data Prep by selecting certain columns
    df_season = df_season.filter(items=['balls',
                             'strikes',
                             'events',
                             'description',
                             'game_date'])
    df_season = df_season.iloc[::-1] # reverse order from earliest to latest
    non_terminal_states = list(zip(df_season.balls, df_season.strikes))
    df_season['non_terminal_states'] = non_terminal_states
    df_season = df_season.reset_index()

    M = get_markov_chain(df_season, simple=True)

    mask = np.tril(np.ones_like(M))
    h = sns.heatmap(M, 
                cmap='viridis', 
                cbar=False,
                fmt='.2f',
                mask=mask,
                annot=True,
                xticklabels=pitch_count_labels,
                yticklabels=pitch_count_labels).set_title(f"Pitch Count Sequence Transition Probability for {args.pitcher_file[5:-4]} in {args.year}")
    plt.show()
    if args.save:
        fig = h.get_figure()
        fig.savefig(f'figures/{args.pitcher_file[5:-4]}_markov_matrix.pdf')
        # save transition probability matrix with batter actions as csv for Value Iteration
        M = get_markov_chain(df_season, simple=False)
        M.to_csv(f"markov_chains/{args.pitcher_file[5:-4]}_{args.year}_MC.csv")

if __name__=="__main__":
    parser = argparse.ArgumentParser(
                    prog='PitcherToCSV',
                    description='Returns pitch data for an individual pitcher as a csv')
    parser.add_argument('-f', '--pitcher_file', required=True)      # option that takes a value
    parser.add_argument('-y', '--year', required=True, type=int)
    parser.add_argument('-s', '--save', action="store_true")
    main(parser.parse_args())