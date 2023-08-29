import argparse
from pitcher_mdp import PitcherMDP
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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

def main(args):
    pitcher_mdp = PitcherMDP(args.first_name, args.last_name, args.start_dt, args.end_dt)
    M = pitcher_mdp.get_markov_chain(no_actions=True)
    print(M)

    mask = np.tril(np.ones_like(M))
    h = sns.heatmap(M, 
                cmap='viridis', 
                cbar=False,
                fmt='.2f',
                mask=mask,
                annot=True,
                xticklabels=pitch_count_labels,
                yticklabels=pitch_count_labels).set_title(f"Pitch Count Sequence Transition Probability for {args.first_name} {args.last_name} in {args.start_dt[:4]}")
    plt.show()


if __name__=="__main__":
    parser = argparse.ArgumentParser(
        prog="Find Batting Strategy",
        description="Given a pitch sequence markov chain for a pitcher, find the best batting strategy"
    )
    parser.add_argument('-f', '--first_name', required=True)      # option that takes a value
    parser.add_argument('-l', '--last_name', required=True)
    parser.add_argument('-s', '--start_dt', required=True, help='YYYY-MM-DD')
    parser.add_argument('-e', '--end_dt', required=True, help='YYYY-MM-DD')
    main(parser.parse_args())