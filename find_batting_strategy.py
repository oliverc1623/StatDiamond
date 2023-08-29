import argparse
from pitcher_mdp import PitcherMDP

def main(args):
    pitcher_mdp = PitcherMDP(args.first_name, args.last_name, args.start_dt, args.end_dt)
    print(pitcher_mdp.data.head(50))


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