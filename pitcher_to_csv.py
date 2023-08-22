# A simple script to retrieve pitch data in a CSV format

import argparse
from pybaseball import statcast_pitcher
from pybaseball import playerid_lookup
import pandas as pd

def main(args):
    id = playerid_lookup(args.last_name,args.first_name).key_mlbam.item()
    data = statcast_pitcher(args.start_dt, args.end_dt, player_id = id)
    output_fname = f"data/{args.last_name}_{args.first_name}.csv"
    data.to_csv(output_fname, index=False)


if __name__=="__main__":
    parser = argparse.ArgumentParser(
                    prog='PitcherToCSV',
                    description='Returns pitch data for an individual pitcher as a csv')
    parser.add_argument('-f', '--first_name', required=True)      # option that takes a value
    parser.add_argument('-l', '--last_name', required=True)
    parser.add_argument('-s', '--start_dt', required=True, help='YYYY-MM-DD')
    parser.add_argument('-e', '--end_dt', required=True, help='YYYY-MM-DD')
    main(parser.parse_args())