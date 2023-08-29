from pybaseball import statcast_pitcher
from pybaseball import playerid_lookup
import pandas as pd

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
        id = playerid_lookup(lname,fname).key_mlbam.item()
        self.data = statcast_pitcher(start_dt, end_dt, player_id = id)
