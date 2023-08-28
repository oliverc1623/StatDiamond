# StatDiamond
AI-Driven Insights into Baseball Performance, Business, and Analytics 

---

## Reinforcement Learning in Baseball

The math that serves as the foundation for reinforcement learning is a Markov Decision Process (MDP). I am interested in representing pitching behavior by a MDP. I believe Value Iteration can help batters strategically counter a pitcher's pitch sequence. In order to compute the Value Iteration algorithm, I need a probability transition matrix. 

### Generate Transition Probability

`generate_pitch_markov.py` is a Python script that will plot the transition probabilities among 12 pitch states, 
(0,0)
(1,0)
(2,0)
(3,0)
(0,1)
(0,2)
(1,1)
(1,2)
(2,1)
(2,2)
(3,1)
(3,2)

To run `generate_pitch_markov.py` specify a pitcher's csv file, year, and save boolean. 
```
python generate_pitch_markov.py -f data/kershaw.csv -y 2017 -s
```

### Download pitcher data

Download any MLB pitcher's data by executing the script, `pitcher_to_csv.py`. For example,
```
python pitcher_to_csv.py --first_name Zac --last_name Gallen --start_dt 2022-04-16 --end_dt 2022-10-04
```

---