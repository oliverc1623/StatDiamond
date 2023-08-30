# StatDiamond
AI-Driven Insights into Baseball Performance, Business, and Analytics 

---

## Reinforcement Learning in Baseball

The math that serves as the foundation for reinforcement learning is a Markov Decision Process (MDP). I am interested in representing pitching behavior by a MDP. I believe Value Iteration can help batters strategically counter a pitcher's pitch sequence. In order to compute the Value Iteration algorithm, I need a probability transition matrix. 

### Generate Transition Probability

`pitcher_mdp.py` is a Python script that creates a MDP class. This class is equipped with various methods for MDP related matter to conduct reinforcement learning. The method `get_markov_chain()` will generate a transition probability matrix (markov chain) among 12 pitch states, 
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

### Value Iteration

Once you've generated a markov chain of the pitching states, you can pass it into the `value_iteration()` function. Value iteration will find the best sequence of actions (swing or stand) to maximize the expectation at each state. 

### Usage

Since `pitcher_mdp.py` is a Python class, we can instantiate it in another python file. For example, `find_batting_strategy.py` instantiates a `pitcher_mpd` object and generates a markov chain and calls the value iteration method. 

If you want to visualize the markov chain (without actions) call the `plot_markov_chain()` function. 

`pitcher_mdp.py` should take in most MLB pitchers. I use the [pybaseball](https://github.com/jldbc/pybaseball#pybaseball) package to acquire pitching data for an individual player. In order to correctly get data from pybaseball follow the parser arguments in the `find_batting_strategy.py` file. 

For example,
```
python find_batting_strategy.py \
        --first_name Zac \
        --last_name Gallen \
        --start_dt 2022-04-16 \
        --end_dt 2022-10-04
```

---