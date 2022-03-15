# minesweeper_ml
Minesweeper implementation with a Reinforcement Learning Agent

## Where to start?
You want to play Minesweeper? 
Run
```python
python gui.py
```
or train a model using
```python
python training.py
```

If you want to play with an existing model, specify the model_path as a parameter of the agent and disable QTrainer.train_step.

## Files

| Filename | Content |
|----------|:-------------|
| gui.py |  Playable Minesweeper GUI program |
| training.py | Trains a DQN on 100.000 minesweeper games | 
| minesweeper.py | Minesweeper logic |
| agent.py | Implements the reinforcement learning agent which is used during training |
| bruteforce.py | Implements a Minesweeper Bruteforce Algorithm |
| dqn.py | holds the DQN model used by the agent |
| model.py | QTrainer class which trains the DQN model and updates the Q Parameter | 
| utils.py | utility functionality |
| __bruteforce_test.py | internal test cases for the bruteforce algorithm |
