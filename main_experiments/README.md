# How to run all experiments

1. Install the dependencies with `pip install requirements.txt`
2. Generate baselines for the offline experiments with `python3 get_baselines.py`
3. Run all experiments with `python3 tmux-runs.py`
4. To test: python3 main.py --env="acrobot" --agent="rainbow" --initial_seed="1" --exp="clip_rewards" --type="online"
