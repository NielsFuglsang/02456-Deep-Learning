# 02456 Deep Learning project
This repository contains code to train baseline ppo agent in Procgen implemented with Pytorch for the course 02456 Deep Learning at The Technical University of Denmark, Fall semester 2020.

# Folder structure
```
├── src
│   └── init.py
│   └── encoder.py         # Classes for each encoder structure.
│   └── experiment.py      # Class for training and evaluating a policy.
│   └── policy.py          # PPO and TRPO classes.
│   └── utils.py           # Functions for keeping track of environments and data.
├── params
│   └── ...                # JSON files specifying experiment hyperparameters.
├── .gitignore
├── README.md
├── job.sh                 # Jobscript for running on HPC.
├── requirements.txt       # Python modules.
├── run_experiment.py      # Read parameters from JSON file, train, and evaluate policy.
├── sender.sh              # Helper function to submit jobs on HPC.
```