# Structured Noise to Reduce Deep BSDE Training Loss

## Structure

For the experiments detailed in [REPORT], I implemented our new noise sampling schemes within the repos of the papers
referenced. The 2018_Paper directory contains a clone of https://github.com/frankhan91/DeepBSDE where I then made
alterations, similarly the fbsde_paper repo contains a clone of https://github.com/YifanJiang233/Deep_BSDE_solver where
I also made alterations.

## Original 2018 Paper Adaptations and Usage

I added support for running multiple runs for a given model type, you can compare the original configs in
2018_Paper/configs with those in 2018_Paper/experiment_configs.

- Within the eqn_config object you now specify the sampler either, Brownian or Hadamard.
  ```
  "eqn_config":
      "sampler": "Brownian"
    }
  ```

- You can also provide a list of seed and dimension values for the runs, this is so that I could 'queue' up multiple
  experiments in one go and let my laptop run without having to start a new training cycle every 10-20 minutes.
  ```
  "run_config": {
    "seeds": [1, 2, 3],
    "dims": [1, 2, 15, 100]
  }
  ```
  Here the given solver will be initialised and trained 3 times for each dimension resulting in 12 sets of training
  data.

I have also written a bash script that will take a list of configs and run the code passing them in series, further
allowing me to leave my laptop to complete all the experiments automatically, this is located at 2018_Paper/run.sh.

## FBSDE Paper Adaptations and Usage

The code here is almost entirely unchanged, I adapted the solver class to take in a method that returns the noise paths
in its constructor. See fbsde_paper/cir_bond.py and fbsde_paper/multi_cir_bond.py and fbsde_paper/README.md for use.

## Notebooks for plotting

I grouped training data from various experiments into directories and then used plotly to create the plots seen in [].
See notebooks/plots.ipynb for examples.