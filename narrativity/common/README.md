# Common

All code that is generally common to all sub-projects.

## Config
The config is a singleton class that has all the configuration information such as locations or paths of different dataset or databases.

## Keyring
The keyring is a singleton class that has keys and secrets information. Ideally, the keys and secrets file itself would be in a safe location and not commited to the repository.

## Experiment Configs List
Consists of `ExperimentConfig` classes. These store information pertinent to one run of the experiment. This is useful to provide mutliple versions of the same pipeline/experiment, if you want to compare performances. One doesn't have to manually run the experiment mutliple times with different options.
