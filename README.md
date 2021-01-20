SPADE: A Semi-supervised Probabilistic Approach for Detecting Errors in Tables
-----------------------------------------------------------------------------
## Setting Up
```
pip install -r requirements.txt
```
## Running Experiments

### General command: 
```
error_detection.py evaluate [OPTIONS]
```
```
Options:
  -d, --data_path TEXT    Path to dataset
  -c, --config_path TEXT  Path to configuration file (default: config/)
  -o, --output_path TEXT  Path to output directory (default: output/)
  -m, --method TEXT       Method for outlier detection
  -i, --interactive       Interactive detection
  --num_gpus INTEGER      Number of GPUs used
  -k INTEGER              Number of iterations
  -e INTEGER              Number of examples per iteration
  --help                  Show this message and exit.
```

### Configuration
Create a config.yml file inside the directory `[config_path]/[method]`.

Example configuration file:
```
psl_config_path: psl/config # path to psl models
psl_data_path: psl/data # path to psl data file
bigram_path: data/train/bigram_count.csv # path to Web table analysis file (included in data/)
combine_method: psl
propagate_level: 1 # (epsilon values. see ReadMe for actual values)
```

### Datasets
Datasets can be downloaded here: 

### Commands to replicate result tables in the paper:
1. Table 3, Figure 3, Figure 4: 20 examples = 4 iterations * 5 examples/iteration
```
PYTHONPATH=.:$PYTHONPATH python kbclean/experiments/error_detection.py evaluate --data_path [data_path] --method lstm  -k 4 -e 5
```

2. Figure 2: 30 examples = 4 iterations * 5 examples/iteration
```
PYTHONPATH=.:$PYTHONPATH python kbclean/experiments/error_detection.py evaluate --data_path [data_path] --method lstm  -i -k 6 -e 5
```

3. Table 5
```
PYTHONPATH=.:$PYTHONPATH python kbclean/experiments/error_detection.py evaluate --data_path [data_path] --method lstm  -k 4 -e 5
PYTHONPATH=.:$PYTHONPATH python kbclean/experiments/error_detection.py evaluate --data_path [data_path] --method random_forest  -k 4 -e 5
PYTHONPATH=.:$PYTHONPATH python kbclean/experiments/error_detection.py evaluate --data_path [data_path] --method xgb  -k 4 -e 5
```

4. Table 6
```
PYTHONPATH=.:$PYTHONPATH python kbclean/experiments/error_detection.py evaluate --data_path [data_path] --method lstm  -k 4 -e 5
PYTHONPATH=.:$PYTHONPATH python kbclean/experiments/error_detection.py evaluate --data_path [data_path] --method nosyntactic  -k 4 -e 5
PYTHONPATH=.:$PYTHONPATH python kbclean/experiments/error_detection.py evaluate --data_path [data_path] --method nosemantic  -k 4 -e 5
```

5. Table 7

Change `propagate_level` argument in `config.yml`: 1 (epsilon = 0.01), 2 (epsilon = 0.001), 3 (epsilon = 0.005), 4 (epsilon = 0.02) before running
```
PYTHONPATH=.:$PYTHONPATH python kbclean/experiments/error_detection.py evaluate --data_path [data_path] --method lstm  -k 4 -e 5
```

