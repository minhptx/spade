# ACT

B-Clean is a library built to support `Bring Your Own Data (BYOD)` project. B-Clean provides different functionalities to detect outliers in tabular datasets and suggest possible transformations to clean the data.

## Roadmap
- [x] Statistical outlier detection
- [ ] Few-shot outlier detection
    - [x] Baseline model ([HoloDetect](https://arxiv.org/pdf/1904.02285.pdf))
    - [ ] Data-driven model (LSTM)
    - [ ] Improve performance
    - [ ] Decrease number of examples
- [ ] Active learning outlier detection
    - [ ] Automatic suggestion based on statistical model
    - [ ] Policy-based active learning model
- [ ] Data transformation


Outlier Detection
-----------------


### Background
We define and detect three different types of outliers as follows:
* Global outliers: values that rarely appear in the real-world data. 
* Local outliers: values that are different from other values in the same attribute. 
* Null outliers: values that have no meaning

### Usage
1. Install and activate  conda environment 
```
conda env create -f environment.yml
conda activate byod
```
2. Run command
```
PYTHONPATH=.:$PYTHONPATH python -m kbclean.main detect --i [input_file] -e [example_file] -o [output_file] --num_gpus [number_of_gpus] 
```

More details of command can be found by running
```
PYTHONPATH=.:$PYTHONPATH python -m kbclean.main detect --help
```

3. Example files can be found in [demo](demo) folder. 

- Input file: csv format
- Example file: json format with the following template
    ```json
    {
    "col1": [
        {
            "raw": "val1",
            "cleaned": "val2"
        },
        {
            "raw": "val3",
            "cleaned": "val4"
        }
    ],
    "col2": [
        {
            "raw": "val1",
            "cleaned": "val2"
        },
        {
            "raw": "val3",
            "cleaned": "val4"
        }
    ]
    }
    ```
- Output file: csv file with outliers annotated as <<<valuez>>>
