from pathlib import Path

import yaml
from test_tube import HyperOptArgumentParser


def load_hparams(config_path):
    config_path = Path(config_path)
    parser = HyperOptArgumentParser()
    hparams = yaml.load(config_path.open("r").read(), Loader=yaml.FullLoader)
    parser.add_argument("-f", required=False)  # for Jupyter Notebook
    for key, value in hparams.items():
        parser.add_argument(f"--{key}", required=False, default=value)
    return parser.parse_args()
