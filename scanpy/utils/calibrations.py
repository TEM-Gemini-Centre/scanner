import json as js
from pathlib import Path
def read_calibration(filename):
    filename = Path(filename)
    if filename.is_file():
        if filename.suffix == '.json':
            with filename.open() as f:
                js.load(f)
        else:
            raise ValueError(f'Calibration file "{filename}" is not a JSON file')
    else:
        raise ValueError(f'Calibration file "{filename}" is not a file.')
