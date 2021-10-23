import pathlib
import environs

env = environs.Env()
env.read_env()

_data_path = pathlib.Path(env.str('DATA_PATH'))

RAW_DATA_PATH = _data_path.joinpath('raw')
PREPARED_DATA_PATH = _data_path.joinpath('prepared')
MODELS_DATA_PATH = _data_path.joinpath('models')
DATA_PATH = _data_path.joinpath('data')
