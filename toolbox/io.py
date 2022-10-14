from six.moves import cPickle as pickle 
import json
from json import JSONEncoder
from xlsx2csv import Xlsx2csv


def pickle_file(file_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(file_, f)
    f.close()


def load_pickle(filename_):
    with open(filename_, 'rb') as f:
        ret_file = pickle.load(f)
    return ret_file


def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, cls=json_encoder)


class json_encoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_json()
        return JSONEncoder.default(self, obj)


def read_excel(path: str, sheet_name=0, index_col=None) -> pd.DataFrame:
    '''
        Turn excel into csv in memory to speed up reading (~50% speedup)
    '''
     # inside here so only is loaded if needed (not all machines can install this package apparently..)
    buffer = StringIO()
    Xlsx2csv(path, outputencoding="utf-8", sheet_name=sheet_name).convert(buffer)
    buffer.seek(0)
    df = pd.read_csv(buffer, index_col=index_col)
    return df
