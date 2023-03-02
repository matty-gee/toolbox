import pandas as pd
import numpy as np
from six.moves import cPickle 
import json
from io import StringIO


def pickle_file(data, filename):
    with open(filename, 'wb') as f:
        cPickle.dump(data, f)
    f.close()


def load_pickle(filename):
    with open(filename, 'rb') as f:
        data = cPickle.load(f)
    return data


def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, cls=json_encoder)
    f.close()


def load_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data


class json_encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_json()
        return json.JSONEncoder.default(self, obj)


# from xlsx2csv import Xlsx2csv
# def read_excel(path: str, sheet_name=0, index_col=None) -> pd.DataFrame:
#     '''
#         Turn excel into csv in memory to speed up reading (~50% speedup)
#     '''
#      # inside here so only is loaded if needed (not all machines can install this package apparently..)
#     buffer = StringIO()
#     Xlsx2csv(path, outputencoding="utf-8", sheet_name=sheet_name).convert(buffer)
#     buffer.seek(0)
#     df = pd.read_csv(buffer, index_col=index_col)
#     return df
