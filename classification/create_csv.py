
import pandas as pd
from sklearn.model_selection import train_test_split
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from poetry_DB import PoetryDB


db=PoetryDB()
res=db.get_all_kik_song()
print(res)
for entry in res[:1]:
    print(entry)