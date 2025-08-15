import sys,os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from poetry_DB import PoetryDB

class DictionaryRetrieval:
    def __init__(self):
        self.db=PoetryDB()