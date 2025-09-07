
import pandas as pd
import numpy as np
import random 
#import tensorflow as tf 


class FF_Classifier():
    
    def __init__(self,split_ratio=0.7,random_seed=47,CSV_PATH="classification/kafe_kniga_songs.csv"):
        self.rarandom_seed=random_seed=47
        np.random.seed(random_seed)
        random.seed(random_seed)
        #tf.random.set_seed(random_seed)
        self.database=pd.read_csv(CSV_PATH)
        print(self.database)
        self.database.drop()



test=FF_Classifier()
