import pandas as pd
import glob

names = glob.glob('counts*')

for filename in names:
    counts = pd.read_csv(filename)
