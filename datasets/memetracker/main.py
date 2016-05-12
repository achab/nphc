import os

d = 100

list_df = ['df_2008-08.csv']
list_df.sort()

dir_name = "d" + str(d) + "_" + str(len(list_df)) + "month_start" + list_df[0][3:-4]

if not os.path.isdir(dir_name):
    os.mkdir(dir_name)

