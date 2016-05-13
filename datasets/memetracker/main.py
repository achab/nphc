import os

#############################
### Define the parameters ###
#############################
#
# d: the number of most cited sites you want to keep
#
# list_df: the list of months you want to keep
#


d = 10

list_df = ['df_2008-08.csv']
list_df.sort()

dir_name = "top" + str(d) + "_" + str(len(list_df)) + "months_start" + list_df[0][3:-4]

if not os.path.isdir(dir_name):
    os.mkdir(dir_name)

# find top d most cited urls
import count_top
count_top.main(list_df,d,dir_name)

# create pp for top d sites previously selected
start_month = list_df[0][3:-4]
start = pd.to_datetime(start_month + '-01 00:00:00')
import create_pp
create_pp.main()