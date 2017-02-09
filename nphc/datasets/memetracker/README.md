# Instructions to get a multivariate point process from raw data

Download Raw MemeTracker phrase data from: http://www.memetracker.org/data.html

Run
```
python raw2df.py
```

You have now a dataframe that encodes link between sites for each month.

Now, run
```
python main.py
```

to obtain a *d*-dimensional multivariate point process, *d* being the most cited sites you want to keep. 
