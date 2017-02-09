import pandas as pd
import numpy as np
import datetime
import gzip, pickle
import glob

d = 50
year = 2009
month = '04'

dir_name = '../top{}_1months_start_{}-{}'.format(d, year, month)

L = glob.glob(dir_name + '/*splitted*')
L.sort()

def ix2str(ix):
    if ix < 10:
        ix_str = '00' + str(ix)
    elif ix < 100:
        ix_str = '0' + str(ix)
    else:
        ix_str = str(ix)
    return ix_str

urls_to_keep = ['http://ameblo.jp',
 'http://ar.answers.yahoo.com',
 'http://boston.craigslist.org',
 'http://chicago.craigslist.org',
 'http://citeulike.org',
 'http://de.answers.yahoo.com',
 'http://fr.answers.yahoo.com',
 'http://golivewire.com',
 'http://it.answers.yahoo.com',
 'http://losangeles.craigslist.org',
 'http://miami.craigslist.org',
 'http://mx.answers.yahoo.com',
 'http://news.bbc.co.uk',
 'http://news.com.au',
 'http://newyork.craigslist.org',
 'http://plaza.rakuten.co.jp',
 'http://pr-inside.com',
 'http://rss.feedsportal.com',
 'http://sandiego.craigslist.org',
 'http://seattle.craigslist.org',
 'http://sfbay.craigslist.org',
 'http://slideshare.net',
 'http://sportsnipe.com',
 'http://us.rd.yahoo.com',
 'http://washingtondc.craigslist.org']

top_d = pd.read_csv(dir_name + '/top_50.csv')
url2ix = { url:ix for ix, url in enumerate(top_d['url']) }

tmp = []

for url in urls_to_keep:
    ix = url2ix[url]
    filename = L[ix]
    f = gzip.open(filename, 'r')
    process = pickle.load(f, encoding='latin1')
    f.close()
    tmp.append(process)

for day in range(len(tmp[0])):
    res = []
    for proc in range(len(tmp)):
        res.append(tmp[proc][day])
    f = gzip.open(dir_name+'/copy_'+ix2str(day)+'.pkl.gz', 'wb')
    pickle.dump(res, f, protocol=2)
    f.close()
