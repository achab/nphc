try:
    import cPickle as pickle
except ImportError:
    import pickle

try:
    # For Python 3.0 and later
    from urllib.request import urlretrieve
except ImportError:
    # Fall back to Python 2's urllib2
    from urllib2 import urlretrieve

def load_data(url):
    substr = 'nphc-data'
    ind = url.find(substr)
    assert ind > -1, "The url should include the substring 'nphc-data' to import the right datasets."
    dataset = 'datasets/' + url[ind+len(substr):]
    
    import gzip, pickle
    print('Downloading data from %s' % url)
    urlretrieve(url, dataset)
    print('... loading data')
    f = gzip.open(dataset, 'rb')
    cumul = pickle.load(f)
    f.close()
    return cumul
