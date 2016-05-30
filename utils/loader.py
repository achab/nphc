try:
    import cPickle as pickle
except ImportError:
    import pickle

try:
    # For Python 3.0 and later
    from urllib.request import urlretrieve
except ImportError:
    # Fall back to Python 2's urllib2
    from urllib import urlretrieve

def load_data(url):
    import os.path
    substr = 'nphc-data'
    ind = url.find(substr)
    assert ind > -1, "The url should include the substring 'nphc-data' to import the right datasets."
    dataset = 'downloads/' + url[ind+len(substr):]
    if not os.path.isfile(dataset):
        print('Downloading data from %s' % url)
        urlretrieve(url, dataset)
        print('... loading data')
    import gzip
    f = gzip.open(dataset, 'rb')
    data = pickle.load(f,encoding='latin1')
    f.close()
    return data
