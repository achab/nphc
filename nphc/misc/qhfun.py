import numpy as np


def PriceUp(df):
    return (df.PriceMove > 0).values

def PriceDown(df):
    return (df.PriceMove < 0).values

def GetLimit(df):
    return (df.OrderType == 1).values

def GetCancel(df):
    return (df.OrderType == -1).values

def GetTrades(df):
    return (df.OrderType == 0).values

def MarketOrdersAsk(df):
    return (df.PriceMove == 0).values * (df.Side == 1).values * (df.OrderType == 0).values

def MarketOrdersBid(df):
    return (df.PriceMove == 0).values * (df.Side == -1).values * (df.OrderType == 0).values

def GetOrderBookEvents(df):
    """ Given df representing one day of data, returns list of array representing
        order book events for mid-price moves (ask/bid), number of market orders
        (ask/bid), number of limit order (ask/bid), number of cancel orders (ask/bid)"""

    out = [np.copy(df.loc[PriceUp(df),"Time"].values),
           np.copy(df.loc[PriceDown(df),"Time"].values),

           # market no price move
           np.copy(df.loc[(df.OrderType == 0) & (df.Side == 1) & (df.PriceMove == 0),"Time"].values),
           np.copy(df.loc[(df.OrderType == 0) & (df.Side == -1) & (df.PriceMove == 0),"Time"].values),

           # limi no price move
           np.copy(df.loc[(df.OrderType == 1) & (df.Side == 1) & (df.PriceMove == 0),"Time"].values),
           np.copy(df.loc[(df.OrderType == 1) & (df.Side == -1) & (df.PriceMove == 0),"Time"].values),

           # cancel mo price move
           np.copy(df.loc[(df.OrderType == -1) & (df.Side == 1) & (df.PriceMove == 0),"Time"].values),
           np.copy(df.loc[(df.OrderType == -1) & (df.Side == -1) & (df.PriceMove == 0),"Time"].values)]

    return out


def GetPriceAndTrades(df):
    """ Given df representing one day of data, returns list of array representing
        order book events for mid-price moves (ask/bid), number of market orders
        (ask/bid) that do not move the midprice """

    out = []
    out += [np.copy(df.loc[PriceUp(df), "Time"].values)]
    out += [np.copy(df.loc[PriceDown(df), "Time"].values)]
    out += [np.copy(df.loc[MarketOrdersAsk(df), "Time"].values)]
    out += [np.copy(df.loc[MarketOrdersBid(df), "Time"].values)]

    return out


def GetPriceComponents(df, edges):
    """ Separate components based on the magnitude of price change they generate"""

    mask = lambda x, y: (df.OrderType==0).values * (np.abs(df.Volume.values)>x) * (np.abs(df.Volume.values)<=y)
    out = []
    for ii in range(len(vol_bins)-1):
        out += [np.copy(df.loc[mask(vol_bins[ii], vol_bins[ii+1]), 'Time'].values)]
    return out


def GetComponents(df, vol_bins):
    """ Given df representing one day of data, returns list of array representing
        trades time for each volume component"""

    mask = lambda x, y: (df.OrderType==0).values * (np.abs(df.Volume.values)>x) * (np.abs(df.Volume.values)<=y)
    out = []
    for ii in range(len(vol_bins)-1):
        out += [np.copy(df.loc[mask(vol_bins[ii], vol_bins[ii+1]), 'Time'].values)]
    return out
