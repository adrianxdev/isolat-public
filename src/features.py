# +
def time_feats(series):
    """
        Input:
        series: input time series of daily counts
        
        Output:
        feats: dataframe of lin, quad time features
    """
    base_date = dt.datetime.strptime('01/01/20', '%m/%d/%y')
    end_date = series.index[-1]
    series_len = series.size
    
    # linear and quadratic time features:
    days = np.array([ (date - base_date).days for date in series.index ])
    poly = skl.preprocessing.PolynomialFeatures(degree=2, include_bias=False)
    poly_feats = poly.fit_transform( days.reshape((days.size, 1)))
    
    
    feats = pd.DataFrame(poly_feats, columns=['d1', 'd2'], index=series.index)
    return feats

def dotw_feats(series):
    """
        Input:
        series: input time series of daily counts
        
        Output:
        feats: dataframe of day of the week features
    """
    dotw_list = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    dotw_feats_list = []
    
    for idx, day in enumerate(dotw_list):
        dotw_feats_list.append(pd.Series( [idx == date.weekday() for date in series.index] ,
                                    index=series.index, name = day) )
    
    feats = pd.concat(dotw_feats_list, axis=1)*1.0
    return feats

def featurize(series, feats_list = None):
    """
    Wrapper function that featurizes available features
    TODO: add feats_list functionality, ignore for now
    """
    
    feats_dict = {'time': time_feats, 'dotw': dotw_feats} 
    
    return pd.concat([feats(series) for _, feats in feats_dict.items()  ], axis=1  )
    

