
from __future__ import division, print_function
import pandas as pd
import numpy as np
from pandas import DataFrame, Series
import warnings


def performance(y_true, y_pred, bins=20, lift_percentage=0.1):
    """ calculating KS / AUROC / LIFT 

        Parameters
        ----------
        
        y_true : label (0 or 1)
        y_pred : prediction value
        bins : int, default=20
        	   # of bins to make with predict values
        lift_percentage : float, default=0.1
        	    point where observing Lift : (i.e. 0.1 -> lift @ upper 10%)

        Returns
        -------
        KS    : float
        AUROC : float 
        LIFT  : float
        agg   : pandas DataFame
        	    Aggregation table, which is used in performance calculation
        """
    warnings.filterwarnings('ignore')
    
    table = DataFrame({"TARGET" : y_true, "PRED" : y_pred})
    table = table.sort_values(by="PRED",ascending=False)
    
    n_group = round(table.shape[0] * (100/bins/100))
    rest = table.shape[0] - n_group * (bins-1)
    result_rank = np.r_[np.repeat(np.arange(1,bins), n_group), np.repeat(bins, rest)]
    table["NTILE"] = result_rank
    
    agg = pd.pivot_table(table, index="NTILE", columns="TARGET", aggfunc=np.size, fill_value=0)
    agg = agg.reset_index().values
    
    cumsum_N = agg[:,1].cumsum().astype(int)
    cumsum_Y = agg[:,2].cumsum().astype(int)
    agg = np.c_[agg, cumsum_N, cumsum_Y].astype(int)
    
    cumRate_N = agg[:,3] / agg[bins-1, 3]
    cumRate_Y = agg[:,4] / agg[bins-1, 4]
    diff_cum = cumRate_Y - cumRate_N
    KS = np.max(diff_cum)
    
    cumRate_N_2 = cumRate_N[1:]
    cumRate_N_2 = cumRate_N_2 - cumRate_N[:bins-1]
    cumRate_N_2 = np.append(cumRate_N[0], cumRate_N_2)
    AUROC = np.sum(cumRate_N_2 * cumRate_Y)
    
    res = agg[:,2] / (agg[:,1] + agg[:,2])
    cumRes = cumsum_Y / (cumsum_N + cumsum_Y)
    
    LIFT = cumRes[int(lift_percentage / (100/bins/100))-1] / cumRes[bins-1]
    
    agg = DataFrame(agg, columns = ["NTILE", "CNT_N", "CNT_Y", "CNT_CUM_N", "CNT_CUM_Y"])
    agg["RATE_CUM_N"] = cumRate_N
    agg["RATE_CUM_Y"] = cumRate_Y
    warnings.filterwarnings('default')
    
    return KS, AUROC, LIFT, agg