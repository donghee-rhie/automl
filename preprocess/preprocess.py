
from __future__ import division, print_function
import pandas as pd
import numpy as np
from pandas import DataFrame, Series

import gc
import pickle
import warnings
import os,shutil
from scipy.stats import skew

from sklearn.preprocessing import MinMaxScaler, StandardScaler


__all__ = [
'initializer',
'nullTreatment',
'outlierTreatment',
'logTransform',
'normalization'
]

class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    OKRED = '\033[91m'
    LIGHTCYAN = "\033[36m"
    BOLD = "\033[1m"
    ENDC = "\033[0m"


def color_print(strMessage, strType, bBold=False):
    dicType = {"HEADER" : bcolors.HEADER, "OKBLUE" : bcolors.OKBLUE, "OKGREEN" : bcolors.OKGREEN, "OKRED" : bcolors.OKRED, "LIGHTCYAN" : bcolors.LIGHTCYAN, 
                      "BOLD" : bcolors.BOLD, "ENDC" : bcolors.ENDC, "NONE" : ""}
    if bBold == False: print(dicType[strType] + strMessage + dicType['ENDC'])
    else : print(dicType['BOLD'] + dicType[strType] + strMessage + dicType['ENDC'])



# def environment():

#     strPath = os.getcwd()

#     if os.path.isdir(strPath + '/Results/'):
#         shutil.rmtree(strPath + '/Results/')
#         os.makedirs(strPath + '/Results/Resources/')
#         os.makedirs(strPath + '/Results/Visualization/')
#         os.makedirs(strPath + '/Results/Models/')
#         os.makedirs(strPath + '/Results/Results/')
#     else:
#         os.makedirs(strPath + '/Results/Resources/')
#         os.makedirs(strPath + '/Results/Visualization/')
#         os.makedirs(strPath + '/Results/Models/')
#         os.makedirs(strPath + '/Results/Results/')

#     strPathResource = strPath + '/Results/Resources/'
#     strPathModel = strPath + '/Results/Models/'
#     strPathVis = strPath + '/Results/Visualization/'
#     strPathResult = strPath + '/Results/Results/'

#     return strPathResource, strPathModel, strPathVis, strPathResult



def initializer(df):
    """ Check whether target exists, and covert columns names into lower cases.

        Parameters
        ----------
        df : pandas DataFrame
             input feature table

        is_train : bool, default=True
             option for print log (train/valid)

        Returns
        -------
        df : pandas DataFrame
        """
        
    # make columns names in lower case
    columns = df.columns
    columns = [c.lower() for c in columns]    
    df.columns = columns
    
    if 'target' not in columns:
        color_print("     Fail : No target in dataset", "OKRED", bBold=True)
        return

    return df



class nullTreatment:

    def __init__(self):
        self.info_median_ = ''        

    def fit(self, df):
        """ get median values per feature

        Parameters
        ----------
        df : pandas DataFrame
             input feature table

        """
        info_median = df.median()
        self.info_median_ = info_median
        return self

    def transform(self, df):
        """ fill null values with median value 

        Parameters
        ----------
        df : pandas DataFrame
             input feature table

        Returns
        -------
        df : pandas DataFrame 
        """

        warnings.filterwarnings('ignore')
        df.fillna(self.info_median_, inplace=True)
        warnings.filterwarnings('default')
        return df



class outlierTreatment:

    def __init__(self, thres_ = 2.5):
        self.thres_ = thres_

    def fit(self, df):
        """ Outlier treatment
        * replace with (mean + thres*std) if there's a value is '> (mean + thres*std)' 
        * replace with (mean - thres*std) if there's a value is '< (mean - thres*std)' 

        Parameters
        ----------
        df : pandas DataFrame (train set)
             input feature table
        thres : float, default = 2.5
                threshold for outlier treatment

        Returns
        -------
        df : pandas DataFrame 
        info_outlier : dict
                       cut-off of outlier for each features 
        """
        warnings.filterwarnings('ignore')

        thres = self.thres_
        info_outlier = {}

        for col in df.columns:
                
            if col == 'target':
                pass
            
            else:
                
                column_values = list(df[col])
                
                outlier_min = np.mean(column_values) - thres*np.std(column_values)
                outlier_max = np.mean(column_values) + thres*np.std(column_values)

                info_outlier[col] = [outlier_min, outlier_max]

                outlier_above = [x>outlier_max for x in column_values]
                outlier_below = [x<outlier_min for x in column_values]

                for i, values in enumerate(zip(column_values, outlier_above)):
                    if values[1]:
                        column_values[i] = outlier_max

                for i, values in enumerate(zip(column_values, outlier_below)):
                    if values[1]:
                        column_values[i] = outlier_min
        
                df.loc[:,col] = column_values

        self.info_outlier_ = info_outlier
        warnings.filterwarnings('default')
        return self

    def transform(self, df):
        """ Outlier treatment
        * replace with (mean + thres*std) if there's a value is '> (mean + thres*std)' 
        * replace with (mean - thres*std) if there's a value is '< (mean - thres*std)' 

        Parameters
        ----------
        df : pandas DataFrame (valid set)
             input feature table
        info_outlier : dict
             information about cut-off of outlier for each columns in training set

        Returns
        -------
        df : pandas DataFrame 
        """

        warnings.filterwarnings('ignore')

        for col in df.columns:
            
            if col == 'target':
                pass
            
            else:
                
                column_values = list(df[col])
                
                info_outlier = self.info_outlier_

                outlier_min = info_outlier[col][0]
                outlier_max = info_outlier[col][1]

                outlier_above = [x>outlier_max for x in column_values]
                outlier_below = [x<outlier_min for x in column_values]

                for i, values in enumerate(zip(column_values, outlier_above)):
                    if values[1]:
                        column_values[i] = outlier_max

                for i, values in enumerate(zip(column_values, outlier_below)):
                    if values[1]:
                        column_values[i] = outlier_min
        
                df.loc[:,col] = column_values

        warnings.filterwarnings('default')

        return df



class logTransform:

    def __init__(self, thres_=2.0):
        self.thres_ = thres_

    # check whether minimum value is negative
    def negative_checker(x):
        if np.min(x) < 0 : bFlag = True
        else : bFlag = False
        return bFlag

    def fit(self, df):
        """ log transformation if feature is skewed than threshold (training set)
        threshold = Float or int, 
                      if feature's abs(skew) > threshold, It will convert feature into log(feature + 0.001)  

        Parameters
        ----------
        df : pandas DataFrame (valid set)
             input feature table
        thres : float
             threshold for checking whether feature is skewed
    

        Returns
        -------
        df : pandas DataFrame 
        cols_skewed : list
                      list of skewed columns
        cols_not_skewed : list
                          list of not skewed columns
        """            

        warnings.filterwarnings('ignore')

        thres = self.thres_

        df_skew = df.copy()
        df_skew.drop('target', axis=1, inplace=True)
        
        skewness = df_skew.skew()
        
        cols_skewed = list(df_skew.columns[skewness > thres])
        cols_not_skewed = list(df_skew.columns[skewness <= thres])
        
        df_skewed = np.log(df_skew.loc[:,cols_skewed]+0.001)
        skewed_columns = [col+"_log" for col in list(df_skewed.columns)]
        df_skewed.columns = skewed_columns
        
        df_skew = pd.concat([df_skewed, df_skew.loc[:,cols_not_skewed]], axis=1)
 
        self.cols_skewed_ = cols_skewed
        self.cols_not_skewed_ = cols_not_skewed  

        warnings.filterwarnings('default')

        return self

    def transform(self, df):

        warnings.filterwarnings('ignore')

        cols_skewed = self.cols_skewed_  
        cols_not_skewed = self.cols_not_skewed_ 

        target = df.target
        df.drop("target", axis=1, inplace=True)

        df_skewed = np.log(df.loc[:,cols_skewed]+0.001)
        skewed_columns = [col+"_log" for col in list(df_skewed.columns)]
        df_skewed.columns = skewed_columns
        
        df = pd.concat([df_skewed, df.loc[:,cols_not_skewed]], axis=1)
        df.loc[:,'target'] = target

        warnings.filterwarnings('default')

        return df



class normalization:

    def __init__(self, method_='MinMax', n_bins_=20):
        self.method_ = method_
        self.n_bins_ = n_bins_

    def fit(self,df):
        """ Feature normalization (trainset)
        
        Parameters
        ----------
        df : pandas DataFrame (train set)
             input feature table
        method : string, default = 'MinMax'
                 method of normalization
                 * MinMax : MinMax Scaling
                 * Standard : Standardization
                 * Binning : Binning
        n_bins : int, default=20
                 number of bins, used when method="Binning"

        Returns
        -------
        df : pandas DataFrame 
             scaled dataset
        df_origin : pandas DataFrame
                    input dataset, Not scaled  
        scaler : scikit-learn scaler ('Standard', 'MinMax'), dict ("Binning")
                 trained scaler on train dataset
        """ 

        warnings.filterwarnings('ignore')

        method = self.method_
        n_bins = self.n_bins_

        df_fit = df.copy()

        if method == "Binning":
            scaler = {}
            for col in list(df_fit.columns):            
                if col != 'target':
                    values = list(df_fit[col])
                    values.sort()
                    n_group = round(df_fit.shape[0] * (100/n_bins/100))
                    cuts = [i * n_group for i in range(n_bins)]
                    criteria = []
                    for i, v in enumerate(values):
                        if i in cuts:
                            criteria.append(v)
                    scaler[col] = criteria

            self.scaler_ = scaler
            return self
                
        if method == "MinMax": scaler = MinMaxScaler()
        else: scaler = StandardScaler()
        
        df_fit.drop("target", axis=1, inplace=True)
        scaler = scaler.fit(df_fit)
        
        self.scaler_ = scaler

        warnings.filterwarnings('default')

        return self

    def transform(self, df):

        warnings.filterwarnings('ignore')

        method = self.method_
        n_bins = self.n_bins_
        scaler = self.scaler_

        df_origin = df.copy()
    
        if method != "Binning":
        
            target = df.target
            df.drop("target", axis=1, inplace=True)
            columns = df.columns
            idx = df.index

            df = scaler.transform(df)
            df = DataFrame(df, columns=columns, index=idx)

            df['target'] = target

        else:

            for col in scaler.keys():                
                values = np.array(df[col])
                bins = np.digitize(values, scaler[col])
                df[col] = bins
        
        warnings.filterwarnings('default') 

        return df, df_origin
