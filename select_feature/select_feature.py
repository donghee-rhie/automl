
from __future__ import division, print_function
import pandas as pd
import numpy as np
from pandas import DataFrame, Series
import gc
import pickle
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from numpy import around
from scipy.stats import linregress
from sklearn.model_selection import train_test_split

import math
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import seaborn as sns

from ..preprocess import bcolors, color_print


__all__ = [
'selectFeature',
'fit_LogisticRegression'
]



class selectFeature:
        
    def fit_IV(self,df, bins=10):
        """ calculate each feature's IV value, 
        and remove feature which has IV value lower than threshold

         * criteria example
            - Less than 0.02 : unpredictive
            - 0.02 to 0.1 : weak
            - 0.1 to 0.3 : medium
            - 0.3 + : strong
            
        Parameters
        ----------
        df : pandas DataFrame
             input feature table
        bins : int, default = 10
               number of bins when calculate IV value
        thres : float, default = 0.02
                criteria of IV value

        Returns
        -------
        dict_IV : dict
                  IV value for each features 
        table_feature : pandas DataFrame
                        information about binning (count, lift, etc)

        df :  pandas DataFrame
        """   

        warnings.filterwarnings('ignore')

        # Function for calculate IV value
        def IVvalue(feature, target, bins):

            table = DataFrame({"column" : feature, "TARGET" : df.target})
            table = table.sort_values(by="column",ascending=False)

            n_group = round(table.shape[0] * (100/bins/100))
            rest = table.shape[0] - n_group * (bins-1)
            result_rank = np.r_[np.repeat(np.arange(1,bins), n_group), np.repeat(bins, rest)]
            table["NTILE"] = result_rank

            agg = pd.pivot_table(table, index="NTILE", columns="TARGET", aggfunc=np.size, fill_value=0)
            agg = agg.reset_index().values.astype(int)

            rate_N = agg[:,1] / agg[:, 1].sum()
            rate_Y = agg[:,2] / agg[:, 2].sum()

            agg = np.c_[agg, rate_N, rate_Y]
            woe = np.log(agg[:,3] / agg[:,4])
            iv = (agg[:,3] - agg[:,4]) * woe

            IV = iv.sum()
            agg = DataFrame(np.c_[agg, woe, iv], columns = ["bin", "cnt_n", "cnt_y", "rate_n", "rate_y", "woe", "iv"])
            agg["cnt"] = agg["cnt_n"] + agg["cnt_y"]
            agg["response"] = agg["cnt_y"] / agg["cnt"]

            return IV, agg
        
        dict_IV = dict()
        table_feature = DataFrame()

        # convert column names in lower cases
        cols = [c.lower() for c in df.columns]
        df.columns = cols

        # update information for each features if IV value's bigger than threshold
        for column in df.columns:
            if column.lower() == "target":
                pass
            else:
                IV, agg = IVvalue(df[column], df.target, bins) 
                dict_IV[column] = IV
                agg['feature'] = column
                table_feature = pd.concat([table_feature, agg], axis=0)

        self.dict_IV_ = dict_IV
        self.table_feature_ = table_feature

        warnings.filterwarnings('default')

        return self

    def fit_corr(self,df):

        warnings.filterwarnings('ignore')

        columns = list(df.columns)
        columns.remove("target")
        
        corr_mtrx = df[columns].corr()

        self.corr_mtrx_ = corr_mtrx

        warnings.filterwarnings('default')

        return self

    def fit_feature_selection(self, df, thres_IV=0.02, thres_corr=0.9):

        warnings.filterwarnings('ignore')

        dict_IV = self.dict_IV_
        
        selected_features = []
        for key, value in dict_IV.items():
            if value >= thres_IV:
                selected_features.append(key)

        corr_mtrx = df[selected_features].corr()
        n_rows_cols = corr_mtrx.shape[0]
        
        for c in range(n_rows_cols):
            for r in range(n_rows_cols):
                if c>r:
                    if abs(corr_mtrx.iloc[r,c]) > thres_corr:
                        try:
                            selected_features.remove(corr_mtrx.index[r])
                        except:
                            pass
        
        self.selected_features_ = selected_features
        warnings.filterwarnings('default')

        return self

    def vis_lift(self, selected_features = [], n_features=25, savepath = ''):        

        table_feature = self.table_feature_
        if selected_features == []:
            selected_features = self.selected_features_
            
        if n_features > len(selected_features):
            n_features = len(selected_features)
        
        warnings.filterwarnings('ignore')
    
        sum_iv = table_feature[["feature", "iv"]].groupby("feature").agg({"iv" : sum}).reset_index()
        sum_iv.columns = ["feature", 'iv_sum']
        table_feature = pd.merge(table_feature, sum_iv, left_on = "feature", right_on = "feature")

        # make lift column
        n_total = table_feature[table_feature.feature == table_feature.feature.unique()[0]].cnt.sum()
        n_target = table_feature[table_feature.feature == table_feature.feature.unique()[0]].cnt_y.sum()
        target_rate = n_target / n_total

        table_feature['lift'] = table_feature.response / target_rate

        table_feature = table_feature[table_feature.feature.isin(selected_features)]
        table_feature.sort_values(by = ["iv_sum", "feature", "bin"], ascending=[False, True, False], inplace=True)

        vis_features = list(table_feature.feature.unique())
        vis_features = vis_features[:n_features]

        width = math.floor(np.sqrt(n_features))
        
        if (n_features % width) == 0:
            height = n_features // width
        else:
            height = (n_features // width) + 1
        
        plt.figure(figsize=(width*18, height*18))

        maxLift = math.ceil(table_feature.lift.max())
        
        for i, f in enumerate(vis_features):

            tmp = table_feature[table_feature.feature == f]
            iv = tmp.iv_sum.unique()

            ax1 = pl.subplot(height, width, i+1)
            pl.title("\n" + f + "\n" + "IV value : " + str(iv) + "\n", loc='left', fontsize=50)
            pl.tight_layout(pad=1)
            pl.xticks(np.arange(len(tmp.bin)), list(tmp.bin), fontsize=35, rotation=45)
            pl.yticks(fontsize=30)

            ax1.plot(np.arange(len(tmp.bin)), list(tmp.lift), color='orange', marker='o', \
                         linewidth=8, markersize=20, markerfacecolor='w', label='LIFT')
            ax1.hlines(y=1, xmin=0, xmax=len(tmp.bin)-1, color='gray', linestyle='--', linewidth=6)
            pl.ylim(0, maxLift)
            pl.yticks(fontsize=30)

            if i==0:
                ax1.set_ylabel("LIFT", fontsize=50)
                plt.legend(fontsize=50)

            plt.savefig(savepath + 'Vis_Lift.jpg') 
            warnings.filterwarnings('default')

    def vis_hist(self, df, selected_features=[], n_features=25, savepath = ''):
        
        table_feature = self.table_feature_
        if selected_features == []:
            selected_features = self.selected_features_

        if n_features > len(selected_features):
            n_features = len(selected_features)
            
        warnings.filterwarnings('ignore')

        sum_iv = table_feature[["feature", "iv"]].groupby("feature").agg({"iv" : sum}).reset_index()
        sum_iv.columns = ["feature", 'iv_sum']
        table_feature = pd.merge(table_feature, sum_iv, left_on = "feature", right_on = "feature")

        # make lift column
        n_total = table_feature[table_feature.feature == table_feature.feature.unique()[0]].cnt.sum()
        n_target = table_feature[table_feature.feature == table_feature.feature.unique()[0]].cnt_y.sum()
        target_rate = n_target / n_total

        table_feature['lift'] = table_feature.response / target_rate

        table_feature = table_feature[table_feature.feature.isin(selected_features)]
        table_feature.sort_values(by = ["iv_sum", "feature", "bin"], ascending=[False, True, False], inplace=True)

        vis_features = list(table_feature.feature.unique())
        vis_features = vis_features[:n_features]

        width = math.floor(np.sqrt(n_features))

        if (n_features % width) == 0:
            height = n_features // width
        else:
            height = (n_features // width) + 1

        f, axes = plt.subplots(height, width, figsize=(30, 30), sharex=True)

        i = 0

        for row in range(height):
            for col in range(width):
                i += 1
                if i > n_features:
                    break
                else:
                    tbl = pd.merge(DataFrame(df.loc[:,vis_features[i-1]]), DataFrame(df.target), left_index=True, right_index=True)
                    sns.distplot(tbl[tbl.target==1].iloc[:,0], color="tomato", label="Y", hist=False, ax=axes[row,col])
                    sns.distplot(tbl[tbl.target==0].iloc[:,0] , color="darkgrey", label="N", hist=False, ax=axes[row,col])

        plt.savefig(savepath + 'Vis_hist.jpg')
        warnings.filterwarnings('default')



class fit_LogisticRegression:

    def fit_binning(self, X, y,max_leaf_nodes=5, min_samples_leaf=0.05, random_state = 1234):

        # bin variables with DecisionTree
        def binning(df, feature, target):
            
            clf =DecisionTreeClassifier(random_state=random_state, 
                                                        max_leaf_nodes=max_leaf_nodes, 
                                                        min_samples_leaf=min_samples_leaf)
            
            table = DataFrame({'feat' : df[feature], "target" : target})
            clf.fit(table[['feat']], table[['target']])
            table['group'] = clf.apply(table[['feat']])
            
            g_max = around(table[['group', 'feat']].groupby("group").max(),3)
            g_min = around(table[['group', 'feat']].groupby("group").min(),3)
            
            g_size = table.groupby("group").size() 
            g_size_y = table[table.target==1].groupby('group').size()
    
            info = pd.merge(g_min, g_max, left_index=True, right_index=True)
            info["total"] = g_size
            info['target'] = g_size_y
            info["rate"] =  info['target'] / info["total"]
            base = info.target.sum() / info.total.sum()
            info['lift'] = around(info.rate / base, 3)
            info['feature'] = feature
            info.columns = ["min", "max", "total", "target", "rate", "lift", "feature"]
            info = info.sort_values(by="min", ascending=True)
            g_name = [str(i) for i in range(info.shape[0])]
            info["g_name"] = g_name
            
            return clf, info

        warnings.filterwarnings('ignore')

        dict_clf = {}
        info = DataFrame()
        
        for col in X.columns:
            clf, col_info = binning(X, col, y)
            info = pd.concat([info, col_info], axis=0)
            dict_clf[col] = clf

        self.dict_clf_ = dict_clf
        self.info_bin_ = info

        return self


    def fit(self, X, y, n_features=50):

        binned = DataFrame()
        dict_clf = self.dict_clf_
        info = self.info_bin_

        for col in list(X.columns):
            clf = dict_clf[col]
            prediction = clf.apply(X[[col]])
            prediction = DataFrame(prediction, columns = ['col'], index=X.index)

            info_bin_col = info[info.feature.isin([col])]
            prediction_g = pd.merge(prediction, info_bin_col, left_on="col", right_index=True)
            prediction_g = prediction_g['g_name']

            colname = col + "_G"
            binned[colname] = prediction_g

        # remove constant
        for col in binned.columns:
            if len(binned[col].unique())==1:
                binned.drop(col, axis=1, inplace=True)
                col = col.replace("_G","")
                info = info[~info.feature.isin([col])]
        var = list(binned.columns)

        # fit logistic regression
        threshold = 0
        for i in range(1000):

            df_binned_OH = pd.get_dummies(binned.loc[:,var])
            
            df_binned_OH.sort_index(inplace=True)
            y.sort_index(inplace=True)
            
            X_train, X_test, y_train, y_test = train_test_split(df_binned_OH, y, test_size=0.3, random_state=1234)
            clf = LogisticRegression(C=0.1)
            clf.fit(X_train, y_train)

            coefs = pd.concat([DataFrame(X_train.columns), DataFrame(clf.coef_).T], axis=1)
            coefs.columns = ['key', 'coef']
            info['key'] = info.feature + "_G_" + info.g_name
            coefs = pd.merge(info, coefs, left_on='key', right_on='key')
            
            coefCols = list(coefs.feature.unique())
            delcols = []
            for col in coefCols:
                df = coefs[coefs.feature == col]
                slope, _,  _,  _, _ = linregress(df.lift, df.coef)
                
                if slope < threshold:
                    delcols.append(col)

            delcols = list(map(lambda x : x+"_G", delcols))
            var = list(set(var).difference(set(delcols)))
            
            
            print("       Trial {} finished. {} variables removed, {} variables survived, AUROC : {:.4f}".format(i, len(delcols), len(var), roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])))

            if len(var) > n_features and len(delcols)==0 : 
                 threshold = threshold + 0.02

            if len(var) <= n_features and len(delcols)==0 : 
                print("      Finished. {} variables survived, AUROC : {:.4f}".format(len(var), roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])))
                break
        
        logit_selected_features = [v.replace("_G","") for v in var]

        self.coefs_ = coefs
        self.selected_features_ = logit_selected_features
        self.clf_ = clf        
        return self

    def predict_proba(self, X):

        binned = DataFrame()
        dict_clf = self.dict_clf_
        info = self.info_bin_
        features = self.selected_features_
        logit_clf = self.clf_

        for col in features:
            clf = dict_clf[col]
            prediction = clf.apply(X[[col]])
            prediction = DataFrame(prediction, columns = ['col'], index=X.index)

            info_bin_col = info[info.feature.isin([col])]
            prediction_g = pd.merge(prediction, info_bin_col, left_on="col", right_index=True)
            prediction_g = prediction_g['g_name']

            colname = col + "_G"
            binned[colname] = prediction_g

        df_binned_OH = pd.get_dummies(binned)
        preds = logit_clf.predict_proba(df_binned_OH)

        return preds