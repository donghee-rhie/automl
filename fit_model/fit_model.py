
from __future__ import division, print_function
import pandas as pd
import numpy as np
from pandas import DataFrame, Series
import gc
import pickle
import warnings
from itertools import product

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier

import xgboost as xgb
import lightgbm as lgb

from automl.preprocess import bcolors, color_print
from automl.fit_model import performance
from functools import reduce
from sklearn.linear_model import LogisticRegression



class fit_classifier:

    def fit_trees(self, X, y, classifier = 'rf', cv=False, gridsearch=False, n_random_search=10, n_cv_fold=5, learning_rate=0.02, early_stopping_rounds=100, n_jobs=10):

        warnings.filterwarnings('ignore')

        ## function for CV
        def fit_cv(X, y, params):

            skf = StratifiedKFold(n_splits=n_cv_fold, random_state=1234, shuffle=True)

            if classifier == 'rf':
                clf = RandomForestClassifier(**params)
            elif classifier == 'xgb':
                clf = xgb.XGBClassifier(**params)
            else:
                clf = lgb.LGBMClassifier(**params)

            result = list()

            X.sort_index(inplace=True)
            y.sort_index(inplace=True)

            for train_idx, test_idx in skf.split(X,y):
                X_train, X_test = X.iloc[train_idx,:], X.iloc[test_idx,:]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                if classifier == 'rf':
                    clf.fit(X_train, y_train)
                elif classifier == 'xgb':
                    clf.fit(X_train, y_train, eval_set = [(X_train,y_train), (X_test,y_test)],
                                                       eval_metric='auc',
                                                       early_stopping_rounds=early_stopping_rounds,
                                                       verbose=False)
                else:
                    clf.fit(X_train, y_train,
                                                        eval_set=(X_test,y_test),
                                                        early_stopping_rounds=early_stopping_rounds,
                                                        eval_metric = ['auc'],
                                                        verbose=False
                                                        )
                    
                auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
                result.append(auc)

            performance = np.array(result).mean()

            return performance

        color_print("     Model building : {0}".format(classifier), "OKBLUE", bBold=True)


        if gridsearch:

            color_print("       GridSerach....", "OKBLUE", bBold=True)

            ## cross-validation and find best paramter

            if classifier == 'rf':
                grid_param = {'n_estimators' : [100, 200, 400], 
                              'max_depth' : [7,9,11], 
                              'max_features' : [0.5,0.7,0.9]} 

            else:
                grid_param = {'n_estimators' : 10000,
                              'learning_rate' : learning_rate,
                              'max_depth' : [5,7,9,11], 
                              'colsample_bytree' : [0.5,0.7,0.9],
                              'subsample' : [0.7,1]} 

            result_grid = {}

            # GridSearch + CV
            if cv:
                color_print("       Cross Validation..", "OKBLUE", bBold=True)
                params_cv = list(grid_param.keys())
                n_params_cv = len(params_cv)
                if classifier == 'rf':
                    combinations = list(product(grid_param[params_cv[0]],grid_param[params_cv[1]], grid_param[params_cv[2]]))
                    for c in combinations:
                        params = {params_cv[0] : c[0],
                                  params_cv[1] : c[1],
                                  params_cv[2] : c[2],
                                  'n_jobs' : n_jobs}
                        auroc = fit_cv(X,y,params)
                        params.pop('n_jobs')
                        result_grid[auroc] = params
                        print("         Parameter : {}, AUROC : {:.4f}".format(params, auroc))

                else:
                    combinations = list(product(
                                                                [grid_param[params_cv[0]]], 
                                                                [grid_param[params_cv[1]]], 
                                                                grid_param[params_cv[2]], 
                                                                grid_param[params_cv[3]], 
                                                                grid_param[params_cv[4]]
                                                                ))
                    for c in combinations:
                        params = {params_cv[0] : c[0],
                                          params_cv[1] : c[1],
                                          params_cv[2] : c[2],
                                          params_cv[3] : c[3],
                                          params_cv[4] : c[4]}
                        if classifier=='xgb':
                            params['nthread'] = n_jobs
                            params['verbose'] = False
                            
                            auroc = fit_cv(X,y,params)
                            
                            params.pop("n_estimators")
                            params.pop("nthread")
                            params.pop("learning_rate")
                            params.pop("verbose")
                        
                        else:
                            params['silent'] = True
                            params['num_threads'] = n_jobs
                            
                            auroc = fit_cv(X,y,params)
                            
                            params.pop('n_estimators')
                            params.pop('silent')
                            params.pop('learning_rate')
                            params.pop('num_threads')
                            
                        result_grid[auroc] = params
                        print("         Parameter : {}, AUROC : {:.4f}".format(params, auroc))

            # GridSearch + No CV
            else:
                color_print("       No Cross Validation..", "OKBLUE", bBold=True)
                X_train, X_test, y_train, y_test = train_test_split(X.sort_index(), y.sort_index(), test_size=0.3, random_state=1234)

                X_train.sort_index(inplace=True)
                X_test.sort_index(inplace=True)
                y_train.sort_index(inplace=True)
                y_test.sort_index(inplace=True)

                params_cv = list(grid_param.keys())
                n_params_cv = len(params_cv)
                if classifier=='rf':
                    combinations = list(product(grid_param[params_cv[0]],grid_param[params_cv[1]], grid_param[params_cv[2]]))
                    for c in combinations:
                        params = {params_cv[0] : c[0],
                                  params_cv[1] : c[1],
                                  params_cv[2] : c[2],
                                   'n_jobs' : n_jobs}
                        clf = RandomForestClassifier(**params)
                        clf.fit(X_train, y_train)
                        auroc = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
                        params.pop('n_jobs')
                        result_grid[auroc] = params
                        print("         Parameter : {}, AUROC : {:.4f}".format(params, auroc))

                else:
                    combinations = list(product(
                                                                [grid_param[params_cv[0]]], 
                                                                [grid_param[params_cv[1]]], 
                                                                grid_param[params_cv[2]], 
                                                                grid_param[params_cv[3]], 
                                                                grid_param[params_cv[4]]
                                                                ))
                    for c in combinations:
                        params = {params_cv[0] : c[0],
                                  params_cv[1] : c[1],
                                  params_cv[2] : c[2],
                                  params_cv[3] : c[3],
                                  params_cv[4] : c[4]}
                        if classifier == 'xgb':
                            params['nthread'] = n_jobs
                            params['verbose'] = False
                            
                            clf = xgb.XGBClassifier(**params)
                            clf.fit(X_train, y_train, eval_set = [(X_train,y_train), (X_test,y_test)],
                                                      eval_metric='auc',
                                                      early_stopping_rounds=early_stopping_rounds,
                                                      verbose=False)
                            
                            auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
                            params['best_iteration'] = clf.best_iteration
                            params.pop("n_estimators")
                            params.pop("nthread")
                            params.pop("learning_rate")
                            params.pop("verbose")
                            
                        else:
                            params['silent'] = True
                            params['num_threads'] = n_jobs
                            
                            clf = lgb.LGBMClassifier(**params)
                            clf.fit(X_train, y_train,
                                                        eval_set=(X_test,y_test),
                                                        early_stopping_rounds=early_stopping_rounds,
                                                        eval_metric = ['auc'],
                                                        verbose=False
                                                        )
                            
                            params['best_iteration'] = clf.best_iteration_
                            params.pop('n_estimators')
                            params.pop('silent')
                            params.pop('learning_rate')
                            params.pop('num_threads')
                        
                        auroc = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
                        result_grid[auroc] = params
                        print("         Parameter : {}, AUROC : {:.4f}".format(params, auroc))

            max_auroc = np.array(list(result_grid.keys())).max()
            best_params = result_grid[max_auroc]

            color_print("\n        Best Parameter for {0} : {1}".format(classifier, best_params), "LIGHTCYAN", bBold=False)
            color_print("\n        Complete : parameter tunning, \n        Processing : fitting with best parameters...\n", "LIGHTCYAN", bBold=False)

        else:

            color_print("       RandomSearch....", "OKBLUE", bBold=True)

            if classifier == 'rf':
                param_range = {'n_estimators' : [100,600], 
                              'max_depth' : [7,11], 
                              'max_features' : [0.5,0.9]} 

            else:
                param_range = {'n_estimators' : 10000,
                              'learning_rate' : learning_rate,
                              'max_depth' : [5,11], 
                              'colsample_bytree' : [0.5,0.9],
                              'subsample' : [0.7,1]} 
            result_grid = {}

            # RandomSearch + CV
            if cv:

                color_print("        Cross Validation...", "OKBLUE", bBold=True)
                for i in range(n_random_search):
                    params = {}
                    for key in list(param_range.keys()):
                        if key == "max_features" or key == "colsample_bytree" or key == "subsample":
                            params[key] = round(np.random.uniform(param_range[key][0], param_range[key][1]),2)
                        elif type(param_range[key]) != list:
                            params[key] = param_range[key]
                        else:
                            params[key] = np.random.randint(param_range[key][0], param_range[key][1])

                    if classifier == 'rf':
                        params['n_jobs'] = n_jobs
                        auroc = fit_cv(X,y,params)
                        params.pop('n_jobs')
                        
                    elif classifier == 'xgb':
                        params['nthread'] = n_jobs
                        params['verbose'] = False
                        
                        auroc = fit_cv(X,y,params)
                        
                        params.pop("n_estimators")
                        params.pop("nthread")
                        params.pop("learning_rate")
                        params.pop("verbose")
                
                        
                    else:
                        params['silent'] = True
                        params['num_threads'] = n_jobs
                        
                        auroc = fit_cv(X,y,params)
                        
                        params.pop('n_estimators')
                        params.pop('silent')
                        params.pop('learning_rate')
                        params.pop('num_threads')
                            
                    result_grid[auroc] = params
                    print("         Parameter : {}, AUROC : {:.4f}".format(params, auroc))

                max_auroc = np.array(list(result_grid.keys())).max()
                best_params = result_grid[max_auroc]

                color_print("        Best Parameter for {0} : {1}".format(classifier, best_params), "LIGHTCYAN", bBold=False)
                color_print("        Complete : parameter tunning \n        Processing : fitting with best parameters...", "LIGHTCYAN", bBold=False)

            # RandomSearch + No CV
            else:

                color_print("        No Cross Validation...", "OKBLUE", bBold=True)

                X_train, X_test, y_train, y_test = train_test_split(X.sort_index(), y.sort_index(), test_size=0.3, random_state=1234)

                X_train.sort_index(inplace=True)
                X_test.sort_index(inplace=True)
                y_train.sort_index(inplace=True)
                y_test.sort_index(inplace=True)

                for i in range(n_random_search):
                    params = {}
                    for key in list(param_range.keys()):
                        if key == "max_features" or key == "colsample_bytree" or key == "subsample":
                            params[key] = round(np.random.uniform(param_range[key][0], param_range[key][1]),2)
                        elif type(param_range[key]) != list:
                            params[key] = param_range[key]
                        else:
                            params[key] = np.random.randint(param_range[key][0], param_range[key][1])

                    if classifier == 'rf':
                        params['n_jobs'] = n_jobs
                        clf = RandomForestClassifier(**params)
                        clf.fit(X_train, y_train)
                        params.pop('n_jobs')

                    elif classifier == 'xgb':
                        params['nthread'] = n_jobs
                        params['verbose'] = False

                        clf = xgb.XGBClassifier(**params)

                        clf.fit(X_train, y_train, eval_set = [(X_train,y_train), (X_test,y_test)],
                                                      eval_metric='auc',
                                                      early_stopping_rounds=early_stopping_rounds,
                                                      verbose=False)

                        params['best_iteration'] = clf.best_iteration
                        params.pop("n_estimators")
                        params.pop("nthread")
                        params.pop("learning_rate")
                        params.pop("verbose")

                    else:

                        params['silent'] = True
                        params['num_threads'] = n_jobs

                        clf = lgb.LGBMClassifier(**params)
                        clf.fit(X_train, y_train,
                                                        eval_set=(X_test,y_test),
                                                        early_stopping_rounds=early_stopping_rounds,
                                                        eval_metric = ['auc'],
                                                        verbose=False
                                                        )

                        params['best_iteration'] = clf.best_iteration_
                        params.pop('n_estimators')
                        params.pop('silent')
                        params.pop('learning_rate')
                        params.pop('num_threads')
                        
                    auroc = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
                    result_grid[auroc] = params
                    print("         Parameter : {}, AUROC : {:.4f}".format(params, auroc))
                
                max_auroc = np.array(list(result_grid.keys())).max()
                best_params = result_grid[max_auroc]

                color_print("        Best Parameter for {0} : {1}".format(classifier, best_params), "LIGHTCYAN", bBold=False)
                color_print("        Processing : fitting with best parameters...", "LIGHTCYAN", bBold=False)

        X.sort_index(inplace=True)
        y.sort_index(inplace=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=1234)

        if classifier == 'rf':
            best_params['n_jobs'] = n_jobs
            clf = RandomForestClassifier(**best_params)
            clf.fit(X_train, y_train)
            best_params.pop('n_jobs')

        elif classifier == 'xgb':
            best_params['nthread'] = n_jobs
            best_params['verbose'] = False
            best_params['n_estimators'] = 10000
            best_params['learning_rate'] = learning_rate
            clf = xgb.XGBClassifier(**best_params)
            clf.fit(X_train, y_train, eval_set = [(X_train,y_train), (X_test,y_test)],
                                                          eval_metric='auc',
                                                          early_stopping_rounds=early_stopping_rounds,
                                                          verbose=False)
            best_params['best_iteration'] = clf.best_iteration
            best_params.pop("n_estimators")
            best_params.pop("nthread")
            best_params.pop("learning_rate")
            best_params.pop("verbose")

        else:
            best_params['silent'] = True
            best_params['num_threads'] = n_jobs
            best_params['n_estimators'] = 10000
            best_params['learning_rate'] = learning_rate
            clf = lgb.LGBMClassifier(**params)
            clf.fit(X_train, y_train,
                                    eval_set=(X_test,y_test),
                                    early_stopping_rounds=early_stopping_rounds,
                                    eval_metric = ['auc'],
                                    verbose=False
                                    )
            best_params['best_iteration'] = clf.best_iteration_
            best_params.pop('n_estimators')
            best_params.pop('silent')
            best_params.pop('learning_rate')
            best_params.pop('num_threads')

        color_print("        Best params : {}, AUROC : {:.4f}".format(best_params, roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])), "LIGHTCYAN", bBold=False)

        param_grid = DataFrame()

        for col, row in result_grid.items():
            r =pd.DataFrame(list(row.values())).T
            r.columns = row.keys()
            r['auc'] = col
            param_grid = pd.concat([param_grid, r], axis=0)

        param_grid.index = [i+1 for i in range(param_grid.shape[0])]

        warnings.filterwarnings('default')

        self.clf_ = clf
        self.param_grid_ = param_grid

        return self


    def fit_MLPClassifier(self, X,y,n_random_search=10):


        """ Find best structure of hidden layers, and Fit MLPclassifier with best structure of hidden layers


            Parameters
            ----------
            
            X : pandas DataFrame 
    			input features

            y : pandas Series (or DataFrame)
    			label (target)

    	    n_random_search : int, default = 10
    			 # of random search trial
    			 
            Returns
            -------
            clf : MLPClassifier, trained with best structure of hidden layers

            param_grid : pandas DataFrame
            		     history of parameter tunning (parameter combination trials and performance) 
        """   

        color_print("     Model building : MLPclassifier", "OKBLUE", bBold=True)

        # make combination of hidden layers

        num_features = X.shape[1]
        result_grid = {}

        # make models with cases

        X.sort_index(inplace=True)
        y.sort_index(inplace=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1234)


        for i in range(n_random_search):

            num_hidden_layers = np.random.randint(1,4)
            hidden_layers = list(np.repeat(1,num_hidden_layers))

            if num_hidden_layers == 1:
                hidden_layers[0] = num_features * np.random.randint(2, 5)
            elif num_hidden_layers == 2:
                hidden_layers[0] = num_features * np.random.randint(3, 6)
                hidden_layers[1] = num_features * np.random.randint(1, 3)
            else:
                hidden_layers[0] = num_features * np.random.randint(3, 6)
                hidden_layers[1] = num_features * np.random.randint(1, 3)    
                hidden_layers[2] = int( num_features * (np.random.random() + 0.1))

            hidden_layers = tuple(hidden_layers)

            if i==0 or hidden_layers not in result_grid.values():
                clf = MLPClassifier(activation = 'relu',
                                    solver = 'adam',
                                    early_stopping = True,
                                    validation_fraction = 0.1,
                                    tol = 1e-4,
                                    random_state=1234,
                                    hidden_layer_sizes = hidden_layers,
                                    max_iter=1000)
                clf.fit(X_train,y_train)
                auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
                result_grid[auc] = hidden_layers
                print("          Hidden Layer : {}, AUROC : {:.4f}".format(hidden_layers, auc))

            else:
                print("          Same hidden layers repeated, Pass")


        max_auc = np.array(list(result_grid.keys())).max()
        best_layer = result_grid[max_auc]
        color_print("        Best Parameter for MLPClassifier : {}".format(best_layer), "LIGHTCYAN", bBold=False)
        color_print("        Processing : fitting with best parameters...", "LIGHTCYAN", bBold=False)

        clf = MLPClassifier(activation = 'relu',
                                solver = 'adam',
                                early_stopping = True,
                                validation_fraction = 0.1,
                                tol = 1e-4,
                                random_state=1234,
                                hidden_layer_sizes = best_layer,
                                max_iter=1000)

        clf.fit(X_train,y_train)
        auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])

        color_print("        Best params : {}, AUROC : {:.4f}".format(best_layer, roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])), "LIGHTCYAN", bBold=False)

        param_grid = DataFrame()

        for col, row in result_grid.items():
            r =pd.DataFrame([row]).T
            r.columns = ['HiddenLayer']
            r['auc'] = col
            param_grid = pd.concat([param_grid, r], axis=0)

        self.clf_ = clf
        self.param_grid_ = param_grid

        return self




class leaderboard:

    ### full model fitting
    def fit(self, X_train, X_valid, y_train, y_valid, 
            model = ['rf', 'xgb', 'lgb', 'mlp'],
            param_rf = {'cv' : False, 'gridsearch' : False, 'n_random_search' : 10, 'n_cv_fold' : 5, 'n_jobs' : 10},
            param_xgb = {'learning_rate' : 0.02, 'early_stopping_rounds' : 100, 'cv' : False, 'gridsearch' : False, 'n_cv_fold' : 5, 'n_random_search' : 10, 'n_jobs' : 10},
            param_lgb = {'learning_rate' : 0.02, 'early_stopping_rounds' : 100, 'cv' : False, 'gridsearch' : False, 'n_cv_fold' : 5, 'n_random_search' : 10, 'n_jobs' : 10},
            param_mlp = {'n_random_search' : 10}
            ):
        
        """ Fit each model with train dataset, and fit ensemble / blending model with predicted value of validset.
          	  and find best performane model


            Parameters
            ----------
            
            X_train : pandas DataFrame 
      			      input features (train set)

            X_valid : pandas DataFrame 
      			      input features (valid set)

            y_train : pandas Series or DataFrame 
      			      target (train set)

            y_valid : pandas Series or DataFrame 
      			      target (valid set)
    			 
            model : list of models to be trained
            		* 'rf' : RandomForestClassifier
            		  'xgb' : Xgboost Classifier
            		  'lgb' : LightGBM Classifieir
            		  'mlp' : MLPclassifier

            params_* : parameters for each model


            Returns
            -------
    		preds : pandas DataFrame
    				predicted value on validset
    		models : dict
    			     trained models for each algorithm
    		leaderboard : pandas DataFrame
    					  performance result for each model
    		table : dict
    				distribution table for each models
        """   

        preds = DataFrame()
        preds_train = DataFrame()
        models = {}
        result_grid = {}

        X_train = X_train.sort_index()
        X_valid = X_valid.sort_index()
        y_train = y_train.sort_index() 
        y_valid = y_valid.sort_index()

        features = list(X_train.columns)


        if 'rf' in model:
            tool = fit_classifier()
            tool.fit_trees(X_train, y_train, classifier='rf', **param_rf)
            clf_rf = tool.clf_
            pred_rf = clf_rf.predict_proba(X_valid)[:,1]
            pred_train_rf = clf_rf.predict_proba(X_train)[:,1]
            preds['rf'] = pred_rf
            preds_train['rf'] = pred_train_rf
            models['rf'] = clf_rf
            result_grid['rf'] = tool.param_grid_
            # visualize_featureImportance(clf_rf, features, 'rf')

        if 'xgb' in model:
            tool = fit_classifier()
            tool.fit_trees(X_train, y_train, classifier='xgb', **param_xgb)
            clf_xgb = tool.clf_
            pred_xgb = clf_xgb.predict_proba(X_valid)[:,1]
            pred_train_xgb = clf_xgb.predict_proba(X_train)[:,1]
            preds['xgb'] = pred_xgb
            preds_train['xgb'] = pred_train_xgb
            models['xgb'] = clf_xgb
            result_grid['xgb'] = tool.param_grid_
            # visualize_featureImportance(clf_xgb, features, 'xgb')
  
        if 'lgb' in model:
            tool = fit_classifier()
            tool.fit_trees(X_train, y_train, classifier='lgb', **param_lgb)
            clf_lgb = tool.clf_
            pred_lgb = clf_lgb.predict_proba(X_valid)[:,1]
            pred_train_lgb = clf_lgb.predict_proba(X_train)[:,1]
            preds['lgb'] = pred_lgb
            preds_train['lgb'] = pred_train_lgb
            models['lgb'] = clf_lgb
            result_grid['lgb'] = tool.param_grid_
            # visualize_featureImportance(clf_lgb, features, 'lgb')       

        if 'mlp' in model:
            tool = fit_classifier()
            tool.fit_MLPClassifier(X_train, y_train, **param_mlp)
            clf_mlp = tool.clf_
            pred_mlp = clf_mlp.predict_proba(X_valid)[:,1]
            pred_mlp_train = clf_mlp.predict_proba(X_train)[:,1]
            preds['mlp'] = pred_mlp
            preds_train['mlp'] = pred_mlp_train
            models['mlp'] = clf_mlp
            result_grid['mlp'] = tool.param_grid_


        ## ensemble

        color_print("== Fitting Ensemble / Blending ==", "OKGREEN", bBold=True)

        preds['ensemble_smean'] = preds.sum(axis=1) / preds.shape[1]
        
        hmean = []
        for i in range(preds.shape[0]):
            row = preds.iloc[i,:-1]
            r = reduce(lambda x,y:x*y, row) ** (1/preds.shape[1])
            hmean.append(r)
        preds['ensemble_hmean'] = hmean
        
        ## stacking
        
        stacker = fit_classifier()
        stacker.fit_trees(preds_train, y_train, classifier='xgb')
        clf_stk_xgb = stacker.clf_
        pred_stk_xgb = clf_stk_xgb.predict_proba(preds.iloc[:,:-2])[:,1]
        preds['stacking_xgb'] = pred_stk_xgb
        models['stacking_xgb'] = clf_stk_xgb
        
        clf_stk_logit = LogisticRegression()
        clf_stk_logit.fit(preds_train,  y_train)
        pred_stk_logit = clf_stk_logit.predict_proba(preds.iloc[:,:-3])[:,1]
        preds['stacking_logit'] = pred_stk_logit
        models['stacking_logit'] = clf_stk_logit

        preds.index= X_valid.index

        # leaderboard

        leaderboard = DataFrame()
        cols = list(preds.columns)
        table = {}

        for m in cols:

            ks, auc, lift, table_performance = performance(y_valid, preds.loc[:,m])
            r = DataFrame({"ks" : ks, "auc" : auc, "lift" : lift}, index=[m])
            leaderboard = pd.concat([leaderboard, r], axis=0)
            table[m] = table_performance

        leaderboard.sort_values(by="ks", ascending=False, inplace=True)

        self.models_ = models
        self.leaderboard_ = leaderboard
        self.table_ = table
        self.result_grid_ = result_grid

        return self
