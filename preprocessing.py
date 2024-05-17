import numpy as np
import pandas as pd
from scipy import linalg
import category_encoders as ce
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
from unsupervised import pick_pca_transform
import missingno as msno
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.feature_selection import (mutual_info_classif, mutual_info_regression, SelectFromModel,
                                       RFECV, SequentialFeatureSelector, f_classif, f_regression, SelectPercentile)
from sklearn.model_selection import StratifiedKFold
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import (IsolationForest, RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier,
                              ExtraTreesRegressor, AdaBoostClassifier, AdaBoostRegressor)
import tkinter as tk
from tkinter import ttk
from skrebate import MultiSURF, ReliefF, SURF
import xgboost as xgb


# Ahead of utilizing the missingno package for correlation of missingness:
#
# *** A NOTE ON MISSING VALUES *** ------------------------------------------------------------------------------------
#
# Missing At Random (MAR): Missingness is not related to the value of the missing observation, but is connected to
# the value of an observed variable. This type can be imputed easily.
#
# Missing Completely At Random (MCAR): Missingness has nothing to do with the missing value or observed values of
# the other variables. In this case, we can only analyze features with complete data, so assuming this without basis
# is a poor decision.
#
# Missing Not at Random (MNAR): Missingness is related to the presumed value of the missing observation. For example,
# this would occur when a rainfall sensor breaks due to extreme rainfall and reports no reading.
#
# *** END *** ---------------------------------------------------------------------------------------------------------

def start_nan_chopper(data, percent_thresh, killrows, explore=False):
    if explore is True:
        msno.matrix(data)
        plt.show()

        # for heatmap
        # -1 indicates that if one variable appears the other is likely to be missing (MAR)
        # 0 indicates that there is no dependence between occurrence of missing values of two variables
        # 1 indicates that if one variable appears then the other is likely to be present as well.
        # these are pairwise relationships

        msno.heatmap(data)
        plt.show()

        # for dendrogram
        # hierarchical clustering algo to bin variables by nullity correlation (measured by binary distance)
        # at each step in tree, variables are split based on which combination minimizes distance of clusters

        # INTERPRETATION:
        # Read top to bottom:

        # Cluster leaves linked together at distance of ZERO fully predict one another's presence
        # MEANING one variable might always be empty when the other is filled, or they might both be filled/empty

        # If split CLOSE TO ZERO, the two features predict one another well but imperfectly. If they SHOULD match
        # in nullity, the leaf height represents how often the records are mismatched

        msno.dendrogram(data)
        plt.show()

    # Kill rows with NaN in specified columns
    if killrows is not None:
        killcount = 0
        for j in killrows:
            for i in data.index:
                if str(data.loc[i][j]) == str(np.nan):
                    killcount += 1
                    data.drop(i, inplace=True)
        print('You killed %.1f rows' % killcount)

    indexes = data.index
    columns = data.columns

    # removing vars with greater than 60% nan presence
    nan_bad = []
    nan_good = []
    for i in columns:
        working = data[i]
        nan_count = sum(working.isna())
        n = len(indexes)
        nan_percent = nan_count / n
        if explore is True:
            print('The NaN percentage of column %s is %.1f' % (i, nan_percent))
        if nan_percent > percent_thresh:
            nan_bad.append(i)
        else:
            nan_good.append(i)

    for i in nan_bad:
        data = data.drop(i, axis=1)

    return data


def explained_var(data):
    cov = np.cov(data, rowvar=False)
    ev = np.linalg.det(cov)
    return ev


def scat_coef(data):
    # converting to df to make column deletion easier with .drop()
    data = pd.DataFrame(data)
    # removing columns with only one unique values, then converting df back to ndarray
    dfdata = data.drop(data.columns[data.nunique() == 1], axis=1)
    new_feats = dfdata.columns
    data = np.array(dfdata)
    corr = np.corrcoef(data, rowvar=False)
    scat = np.linalg.det(corr)
    return scat, corr, new_feats, dfdata


def eig_eval(data):
    (scat, corr, new_feats, dfdata) = scat_coef(data)
    # symmetrizing correlation matrix, becomes asymmetrical due to floating point rounding errors
    corr = (corr + np.transpose(corr)) / 2
    # only proceed if the correlation matrix is symmetric, def= matrix - matrix''' == [bunch of zeros]
    check = corr - np.transpose(corr)
    if len(check[check != 0]) > 0:
        return print('The correlation matrix is not symmetrical')
    # calculate eigenvalues of the symmetrical correlation matrix
    eig_val = np.array(linalg.eigvalsh(corr))
    # some eigenvalues are negative, but with magnitude 1*10^-13
    # these are zero, but treated as otherwise by the algorithm, so make them actually zero
    eig_val[eig_val < 0] = 0
    # sort eigenvalues in descending order
    eig_val = eig_val[::-1]
    # Gauge PCA usefulness with metrics that can tolerate eigenvalues of 0
    # calculate psi index, where def=sum((eig-1)^2)
    psi = np.sum((eig_val - 1) ** 2)
    print('The psi index of this dataset is %.4f' % psi)
    # calculate information statistic, where def=-0.5*sum(ln(eig))
    # first, prep data for log operation
    eig_val[eig_val == 0] = 10 ** -10
    infostat = -(1 / 2) * np.sum(np.log(eig_val))
    print('The information statistic of this dataset is %.4f' % infostat)
    return


def vif_check(data, thresh=15):
    # first, drop columns where there is only one unique value, as it impedes the VIF calculation,

    df = data.drop(data.columns[data.nunique() == 1], axis=1)
    new_feats = df.columns

    # calculate the VIF score for each feature in the set

    vif_scores = pd.DataFrame()
    vif_scores['Feature'] = new_feats
    vif_scores['VIF'] = [variance_inflation_factor(df.values, i) for i in range(len(new_feats))]

    # perform this recursively until all features have a score less than 15

    final_vif, df = recursive_vif(vif_scores, df, thresh)
    return df


def recursive_vif(vif_scores, df, thresh=15):
    # the secondary sort by feature lexicographically makes it so low cardinality features will be chosen first in
    # cases where the maximum score is a tie.

    super_columns = list(vif_scores[vif_scores['VIF'] > thresh].sort_values(by=['VIF', 'Feature'],
                                                                            ascending=[False, True])['Feature'])

    # drop the column with the highest VIF score from the dataframe

    df.drop(super_columns[0], axis=1, inplace=True)

    # recalculate the VIF scores for the new dataframe

    new_vif = pd.DataFrame()
    new_vif['Feature'], new_vif['VIF'] = df.columns, [variance_inflation_factor(df.values, i) for i in
                                                      range(len(df.columns))]

    # if there are still features with a score greater than 15, repeat the selection process

    if len(new_vif[new_vif['VIF'] > thresh]) > 0:
        return recursive_vif(new_vif, df, thresh)
    else:
        return new_vif, df


# perform basic EDA if required, change types to string or dt depending on intent
def type_fixer_upper(datadf, makestrcol, makedtcol, bigstats=False):
    if bigstats is True:
        # pandas info method gives the number of observations, features names, non-null count for each, and the dtype
        print(datadf.info(verbose=True))
        # pandas describe method gives summary statistics for
        print(datadf.describe())
        input('Have string column and datetime column arguments been set appropriately?')
    if makestrcol:
        for i in makestrcol:
            datadf[i] = datadf[i].astype(str)
    if makedtcol:
        for i in makedtcol:
            datadf[i] = pd.to_datetime(datadf[i])
    return datadf


# soon, add functionality for treating ordinally encoded variables as categorical (not imputed)
def missing_outlier_machine(df, target, ordinal_cats=None, ordinal_cats_weights=None, z=3, trim=True):
    cut_target = target.copy(deep=True)
    # this machine needs to be able to handle NaNs
    df_num_only = df.copy(deep=True)
    n = len(df)
    # separate nominal categorical features from the rest of the dataframe, so that they are not interpreted by loop
    # as a low or high cardinality
    # keep option for ordinal_cats to not exist, as model performance may improve when treated as nominal
    if ordinal_cats:
        ordinal_df = df[ordinal_cats]
        df.drop(ordinal_cats, axis=1, inplace=True)
        df_num_only.drop(ordinal_cats, axis=1, inplace=True)
        # you can encode these immediately, as we are acting under the assumption that ordinally encoded
        # categorical variables can be treated as continuous. And are therefore viable for imputation!
        # BUT this relies on us inputting the respective category weights beforehand
        enc = OrdinalEncoder(categories=ordinal_cats_weights, handle_unknown='use_encoded_value', unknown_value=np.nan)
        ordinal_df = pd.DataFrame(enc.fit_transform(ordinal_df), index=ordinal_df.index, columns=ordinal_df.columns)

    # instantiate storage structure for high/low cardinality categorical features, numerical features,
    # and potential outliers
    hi_card_cat_cols = []
    lo_card_cat_cols = []
    num_cols = []
    cut_cols = pd.DataFrame()
    potent_outlier_count = []
    for i in df.columns:
        # extract each feature as a pd series and infer its type with API function
        col_series = df[i]
        cur_type = pd.api.types.infer_dtype(col_series)
        if cur_type == 'string' or cur_type == 'datetime64' or cur_type == 'mixed' or cur_type == 'categorical':
            # treat types of string, datetime, mixed, or categorical as categorical variables
            if len(col_series.unique()) > 5:
                # if they have more than 15 unique categories, an encoding strategy like target or catboost encoding
                # will prove to be more useful, so extract column name to a high_cardinality storage array
                hi_card_cat_cols.append(i)
            else:
                # otherwise, one hot encoding should be sufficient, so store these features in a separate array.
                lo_card_cat_cols.append(i)
            # for categorical features, we cannot calculate outliers with isolation forest before encoding
            # I'd rather not consider categorical outliers, because smaller instances of some classes may
            # prove to be useful for analysis
            # so temporarily drop these from a copy of the dataframe and add them to a new dataframe with categorical
            # columns only
            df_num_only.drop(i, axis=1, inplace=True)
            cut_cols = pd.concat([cut_cols, col_series], axis=1)
        else:
            # for categorical outliers, calculate the z score.
            # if observations have a value for this feature with a score greater than 3,
            # add them as a nparray to potential outlier storage array
            num_cols.append(i)
            if trim:
                col_z = (col_series - col_series.mean()) / (col_series.std(ddof=0))
                potent_outlier_count.append(np.array(col_z[col_z > z].index))

    if trim:
        # with bincount, we can count the number of potential feature outliers (z > 3) for each observation
        out_count = np.bincount([j for i in potent_outlier_count for j in i])
        # arbitrarily decided that if an observation has two or more potential outliers,
        # it could be considered an anomaly
        # later on it would be better to represent this value as a percentage of total features,
        # but more research needed.
        contam = len(out_count[out_count >= 2]) / n
        print('The proposed contamination value based on feature outlier count of 2 and z score of 3 is: %.3f' % contam)

    # There will be no outliers in categorical columns, so while you're here replace NaN in them with a constant
    # string that will act as an additional category once encoded.
    # Additionally, since we do not want "missing" as a filled value for ordinal features, impute missing values with
    # the most frequent occurrence

    filled_cut_cols = pd.DataFrame((SimpleImputer(strategy='constant', fill_value='missing', missing_values=np.nan).
                                    fit_transform(cut_cols)), index=cut_cols.index, columns=cut_cols.columns)

    cut_cut_cols = filled_cut_cols.copy(deep=True)

    cut_num_only = df_num_only.copy(deep=True)

    # use isolation forest to detect anomalies, use the contamination value that we estimated with z score
    # this relies on the assumption that there are no categorical outliers:

    # for nominal cat, this is because isolation forest cannot handle strings

    # For ordinal, this is for the sake of missing value imputation later on, as rarer categories AKA extremes
    # may be associated with other numerical features. So, I want to avoid edge cases where the inclusion of an
    # ordinally encoded feature pushes an observation to be considered an anomaly, due to the fact that its category
    # is rare.

    if trim:
        if contam < 0.001:
            print('The predicted contamination value for numerical features is extremely low, '
                  'which means that numerical outliers are '
                  'not prevalent in the set.')
            df_new = pd.concat([cut_num_only, cut_cut_cols], axis=1)
        else:
            # to reduce impact of missing values on anomaly detection, impute them with the mean
            # later on, it may be useful to use the median for some instances instead
            temp_fill = SimpleImputer(strategy='mean', missing_values=np.nan).fit_transform(df_num_only)
            detector = IsolationForest(n_estimators=100, contamination=contam).fit_predict(temp_fill)

            # save anomaly indices and drop them from all dataframes (categorical, numerical, AND target)

            indices = [int(x) for x in np.argwhere(detector < 0)]
            cut_num_only.drop(indices, axis=0, inplace=True)
            cut_cut_cols.drop(indices, axis=0, inplace=True)
            cut_target.drop(indices, axis=0, inplace=True)
            if ordinal_cats:
                ordinal_df.drop(indices, axis=0, inplace=True)
            # add ordinally encoded features back to numerical, with missing values (interpreted as outside of
            # categories given) included for later imputation.
            cut_num_only = pd.concat([cut_num_only, ordinal_df], axis=1)
            df_new = pd.concat([cut_num_only, cut_cut_cols], axis=1)
    else:
        # no row indices dropped, but categorical features have added 'missing' category and ordinal features have
        # been imputed with the most frequent category and then integer encoded by the weights array
        if ordinal_cats:
            cut_num_only = pd.concat([cut_num_only, ordinal_df], axis=1)
        df_new = pd.concat([cut_num_only, cut_cut_cols], axis=1)

    return df_new, cut_target, hi_card_cat_cols, lo_card_cat_cols


# things you can do prior to splitting data
def basic_pre_processing(df, ordinal_cats, ordinal_cats_weights, killrows_nan=None, percentthresh_nan=0.7,
                         target_cols=None, EDA=False, trim=True):
    # Step 1: repeated observations

    # Drop duplicates and irrelevant observations
    # The irrelevant observations part is subjective, depending on the purpose of analysis
    # looking at certain subpopulation for example

    df = df.drop_duplicates()

    # Step 1.1: type conversion

    # turn numerical data that is supposed to be datetime to that
    # turn certain columns to string if they are treated as 'object' currently
    # the function displays an example of each feature's data and the coded type

    df = type_fixer_upper(df, makestrcol=None, makedtcol=None, bigstats=EDA)

    # -----------------------------------------------------------------------------------------------------------------

    # Step 1.3: target extraction

    # do not want to include labels in imputation process, or its setup.
    # The model must have no knowledge of the right answer at any point

    # also remove rows where the target variable is missing, cannot impute that unfortunately

    if target_cols:
        target = df[target_cols]
        df = df[~target.isna()]
        target = pd.DataFrame(target[~target.isna()].reset_index(drop=True))
        df = df.drop(target_cols, axis=1).reset_index(drop=True)
    else:
        target = None

    # -----------------------------------------------------------------------------------------------------------------

    # Step 2: Big Picture Processing

    # Deal with features that may not be informative due to abundance of missing values: NaN % threshold set by param
    # Additionally, killrows argument allows for observations to be deleted if they have missing values in specific cols
    # e.g. in study of obese rats weight is missing, may as well toss the whole observation

    # Nifty arg: explore!
    # First, it uses missingno package to analyze "missingness" by feature
    # three graphs will be shown sequentially, see nan_chopper function for interpretation
    # default for killrows is None and 0.7 for percent_thresh

    df = start_nan_chopper(df, explore=EDA, killrows=killrows_nan, percent_thresh=percentthresh_nan)

    # -----------------------------------------------------------------------------------------------------------------

    # Step 3: Cardinality Classification and Anomaly Detection

    # missing_outlier_machine function methodology:

    # 3.1. Removes ordinal categorical variables and encodes them based on predefined inputs, retaining missing values.

    # 3.2. For the remaining features, type is determined with pandas API. By this type, it is concluded whether the
    # variable is numerical or categorical.

    # 3.31. For categorical features, the number of unique categories allows us to label their cardinality as low
    # or high. Either way, we drop them from a copy of the non-ordinal dataset

    # 3.32. For numerical features, which should all be continuous, the z score is calculated. Based on a pre-designated
    # z score threshold, observations of potential outliers in a given feature are added to a storage np array.

    # 3.4. If we are looking to trim the dataset, for each observation, tally the number of features where its value
    # could be an outlier based on z score. Observations that have more than two potential outliers contribute to
    # the contamination value for Isolation Forest anomaly detection.

    # 3.5. In the separated dataset with nominal categorical features ONLY, replace np.nan with a new category: missing

    # 3.6. Next, based on if the proposed contamination value is large enough, engage in Anomaly Detection protocol for
    # numerical features. This first requires missing values to be temporarily imputed with the mean,
    # which has an additive effect of strengthening the center of the distribution. Then, execute isolation forest
    # with the proposed contamination value. This will give us the indices of observations considered to be anomalies.
    # Remove them from all dataframes if the trim argument holds True.

    # ***A NOTE ON ISOLATION FOREST FOR ANOMALY DETECTION**************************************************************

    # Functions similarly to Random Forest, which is a collection of binary decision trees (BDTs). In this, the root
    # node is split based on "questions" posed by dataset features.
    #
    # This process begins with bagging (bootstrap aggregation), which creates multiple subsets of the training data
    # by randomly selection W/ REPLACEMENT. A singular BDT is trained on all subsets, and the predictions
    # (class or regression value for each point) are averaged to produce the final decisions.
    #
    # Another key feature here is the random selection of features prior to splitting. For Random Forest, it is a
    # subset of features initially selected, while in Isolation Forest a random feature axis AND threshold value for it
    # are chosen with each split.
    #
    # For a root node on a particular tree in the Isolation Forest, spliting is simple as the randomly selected
    # threshold value becomes the criteria for observations to travel down one path or the other. Random forest however,
    # is more complicated in this regard. More on that in supervised.py. Each split will result in observations
    # "traveling" down branches of the tree.
    #
    # After splitting has occurred, pre- and post- pruning take place.
    #
    # Pre-pruning is determined by the hyperparameters max_depth, min_samples_leaf, and min_samples_split as
    # early-stopping mechanisms. These are as simple as they sound, determining how many samples are needed to perform
    # a split, the minimum number of samples that constitute a stable leaf, and the max number of splits.
    #
    # In contrast, post-pruning occurs AFTER the tree has split to full depth, in which branches are removed to prevent
    # overfitting. To do this, the tree is partitioned into smaller branch subsets until the decisions made by it are
    # similar in terms of the decisions reached for each observation (removing redundant/uninformative splits). The
    # relevant hyperparameter here is ccp_alpha.
    #
    # So in Isolation Forest, this splitting by random feature axes occurs until full depth. Outliers/anomalies are
    # more likely to occur at an earlier depth compared to full depth. This proportionality is reflected through the
    # anomaly score for each point. Anomaly scores are then averaged across the all estimators (BDTs) to reach a
    # conclusion. This algorithm hinges on the contamination hyperparameters, which gives the machine a rough idea
    # of how many anomalies to expect in n observations.
    #
    # *** NOTE ON ISOLATION FOREST CONCLUDED **************************************************************************
    #
    # 3.7. Finally, concatenate the following: numerical features, ordinally encoded features, and nominal features
    # with the new 'missing' category. Return this and the target variable, along with the cardinality classifications.

    new_df, new_target, hi_card_cat, lo_card_cat = (
        missing_outlier_machine(df, ordinal_cats=ordinal_cats,
                                ordinal_cats_weights=ordinal_cats_weights, target=target, trim=trim))

    print('These features will be considered as high cardinality categorical for catboost encoding', hi_card_cat)
    print('These features will be considered as low cardinality categorical for one hot encoding', lo_card_cat)

    return new_df, new_target, lo_card_cat, hi_card_cat


def add_imputer(choice_string, df):
    if choice_string == 'MICE':

        # Multiple Imputation through Chained Equations is complex, yet the most effective method of filling NaN.

        # ***KEY ASSUMPTION***
        # The missing values are Missing At Random (MAR), meaning that the probability of missingness is dependent on
        # observed values, not unobserved ones. (a note on this later)

        # Step A. Execute the initial strategy, which is filling np.nan with the mean/median/mode of that feature.

        # Step B. For ONE feature, the placeholder values are switched back to np.nan.

        # Step C. The feature in question then is treated as the dependent variable in a regression function, and the
        # missing value is predicted based on all other variables in the set being predictors. The default setting uses
        # Ridge Regression (more on this later)

        # Step D. Repeat steps B and C for all features with missing values, in order of most missing to least. This is
        # ONE cycle of the MICE algorithm. Repeat this process for a designated number of cycles (paramater max_iter)
        # or until convergence is met (defined by parameter tol).

        # since the estimators for MICE are ridge regression and KNN Regression, normalization of the dataset
        # is required prior to use

        feat_norm = FeatureNormalizer()
        temp_imputer = IterativeImputer(random_state=3)
        imputer = clone(temp_imputer)
        unfit_pipe = Pipeline([('norm', feat_norm), ('imp', imputer)])
        temp_pipe = Pipeline([('norm', feat_norm), ('quick_imp', temp_imputer)])
    elif choice_string == 'KNN':
        # KNN is a distance based method, so all feature values must be normalized prior to use
        feat_norm = FeatureNormalizer()
        some_k = int(np.sqrt(len(df)) // 2)
        temp_imputer = KNNImputer(n_neighbors=some_k)
        imputer = clone(temp_imputer)
        unfit_pipe = Pipeline([('norm', feat_norm), ('imp', imputer)])
        temp_pipe = Pipeline([('norm', feat_norm), ('quick_imp', temp_imputer)])
    elif choice_string == 'Zeros':
        temp_imputer = SimpleImputer(strategy='constant', fill_value=0)
        imputer = SimpleImputer(strategy='constant', fill_value=0)
        unfit_pipe = Pipeline([('imp', imputer)])
        temp_pipe = Pipeline([('quick_imp', temp_imputer)])
    else:
        # this should only be engaged if the analysis algorithm can handle or is aided by missing values
        unfit_pipe = Pipeline([])
        temp_pipe = Pipeline([])
    return unfit_pipe, temp_pipe


def add_selector(choice_string, test_df, cat_transformed_cols, unfit_pipe, discrete_indices, cat_target, impute_choice,
                 soft=False, norm=True):
    # add normalization if the feature selection method relies on Euclidean distance
    # after encoding categorical variables and imputing missing values, add a normalization step for the case of
    # Relief or Information gain based selection, which rely on nearest neighbor determination for feature
    # selection. UNLESS, the imputing method is KNN, in which a normalization step has already been applied.

    if (choice_string == 'PCA' or choice_string == 'Recursive VIF' or choice_string == 'F Score') and norm:

        unfit_pipe.steps.append(['min_max_scale', StandardScaler()])

    elif (impute_choice != 'KNN' and impute_choice != 'MICE') and norm:

        # this is insufficient, Normalizer() by default normalizes by observation rather than by feature
        # to avoid information loss, create a custom transformer that uses preprocessing.normalize(axis=1)
        # We want the FEATURE vector to have a unit length of 1

        unfit_pipe.steps.append(['norm_by_feat', FeatureNormalizer()])

    if soft:
        sp_thresh = 40
        vif_thresh = 25
        sfm_thresh = 'mean'
    else:
        sp_thresh = 25
        vif_thresh = 10
        sfm_thresh = '1.5*mean'

    if choice_string == 'PCA':

        # in the case of PCA, use your own function to decide a good number of principal components to use based on
        # the amount of variance explained. This method of feature extraction is useful when multicollinearity is
        # intolerable by a predictive analysis method. Then, add a PCA transformer to the pipeline.

        add_to_pipe = pick_pca_transform(test_df)
        unfit_pipe.steps.append(['pca', add_to_pipe])


    # *** FILTER BASED METHODS -----------------------------------------------------------------------------------

    elif choice_string == 'Recursive VIF':

        # This uses variance inflation factor as a means to reduce the amount of correlated variables in the set.
        # The calculation involves running an ordinary least squares regression on the dataset, where a given
        # feature is represented as a function of all other variables in the set.
        #
        # The key value is the r^2 of this regression:
        # VIF = 1 / (1 - R^2)
        #
        # VIF values greater than 15 are considered to be highly correlated with the other features in the set.
        # Recursively, the highest VIF variables are sequentially removed from the set until all of them have a
        # score less than 15. In the case where the highest VIF score is shared by two or more features,
        # low cardinality features are prioritized.
        vif_selected_df = vif_check(test_df, thresh=vif_thresh)
        vif_select = vif_selected_df.columns

        # Here, I created a custom sklearn transformer that converts the output nparray to a dataframe, removes
        # named columns excluded based on their VIF score, and then converts back to nparray for use in later
        # analysis algorithms. This is useful for the purposes of feature selection, where we would like to get
        # rid of features based on their name. Add this VIF selection transformer to the pipeline.

        retain_names = SelectPDColumns(columns=cat_transformed_cols, want_columns=vif_select)
        unfit_pipe.steps.append(['vif_exclude', retain_names])


    elif choice_string == 'Information Gain':

        # set a reasonable value of neighbors for KNN, explained more below.

        k = int(np.sqrt(len(test_df)) // 2)

        # The second feature selection method I have included is based on information gain, an entropy based
        # calculation that many modern ML algorithms rely on.

        # *** A NOTE ON ENTROPY, INFORMATION  GAIN, AND KNN *** ---------------------------------------------------

        # Entropy is a measure of variance/disorder in a dataset based on its probability distribution:
        #
        # For a training set T: H(T) = -sigma( p(t)log(p(t) ). Where p(t) is the probability of choosing observation
        # t from the entire training set. More on how we create that distribution later.
        #
        # Conditional entropy is a modification on this equation which assumes that a given feature "a" is known:
        #
        # H(T|a) = sigma_for_all_v( (|S(a=v)|/|T|)*H(S(a=v)) ). Where v is a given value of feature a and S(a=v) is
        # the subset of the training data where feature a has a value of v. |T| and |S(a=v)| is the size of the
        # full training set and subset where feature a = v.
        #
        # With the concepts of both entropy and conditional entropy understood, we combine them to represent the
        # information gain of feature "a."
        #
        # IG(T, a) = H(T) - H(T|a)
        #
        # This can be interpeted as the following question: by knowing what the value of feature "a" is, how much do
        # we reduce the amount of disorder/variance in the training set? When entropy is reduced, it is a gain
        # in order and information. By calculating information gain for each variable, we can remove redundant ones
        # that do not provide much information.
        #
        # But, we are missing a key aspect of the entropy/IG calculation, how do we create a probability
        # distribution rom a high dimensional dataset? In sklearn's mutual info functions, the answer is through
        # using k-nearest neighbors as a model framework to create a distribution based on the target variable.
        #
        # In this, distance (tradtionally Euclidean, but can be Hamming) is used as a basis for point similarity
        # in a dataset. For p(t), the probability of coming across point t in dataset T, we assume that the target
        # class/value is unknown, then evaluate the target value of the k nearest points. With these values in
        # tow and the distance of each neighbor to point t, we give a likely estimate for the target value/class
        # based on weighted average or majority vote respectively. The error (MSE or right/wrong) in this estimate
        # compared to the actual value informs p(t)!
        #
        # *** NOTE ENDED *** --------------------------------------------------------------------------------------

        # Now, we either use mutual_info_classification or mutual_info_regression depending on the target. Sklearn
        # then selects the features that have an information gain value greater than that of the mean. This should
        # be tweaked in future versions of the pipeline. Add this selection step to the pipeline.

        if cat_target:
            best_ig = SelectFromModel(estimator=MutualInfoEstimatorClass(n_neighbors=k, random_state=3,
                                                                         discrete_features=discrete_indices,
                                                                         feature_names=cat_transformed_cols),
                                      threshold=sfm_thresh)

        else:
            best_ig = SelectFromModel(estimator=MutualInfoEstimatorRegress(n_neighbors=k, random_state=3,
                                                                           discrete_features=discrete_indices,
                                                                           feature_names=cat_transformed_cols),
                                      threshold=sfm_thresh)

        unfit_pipe.steps.append(['ig_exclude', best_ig])

    elif choice_string == 'F Score':

        if cat_target:
            corr_excl = SelectPercentile(score_func=f_classif, percentile=sp_thresh)
        else:
            corr_excl = SelectPercentile(score_func=f_regression, percentile=sp_thresh)

        unfit_pipe.steps.append(['f_exclude', corr_excl])

    elif choice_string == 'ReliefF':

        n_features = len(cat_transformed_cols)
        relief = ReliefF(n_features_to_select=n_features, n_neighbors=10, n_jobs=-1)
        relief_selec = SelectFromModel(estimator=relief, threshold=sfm_thresh)
        unfit_pipe.steps.append(['relief_f', relief_selec])

    elif choice_string == 'SURF':

        n_features = len(cat_transformed_cols)
        relief = SURF(n_features_to_select=n_features, n_jobs=-1)
        relief_selec = SelectFromModel(estimator=relief, threshold=sfm_thresh)
        unfit_pipe.steps.append(['surf', relief_selec])

    elif choice_string == 'MultiSURF':

        # we want all features to be scored,  but not from the original dataframe, rather the transformed one.

        n_features = len(cat_transformed_cols)

        # no need to determine number of neighbors to consider, as MultiSURF utilizes multi thresholding by creating
        # a similarity metric T for each observation i. This is calculated by taking the average pairwise distance
        # between i and all other observations. Any point within distance T of i is considered a neighbor and used to
        # update the weight array element for each feature.

        # This distance parameter T is the SU in SURF, spatially uniform! The multi- prefix comes from the fact that
        # distance parameter T is calculated for each observation rather than across all observations.

        # instantiate MultiSURF with all features selected, as we want SelectFromModel to make that decision instead
        relief = MultiSURF(n_features_to_select=n_features, n_jobs=-1)
        relief_selec = SelectFromModel(estimator=relief, threshold=sfm_thresh)
        unfit_pipe.steps.append(['multi_surf', relief_selec])

        # WRAPPER BASED METHODS ---------------------------------------------------------------------------------------

    elif choice_string == 'RFE':
        # super high computational load on its own...
        # so do not put a pipeline with it into a random grid search or ur computer will blow up
        # EITHER:
        # 1. fit prior to predictive model and just input selected features
        # 2. use it after a filter method (to be implemented later)
        if cat_target:
            feature_select_cv = StratifiedKFold(n_splits=4)
            wrapper_model = RandomForestClassifier()
            select_1 = RFECV(estimator=wrapper_model, scoring='accuracy', cv=feature_select_cv, n_jobs=-1)
            unfit_pipe.steps.append(['wrapper', select_1])

        else:
            wrapper_model = RandomForestRegressor()
            select_1 = RFECV(estimator=wrapper_model, scoring='neg_mean_squared_error', n_jobs=-1)
            unfit_pipe.steps.append(['wrapper', select_1])

    elif choice_string == 'Forward':
        # super high computational load on its own...
        # so do not put a pipeline with it into a random grid search or ur computer will blow up
        # EITHER:
        # 1. fit prior to predictive model and just input selected features
        # 2. use it after a filter method (to be implemented later)
        if cat_target:
            feature_select_cv = StratifiedKFold(n_splits=4)
            wrapper_model = RandomForestClassifier()
            select_1 = SequentialFeatureSelector(estimator=wrapper_model, scoring='accuracy', cv=feature_select_cv,
                                                 n_jobs=-1, tol=0.01)
            unfit_pipe.steps.append(['wrapper', select_1])

        else:
            wrapper_model = RandomForestRegressor()
            select_1 = SequentialFeatureSelector(estimator=wrapper_model, scoring='neg_mean_squared_error',
                                                 n_jobs=-1, tol=100)
            unfit_pipe.steps.append(['wrapper', select_1])

    elif choice_string == 'Backward':
        # super high computational load on its own...
        # so do not put a pipeline with it into a random grid search or ur computer will blow up
        # EITHER:
        # 1. fit prior to predictive model and just input selected features
        # 2. use it after a filter method (to be implemented later)
        if cat_target:
            feature_select_cv = StratifiedKFold(n_splits=4)
            wrapper_model = RandomForestClassifier()
            select_1 = SequentialFeatureSelector(estimator=wrapper_model, scoring='accuracy', cv=feature_select_cv,
                                                 n_jobs=-1, tol=0.01, direction='backward')
            unfit_pipe.steps.append(['wrapper', select_1])

        else:
            wrapper_model = RandomForestRegressor()
            select_1 = SequentialFeatureSelector(estimator=wrapper_model, scoring='neg_mean_squared_error',
                                                 n_jobs=-1, tol=100, direction='backward')
            unfit_pipe.steps.append(['wrapper', select_1])

    return unfit_pipe


def setup_pipeline(df, target, imp_choice='MICE', ML=False, decision=None, hi_cat_cols=None, lo_cat_cols=None,
                   cat_target=False, hybrid_plan=None):
    # this function initializes a sklearn pipeline for later machine learning analysis (RF, logit, LDA, etc.)

    # -----------------------------------------------------------------------------------------------------------------

    # 1. Establish two pipelines with the imputer of choice. One of which will be for temporary analysis for
    # feature selection. This function interpets choice string and instantiates sklearn pipelines with it. Requires
    # df as an input as well to determine an appropriate k for KNN-based imputation.

    unfit_pipe, temp_pipe = add_imputer(imp_choice, df)

    # -----------------------------------------------------------------------------------------------------------------

    # Now, add categorical feature encoding transformers. For features with many unique categories, one hot encoding
    # will incur the curse of dimensionality and turn the final dataframe into a sparse matrix. To avoid this, we will
    # use CatBoost encoding to turn these categorical variables into continuous ones.

    # *** A NOTE ON CATBOOST **** -------------------------------------------------------------------------------------

    # This method uses the target variable to encode a high cardinality categorical feature. The initial step is drawing
    # a random permutation order of the dataset.

    # Then, the target variable must be converted to an integer. This is simple for a classification problem, but
    # for regression quantization must be utilized to convert from float32 --> float16 --> int8. How do we get from
    # a float representation to one where only 256 values are possible? See below:

    # For a float range [a, b], consider float value X for affine quantization: X = S * (x - Z) where
    # x = int8 representation of float X
    # S = scale (float32)
    # Z = zero-point, AKA the int8 value corresponding to 0 in the float32 space
    # Rearranging yield: x = round(X/S + Z)
    # these integer values are then placed into buckets or bordered ranges, now represented similarly to an integer
    # encoded categorical variable.

    # With the target as an integer, we then iterate sequentially through observations. For each the statistic of
    # interest is calculated, which would look like this for a quantized target variable:

    # ctr(i) = (class_ct + prior)/(tot_ct + 1) where:
    # i is the current observation
    # class_ct = number of observations in the category of i where the integer target matches that of i
    # max_ct = total number of objects (UP TO CURRENT) that have a categorical value matching that of i
    # prior = constant

    # For classification problems Binarized Target Mean Value is used as the statistic of interest instead, which is
    # a pretty much identical equation

    # After the statistic of interest is calculated with each observation for the given permutation, multiple other
    # permutations are drawn and the results are averaged, just like an ensemble learning method.

    # *** END NOTE ON CATBOOST *** ------------------------------------------------------------------------------------

    # Add categorical encoding transformers to the pipeline if necessary:

    if hi_cat_cols and lo_cat_cols:
        catboost = ce.cat_boost.CatBoostEncoder()
        onehot = OneHotEncoder(drop='first')
        cat = ColumnTransformer([('high_card', catboost, hi_cat_cols), ('0_low_card', onehot, lo_cat_cols)],
                                remainder='passthrough', verbose_feature_names_out=True)
        unfit_pipe.steps.insert(0, ['cat', cat])
        temp_pipe.steps.insert(0, ['cat', cat])
        cat_cols = hi_cat_cols + lo_cat_cols
    elif hi_cat_cols:
        catboost = ce.cat_boost.CatBoostEncoder()
        cat = ColumnTransformer([('high_card', catboost, hi_cat_cols)],
                                remainder='passthrough', verbose_feature_names_out=True)
        unfit_pipe.steps.insert(0, ['cat', cat])
        temp_pipe.steps.insert(0, ['cat', cat])
        cat_cols = hi_cat_cols
    elif lo_cat_cols:
        onehot = OneHotEncoder(drop='first')
        cat = ColumnTransformer([('0_low_card', onehot, lo_cat_cols)],
                                remainder='passthrough', verbose_feature_names_out=True)
        unfit_pipe.steps.insert(0, ['cat', cat])
        temp_pipe.steps.insert(0, ['cat', cat])
        cat_cols = lo_cat_cols
    else:
        cat_cols = []

    # -----------------------------------------------------------------------------------------------------------------

    # reordering dataframe such that categorical columns appear first, convenient for error testing.

    if cat_cols:
        num_cols = df.columns[~df.columns.isin(cat_cols)]
        new_order = list(cat_cols) + list(num_cols)
        df = df[new_order]

    # -----------------------------------------------------------------------------------------------------------------

    # If the amount of NaNs is manageable, shotgun imputation should not impact multi-collinearity check results
    # too much. However, this is a critical assumption of the pre-processing model. By shotgun imputation, it is meant
    # that a minimally tuned version of the imputer and categorical encoder are applied to the dataset so that it is
    # usable for feature selection.

    if ML:
        # impute missing values with the chosen imputer and encode categorical variables. Here, it is key to store the
        # feature names of the encoded variables, as the default output of sklearn transformers is an unlabeled nparray.

        shotgun = temp_pipe.fit(df, target)
        new_names = shotgun['cat'].get_feature_names_out()

        # Also, one hot encoded variables should be treated as discrete, not continuous like with catboost or ordinally
        # encoded ones. Store these for later use in calculating information gain, as sklearn's MutualInfo functions
        # perform poorly when they consider all features to be either discrete or continuous.

        discrete_indices = pd.Series(new_names)[pd.Series(new_names).str.startswith('0_low_card')].index.to_numpy()
        shotgunned = temp_pipe.transform(df)
        test_this = pd.DataFrame(shotgunned, columns=new_names, index=df.index)

        # choose feature selection method based on ml_method string
        if decision == 'Hybrid':
            filter_choice = hybrid_plan[0]
            unfit_pipe = add_selector(filter_choice, test_this, new_names, unfit_pipe, discrete_indices,
                                      cat_target, impute_choice=imp_choice, soft=True)
            wrapper_choice = hybrid_plan[1]
            unfit_pipe = add_selector(wrapper_choice, test_this, new_names, unfit_pipe, discrete_indices, cat_target,
                                      norm=False, impute_choice=imp_choice)
        else:
            unfit_pipe = add_selector(decision, test_this, new_names, unfit_pipe, discrete_indices, cat_target,
                                      impute_choice=imp_choice)

        return unfit_pipe, hybrid_plan

    else:
        return unfit_pipe, hybrid_plan


# create transformer for use in pipeline that normalizes by feature vector rather than observation
# sklearn's Normalizer() does not meet this need.

class FeatureNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self, norm='l2', copy=True):
        self.norm = norm
        self.copy = copy

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # we need this function to tolerate NaN values, so sklearn default normalization functions will not work
        # furthermore, note that the input to this transformation is a numpy array

        # contrary to df.drop(axis=1), where the drop is done on the column, remember that operations are performed in
        # the DIRECTION of the axis rather than on the axis parameter itself. So, for normalization down a column,
        # the np.nansum() must be performed on axis=0, summing the column values down the row AXIS.

        # Just a quick reminder!

        if self.norm == 'l2':
            scale = np.sqrt(np.nansum(X**2, axis=0))
        elif self.norm == 'l1':
            scale = np.nansum(np.abs(X), axis=0)
        else:
            scale = np.nanmax(np.abs(X), axis=0)

        # new axis here is used to increase the dimension of the scale nparray to 2D instead of 1D, such that X
        # can be divided by it.
        normed = X / scale[np.newaxis, :]
        print('The feature vectors have been normalized.')

        return normed


# mutual information estimator cannot be called without an input training set, so create a custom transformer so that
# SelectFromModel can be added to pipeline and work upon fitting. If we don't do this, mutual information will be
# calculated for the set prior to categorical encoding, which will be incompatible with the transformed set.


class MutualInfoEstimatorClass(BaseEstimator):
    def __init__(self, discrete_features='auto', n_neighbors=3, random_state=None, copy=True, feature_names=None):
        self.discrete_features = discrete_features
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.copy = copy
        self.feature_names = feature_names
        # temporary assignment before fit fills with actual feature importance
        self.feature_importances_ = None

    def fit(self, X, y):
        print('Information gain is running')
        self.feature_importances_ = mutual_info_classif(X, y, discrete_features=self.discrete_features,
                                                        n_neighbors=self.n_neighbors,
                                                        random_state=self.random_state, copy=self.copy)


class MutualInfoEstimatorRegress(BaseEstimator):
    def __init__(self, discrete_features='auto', n_neighbors=3, random_state=None, copy=True, feature_names=[]):
        self.discrete_features = discrete_features
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.copy = copy
        self.feature_names = feature_names
        # temporary assignment before fit fills with actual feature importances
        self.feature_importances_ = None

    def fit(self, X, y):
        print('Information Gain is running')
        self.feature_importances_ = mutual_info_regression(X, y, discrete_features=self.discrete_features,
                                                           n_neighbors=self.n_neighbors,
                                                           random_state=self.random_state, copy=self.copy)


# this is a transformer that takes in the full dataframe column names and desired column names from VIF selection
# to output a numpy array structure of the vif selected version. For use after MICE imputation removes column labels
# from the categorically encoded dataset


class SelectPDColumns(BaseEstimator, TransformerMixin):
    # initialization of transformer requires desired column names (vif_selected in our case)
    def __init__(self, columns, want_columns):
        self.columns = columns
        self.want_columns = want_columns

    # transformation method of transformer requires an input dataframe X, where selected columns are those specified in
    # the initialization of the estimator

    def fit(self, nump, y=None):
        return self

    def transform(self, nump, y=None):
        full_frame = pd.DataFrame(nump, columns=self.columns)
        select_frame = full_frame[self.want_columns].copy(deep=True).to_numpy()
        print('Below are the VIF score selected features:')
        print(self.want_columns)
        return select_frame

def selector_params_to_grid(choice_string, cat_target, grid, trim):
    if choice_string == 'ReliefF':
        grid['relief_f__n_neighbors'] = [5, 10, 20, 40, 60, 80, 100, 500]
    elif choice_string == 'Forward' or choice_string == 'Backward' or choice_string == 'RFE':
        if cat_target:
            # for xgb, default is gradient boosted regression tree with hist tree method. This tree method is similar
            # to approx with binned values based on quantile sketch, but bins are reused rather than remade w/ each
            # iteration. Avoiding parameter optimization for now for the sake of computational cost.
            # only thing I will set is the objective,
            # softprob is the only viable option for multiclass solutions, so this will be default
            # for regression, use MSE if outliers have been excluded, otherwise use pseudo Huber which can tolerate them
            grid['wrapper__estimator'] = [RandomForestClassifier(), ExtraTreesClassifier(), AdaBoostClassifier(),
                                          xgb.XGBClassifier(objective='multi:softprob')]
        else:
            if trim is False:
                obj = 'reg:pseudohubererror'
            else:
                obj = 'reg:squarederror'
            grid['wrapper__estimator'] = [RandomForestRegressor(), ExtraTreesRegressor(), AdaBoostRegressor(),
                                          xgb.XGBRegressor(objective=obj)]
    return grid

def data_sanitation(data, target, percent_thresh=0.65, kill_row_if_nan_in=None, ordinal_cats=None,
                    ordinal_cats_order=None):
    # Add GUI to the function in order to avoid lengthy function call

    # need helper function to track when a dropdown menu option is changed
    # instatiate window title and size
    base = tk.Tk()
    base.title('Data Cleaning Tool')
    base.geometry('800x600')

    # title for window, use grid to center it at the top.
    ttk.Label(base, text='Select preferences for data processing:', font=('Times New Roman', 12)).grid(row=0, column=1)

    # first dropdown menu, use label and position it at the 0th column to leave room for menu
    ttk.Label(base, text='EDA?', font=('Times New Roman', 10)).grid(column=0, row=5, padx=10, pady=25)
    # establish string variable for Combobox widget
    n1 = tk.StringVar()
    option_eda = ttk.Combobox(base, width=27, textvariable=n1)
    # set dropdown menu options
    option_eda['values'] = ('ON', 'OFF')
    # position the menu using grid, this time in the 1st column
    option_eda.grid(column=1, row=5)
    # set default menu option
    option_eda.current(1)

    # second dropdown menu
    ttk.Label(base, text='Include Isolation Forest-identified anomalies?',
              font=('Times New Roman', 10)).grid(column=0, row=6, padx=10, pady=25)
    n2 = tk.StringVar()
    option_outliers = ttk.Combobox(base, width=27, textvariable=n2)
    option_outliers['values'] = ('Include', 'Exclude')
    option_outliers.grid(column=1, row=6)
    option_outliers.current(0)


    # fourth dropdown menu
    ttk.Label(base, text='Is the target categorical?', font=('Times New Roman', 10)).grid(column=0, row=7, padx=10,
                                                                                          pady=25)
    n4 = tk.StringVar()
    option_target = ttk.Combobox(base, width=27, textvariable=n4)
    option_target['values'] = ('Yes', 'No')
    option_target.grid(column=1, row=7)
    option_target.current(1)

    # fifth dropdown menu
    ttk.Label(base, text='How to impute missing values?', font=('Times New Roman', 10)).grid(column=0, row=8, padx=10,
                                                                                             pady=25)
    n5 = tk.StringVar()
    option_impute = ttk.Combobox(base, width=27, textvariable=n5)
    option_impute['values'] = ('MICE', 'KNN', 'Zeros', 'Leave them in')
    option_impute.grid(column=1, row=8)
    option_impute.current(2)

    # sixth dropdown menu
    ttk.Label(base, text='What feature selection/extraction method?', font=('Times New Roman', 10)).grid(column=0,
                                                                                                         row=9,
                                                                                                         padx=10,
                                                                                                         pady=25)
    n6 = tk.StringVar()
    option_ml = ttk.Combobox(base, width=27, textvariable=n6)
    option_ml['values'] = ('PCA', 'Recursive VIF', 'Information Gain', 'SURF', 'ReliefF', 'MultiSURF', 'F Score',
                           'Hybrid', 'Backward', 'Forward', 'RFE', 'None')
    option_ml.grid(row=9, column=1)
    option_ml.current(3)

    # seventh widget as an entry box for selecting hybrid feature selection methods
    ttk.Label(base, text="If hybrid, input selection w/ format 'Filter, Wrapper':",
              font=('Times New Roman', 10)).grid(column=0, row=10, padx=10, pady=25)
    n7 = tk.StringVar()
    entry_hybrid = tk.Entry(base, textvariable=n7, font=('Times New Roman', 10))
    entry_hybrid.grid(column=1, row=10)

    # submit button to close the window and proceed with the function

    exit_button = tk.Button(base, text='Start', command=base.destroy)
    exit_button.grid(column=1, row=11)

    base.mainloop()

    # parse the text inputs given by the dropdown menus as acceptable binary or string inputs for the functions.

    eda_call = n1.get()
    if eda_call == 'ON':
        eda_call = True
    else:
        eda_call = False
    trim = n2.get()
    if trim == 'Exclude':
        trim = True
    else:
        trim = False
    target_status = n4.get()
    if target_status == 'Yes':
        cat_target = True
    else:
        cat_target = False
    imputer = n5.get()
    ml_method = n6.get()
    if ml_method != 'None':
        ml_check = True
    else:
        ml_check = False

    # If we want a hybrid feature selection method, we will need the text from the entry box widget to let function
    # know the choice of filter and wrapper.
    if ml_method == 'Hybrid':
        filter_wrapper = n7.get()
        hybrid_plan = filter_wrapper.split(', ')
    else:
        hybrid_plan = None


    # check if weights have been assigned to the declared ordinal categorical variables

    if ordinal_cats and eda_call and not ordinal_cats_order:
        print("Decide the order for ordinal categorical variable encoding")
        for i in ordinal_cats:
            print(data[i].unique())
    df, targets, tb_onehot, tb_catboost = basic_pre_processing(data,
                                                               killrows_nan=kill_row_if_nan_in,
                                                               percentthresh_nan=percent_thresh,
                                                               target_cols=target,
                                                               EDA=eda_call,
                                                               ordinal_cats=ordinal_cats,
                                                               ordinal_cats_weights=ordinal_cats_order,
                                                               trim=trim)
    if cat_target is True:
        targets = OrdinalEncoder().fit_transform(targets)

    # Set up the sklearn processing pipeline according to the needs of the dataset

    pipe, filter_wrapper = setup_pipeline(df, targets, ML=ml_check, decision=ml_method, hi_cat_cols=tb_catboost,
                                          lo_cat_cols=tb_onehot, cat_target=cat_target, imp_choice=imputer,
                                          hybrid_plan=hybrid_plan)


    # additional feature: output a dictionary for tuned elements in the initial pipeline, so that they don't
    # have to be used in the predictive algo
    # For this to be possible, include options for non-MICE imputation

    if imputer == 'MICE':
        tols = [1e-1, 1e-2, 1e-3, 1e-4]
        max_iter = [3, 5, 8, 10, 15]
        initial_strategy = ['most_frequent', 'median', 'mean']
        # random forest regressor is probably the best one, but is computationally expensive!
        estimator = [BayesianRidge(), KNeighborsRegressor(n_neighbors=int(np.sqrt(len(df))))]
        tune_grid = dict()
        tune_grid['imp__tol'] = tols
        tune_grid['imp__max_iter'] = max_iter
        tune_grid['imp__initial_strategy'] = initial_strategy
        tune_grid['imp__estimator'] = estimator
    elif imputer == 'KNN':
        ks = [3, 10, int(np.sqrt(len(df)) // 2), int(np.sqrt(len(df)))]
        weights = ['uniform', 'distance']
        tune_grid = dict()
        tune_grid['imp__n_neighbors'] = ks
        tune_grid['imp__weights'] = weights
    else:
        tune_grid = dict()

    # include tunable elements for feature selection techniques as well
    if filter_wrapper is not None:
        for i in hybrid_plan:
            tune_grid = selector_params_to_grid(i, cat_target, tune_grid, trim)
    else:
        tune_grid = selector_params_to_grid(ml_method, cat_target, tune_grid, trim)

    # TO DO, add grid constructor loop for when feature selection method is hybrid

    # return the dataframe, targets, pre-processing pipeline, and initial parameter tuning grid

    # but before this, unravel target to avoid column vector error message that comes from passing pd.Series
    # values method gives 1d numpy array, ravel converts from column to row vector

    targets = targets.values.ravel()

    return df, targets, pipe, tune_grid
