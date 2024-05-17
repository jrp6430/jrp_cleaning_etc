import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import RandomizedSearchCV, RepeatedKFold, cross_val_score, GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline


# add iterative imputer as an option to every classifier you have
# the ideal pipeline is to determine outliers and exclude them as NaN, categorically encode string columns,
# and THEN performing iterative imputation
# the tricky part will be automating the multi-collinearity check
# ideally, we perform multiple different imputations,
# but in each do an ML check before proceeding to the classifier/regressor
# Need to balance data cleaning and not disrupting assumptions of the models we are using

# Random forest can over fit if the number of features exceeds the number of observations
# To remedy this, use a dimensionality reduction technique like PCA
# For gene expression set, 20k vars --> 184 vars

# before training MICE parameters with a random or grid search, instantiate the model with basic parameters and perform
# multi-collinearity check to make sure a PCA is not necessary before proceeding


def rf_rando_best_params(data_df, labels, pipe, grid, know_params, regressor=False):
    train_features, test_features, train_labels, test_labels = train_test_split(
        data_df, labels, test_size=0.2, random_state=3)
    cv = RepeatedKFold(n_splits=3, n_repeats=1, random_state=3)
    # set up pipeline without multi-collinearity check, as it is not required before random forest
    # for random forest, normalization is not necessary
    # if you run into issues with runtime, use the PCA product instead (set in preprocessing)

    if know_params:
        # imp_params = {'imp__max_iter': 3, 'imp__initial_strategy': 'most_frequent', 'imp__tol': 0.01}
        # pipe.set_params(**imp_params)
        if regressor is False:
            best = RandomForestClassifier(n_estimators=90, max_depth=2, min_samples_split=6, min_samples_leaf=1,
                                         bootstrap=True)
            crit = 'accuracy'
        else:
            best = RandomForestRegressor(n_estimators=400, max_depth=7, min_samples_split=2, min_samples_leaf=4,
                                         bootstrap=True, max_features='log2')
            crit = 'neg_mean_squared_error'
        pipe.steps.append(['rf', best])
        best = pipe
        scores = cross_val_score(best, train_features, train_labels, cv=cv, scoring=crit)
        print('The cross validated training score is %.2f with a spread of %.2f' % (np.mean(scores), np.std(scores)))
        best.fit(train_features, train_labels)
    else:
        if regressor is False:
            rf = RandomForestClassifier()
        else:
            rf = RandomForestRegressor()
        pipe.steps.append(['rf', rf])
        # optimizing hyperparameters
        # 1. random forest
        # number of trees ranging from 50 to 1000
        grid['rf__n_estimators'] = [int(x) for x in np.linspace(350, 550, 15)]
        # max number of features being sqrt(n_features), log2(n_features), or n_features
        # max tree depth ranging from 3 to 50
        max_depth = [int(x) for x in np.linspace(2, 10, 9)]
        max_depth.append(None)
        grid['rf__max_depth'] = max_depth
        # minimum number of samples required to split a node
        grid['rf__min_samples_split'] = [2, 4, 6, 8]
        # minimum number of samples for a leaf node to exist
        grid['rf__min_samples_leaf'] = [2, 4, 6, 8]
        # bootstrapping or no
        grid['rf__bootstrap'] = [True, False]
        # criterion
        if regressor is False:
            grid['rf__criterion'] = ['gini', 'entropy', 'log_loss']

        else:
            grid['rf__criterion'] = ['squared_error', 'absolute_error', 'friedman_mse', 'poisson']

        # max features
        grid['rf__max_features'] = ['sqrt', 'log2']

        # instantiating random grid
        rf_hyper = RandomizedSearchCV(estimator=pipe, param_distributions=grid, n_iter=25, cv=cv, verbose=2,
                                      random_state=3, n_jobs=-1)
        rf_hyper.fit(train_features, train_labels)
        print('The best parameters from cross validated random search are:', rf_hyper.best_params_)
        print('with a training score of %.2f' % rf_hyper.best_score_)
        best = rf_hyper.best_estimator_

    # evaluate the performance of the hyperparameter tuned model
    test_score = best.score(test_features, test_labels)*100
    print('The test score of this RF model is %.2f' % test_score)
    return


# LDA PERFORMS POORLY WHEN D >> N, that is why the RF model performs so much better
# Use LDA/QDA when n > 5 x D

def lda(data, labels, pipe, grid, classifier=True):
    # scaling data before modeling, as LDA assumes each input variable has the same variance (identical covar matrix)
    #  We also need to remove multi collinearity from the dataset before proceeding
    # Predictive power can DECREASE with an increase in correlation between variables
    # SO if you are not using PCA transformed data for this, then remove features w VIF > 10

    # encoding categorical labels before computation, maybe will speed it up?

    categories = np.unique(labels)
    n = len(categories)

    if classifier:
        # creating train/class split, stratified sampling to make up for imbalance in classes

        xtrain, xtest, ytrain, ytest = train_test_split(data, labels, test_size=0.2, random_state=3, stratify=labels)

        # instantiating model then evaluating with repeated k fold validation
        # remember kf cv splits data into n parts, using one part as validation for each of n runs
        # this way, each fold is checked against at one point
        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=3)
        lda1 = LinearDiscriminantAnalysis(n_components=n-1)
        pipe.steps.append(['lda', lda1])
        # hyperparameter tuning with GridSearchCv
        grid['lda__solver'] = ['svd', 'lsqr', 'eigen']
        search = GridSearchCV(pipe, grid, scoring='accuracy', n_jobs=-1, cv=cv, verbose=2)
        search.fit(xtrain, ytrain)
        lda1 = search.best_estimator_
        train_score = search.best_score_*100
        print('The training score of the LDA classifier is %.2f' % train_score)
        test_score = lda1.score(xtest, ytest)*100
        print('The testing score of this LDA classifier is %.2f' % test_score)
        y_pred = lda1.predict(xtest)
        print(classification_report(ytest, y_pred))
        return

    else:
        lda1 = LinearDiscriminantAnalysis(n_components=n-1)
        lda1.fit(data, labels)
        transformed = lda1.transform(data)
        return transformed


def logit_classifier(data, labels, pipe, grid, know_params=False, kind='multinomial'):
    classes = np.unique(labels)
    # validate that the assumptions of logistic regression have been met, starting with no multi collinearity
    # calculate VIF for each variable, remove if greater than 10
    # turn this off if you have used PCA to make set uncorrelated

    # then create test train split and cross validation repeated k fold
    # but stratify it to ensure that class representation is equal

    xtrain, xtest, ytrain, ytest = train_test_split(data, labels, test_size=0.1, random_state=3, stratify=labels)
    cv = RepeatedKFold(n_splits=3, n_repeats=3, random_state=3)

    # instantiating multinomial logit model
    if know_params:
        imp_params = {'imp__max_iter': 3, 'imp__initial_strategy': 'most_frequent', 'imp__tol': 0.01}
        pipe.set_params(**imp_params)
        best = LogisticRegression(multi_class=kind, penalty='l2', solver='newton-cg', C=100)
        pipe.steps.append(['logit', best])
        scores = cross_val_score(pipe, xtrain, ytrain, cv=cv, scoring='accuracy')
        print('The cross validated train score is %.2f with a spread of %.2f' % (np.mean(scores), np.std(scores)))
        best.fit(xtrain, ytrain)
    else:
        logit = LogisticRegression(multi_class=kind)
        pipe.steps.append(['logit', logit])
        # creating grid of parameters to tune
        if kind == 'multinomial':
            grid['logit__solver'] = ['newton-cg', 'lbfgs', 'sag', 'saga']
            grid['logit__penalty'] = [None, 'l2']
        else:
            grid['logit__solver'] = ['newton-cg', 'lbfgs', 'sag', 'saga', 'liblinear', 'newton-cholesky']
            grid['logit__penalty'] = [None, 'l1', 'l2']
        grid['logit__C'] = [1000, 100, 10, 1, 0.1, 0.01, 0.001]
        # instantiate grid search and fit on training data
        search = RandomizedSearchCV(pipe, grid, n_jobs=-1, scoring='accuracy', verbose=2, cv=cv)
        search.fit(xtrain, ytrain)
        # export the best parameters from the grid search to move forward with
        best = search.best_estimator_
        best_score = search.best_score_*100
        print('The tuned hyperparameters are:', best.get_params,
              'Due to their training score of %.2f' % best_score)

    # evaluate best estimator
    test_score = best.score(xtest, ytest)*100
    print('The final test score of this logit classifier is %.2f' % test_score)
    y_pred = best.predict(xtest)
    print(classification_report(ytest, y_pred))

    input('stop')
    # visualize confusion matrix

    cnf = confusion_matrix(ytest, y_pred)
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    sns.heatmap(pd.DataFrame(cnf), annot=True, cmap='Purples', fmt='g')
    ax.xaxis.set_label_position('top')
    plt.tight_layout()
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    return
