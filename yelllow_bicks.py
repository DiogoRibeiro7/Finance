# Import Needed libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from yellowbrick.target import FeatureCorrelation
from yellowbrick.classifier import ClassBalance, ClassificationReport, ConfusionMatrix, DiscriminationThreshold
from yellowbrick.features import JointPlotVisualizer, PCADecomposition, RadViz, Rank1D, Rank2D
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier, XGBRFClassifier
import warnings


warnings.filterwarnings("ignore", category=FutureWarning)
sns.set(rc={'figure.figsize': (15, 6)})


# Load data from UCI dataset repo
bank_note_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt'
data = np.loadtxt(bank_note_url, delimiter=',')
data = pd.DataFrame(data)

# Add column names
clean_columns = ['variance_of_wavelet', 'skewness_of_wavelet',
                 'curtosis_of_wavelet', 'entropy_of_wavelet',
                 'class']

data.columns = clean_columns

data.head()


# Separate the target and features as separate dataframes for sklearn APIs
X = data.drop('class', axis=1)
y = data[['class']].astype('int')


# Specify the design matrix and the target vector for yellowbrick as arrays
design_matrix = X.values
target_vector = y.values.flatten()


# Stratified sampling based on the distribution of the target vector, y
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y,
                                                    test_size=0.20,
                                                    random_state=30)


# Stage 1. Feature Analysers
feature_names = X.columns.tolist()
target_name = y.columns.tolist()



# Target balance
target_balance = ClassBalance(labels=['not_authentic', 'authentic'])
target_balance.fit(target_vector)
target_balance.show()


# Feature correlation (** requires dataframe and 1D target vector)
feature_correlation = FeatureCorrelation(method='mutual_info-classification',
                                         feature_names=feature_names, sort=True)
feature_correlation.fit(X, y.values.flatten())
feature_correlation.show()

# Radviz
rad_viz = RadViz(classes=['not_authentic', 'authentic'],
                 features=feature_names)

rad_viz.fit(design_matrix, target_vector)
rad_viz.show()


# 1D Rank
rank_1d = Rank1D(algorithm='shapiro', features=feature_names)
rank_1d.fit_transform_show(design_matrix, target_vector)


# 2D Rank
rank_2d = Rank2D(algorithm='pearson', features=feature_names)
rank_2d.fit_transform_show(design_matrix, target_vector);


# PCA Projection
colors = np.array(['red' if yi else 'blue' for yi in target_vector])

pca = PCADecomposition(scale=True, proj_features=True, features=feature_names)
pca.fit_transform_show(design_matrix, target_vector, colors=colors)


# Model Selection
models = [
    DummyClassifier(strategy='most_frequent'), LogisticRegression(),
    SGDClassifier(), LogisticRegressionCV(), HistGradientBoostingClassifier(),
    RandomForestClassifier(), ExtraTreesClassifier(), GradientBoostingClassifier(),
    BaggingClassifier(), AdaBoostClassifier(), XGBClassifier(),
    XGBRFClassifier(), MLPClassifier(max_iter=1000), LinearSVC(), NuSVC(), SVC(),
    GaussianNB(), DecisionTreeClassifier(), QuadraticDiscriminantAnalysis(),
    KNeighborsClassifier(n_neighbors=2), GaussianProcessClassifier()
]



def training_cv_score_model(X, y, model):
    """
    This function takes the design matrix and target vector of the training set,
    along with a classification estimator and computes a 10 fold cross validated
    mean and standard deviation based on balanced accuracy.
    This score is printed to the end user.
    """
    numeric_transformer = Pipeline(steps=[
        ('scale_x_num', StandardScaler())
    ])

    pre_processor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, feature_names)
    ])

    clf = Pipeline(steps=[
        ('preprocessor', pre_processor),
        ('classifier', model)
    ])

    scores = cross_val_score(clf,
                             X,
                             y.values.flatten(),
                             scoring='balanced_accuracy',
                             cv=10)

    mean_score = scores.mean()
    avg_score = scores.std()

    model_name = clf.named_steps['classifier'].__class__.__name__

    print(f'{model_name}, Average Score : {mean_score} & Standard Deviation: {avg_score}')
    print('-'*90)




def out_of_sample_score(train_features, train_target, test_features, test_target, model):
    """
    This function takes the design matrix and target vector of the validation set,
    along with a classification estimator and
    scores the predicted classes and true values via a balanced accuracy score.
    This score is printed to the end user.
    """
    numeric_transformer = Pipeline(steps=[
        ('scale_x_num', StandardScaler())
    ])

    pre_processor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, feature_names)
    ])

    clf = Pipeline(steps=[
        ('preprocessor', pre_processor),
        ('classifier', model)
    ])

    model_name = clf.named_steps['classifier'].__class__.__name__

    clf.fit(train_features, train_target.values.ravel())

    preds = clf.predict(test_features)
    score = balanced_accuracy_score(y_true=test_target, y_pred=preds)

    print(f'{model_name} with balanced accuracy on test set {score}')
    print('-'*90)



def visualise_out_of_sample(train_features, train_target, test_features, test_target, model):
    """
    This function takes the design matrix and target vector of the validation set,
    along with a classification estimator and
    scores the predicted classes and true values via a classification report and
    a confusion matrix.
    """
    # Classification pipeline
    numeric_transformer = Pipeline(steps=[
        ('scale_x_num', StandardScaler())
    ])

    pre_processor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, feature_names)
    ])

    clf = Pipeline(steps=[
        ('preprocessor', pre_processor),
        ('classifier', model)
    ])

    # Classification report
    clf_report = ClassificationReport(clf, classes=['not_authentic', 'authentic'],
                                      cmap="YlGn", size=(600, 360))

    clf_report.fit(train_features, train_target.values.flatten())
    clf_report.score(test_features, test_target.values.flatten())
    clf_report.show()

    # Confusion matrix
    cm = ConfusionMatrix(clf, classes=['not_authentic', 'authentic'])


    cm.fit(train_features, y_train.values.flatten())
    cm.score(test_features, test_target.values.flatten())
    cm.show()


for model in models:
    try:
        training_cv_score_model(X_train, y_train, model)
    except Exception as e:
        raise e


for model in models:
    try:
        out_of_sample_score(X_train, y_train, X_test, y_test, model)
    except Exception as e:
        raise e

for model in models:
    try:
        visualise_out_of_sample(X_train, y_train, X_test, y_test, model)
    except Exception as e:
        raise e

# Xgboost GridSearch Pipeline
numeric_transformer = Pipeline(steps=[
        ('scale_x_num', StandardScaler())
    ])

pre_processor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, feature_names)
])

clf = Pipeline(steps=[
    ('preprocessor', pre_processor),
    ('classifier', XGBClassifier())
    ])

class_weight = y_train['class'].value_counts()[0] / y_train['class'].value_counts()[1]


search_space = {
    'classifier__n_estimators': [x for x in np.arange(100, 500).tolist()],
    'classifier__base_score': [np.round(x, 1) for x in np.arange(0.1, 0.99, 0.1).tolist()],
    'classifier__scale_pos_weight': [1, class_weight]
}


xg_grid_search = GridSearchCV(estimator=clf,
                              param_grid=search_space,
                              scoring='balanced_accuracy',
                              cv=10,
                              n_jobs=2,
                              verbose=5)


xg_grid_search.fit(X_train, y_train.values.flatten())


xg_grid_search.best_score_

xg_grid_search.score(X=X_test, y=y_test)




# Final Pipeline
final_num_transformer = Pipeline(steps=[
        ('scale_x_num', StandardScaler())
    ])

final_processor = ColumnTransformer(transformers=[
    ('num', final_num_transformer, feature_names)
])


# Bundled models
voters = VotingClassifier(estimators=[
    ('mlp', MLPClassifier(max_iter=1000)),
    ('svc', SVC())
], voting='hard')

final_clf = Pipeline(steps=[
    ('final_preprocessor', final_processor),
    ('final_classifier', voters)
])


final_clf.fit(X, y.values.flatten())
final_preds = final_clf.predict(X)
print(balanced_accuracy_score(y_true=y, y_pred=final_preds))

# Final Confusion Matrix
final_cm = ConfusionMatrix(final_clf, classes=['not_authentic', 'authentic'])
final_cm.fit(X, y.values.flatten())
final_cm.score(X, y.values.flatten())
final_cm.show();

