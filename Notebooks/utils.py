import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline

def compute_costs(LoanAmount):
     return({'Risk_No Risk': 5.0 + .6 * LoanAmount, 'No Risk_No Risk': 1.0 - .05 * LoanAmount,
         'Risk_Risk': 1.0, 'No Risk_Risk': 1.0})

def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
   '''
   A custom metric for the German credit dataset
   '''
   real_prop = {'Risk': .02, 'No Risk': .98}
   train_prop = {'Risk': 1/3, 'No Risk': 2/3}
   custom_weight = {'Risk': real_prop['Risk']/train_prop['Risk'], 'No Risk': real_prop['No Risk']/train_prop['No Risk']}
   costs = compute_costs(solution['LoanAmount'])
   y_true = solution['Risk']
   y_pred = submission['Risk']
   loss = (y_true=='Risk') * custom_weight['Risk'] *\
               ((y_pred=='Risk') * costs['Risk_Risk'] + (y_pred=='No Risk') * costs['Risk_No Risk']) +\
            (y_true=='No Risk') * custom_weight['No Risk'] *\
               ((y_pred=='Risk') * costs['No Risk_Risk'] + (y_pred=='No Risk') * costs['No Risk_No Risk'])
   return loss.mean()


def objective(trial, X_train, y_train, X_valid, y_valid):

    categorical_cols = (
        X_train
        .select_dtypes(include=["object", "category"])
        .columns.tolist()
    )
    numerical_cols = X_train.select_dtypes(include = np.number).columns.tolist()


    preprocessor = ColumnTransformer(
        transformers=
            [("categorical", OrdinalEncoder(), categorical_cols),
             ("numerical", StandardScaler(), numerical_cols)
             ],
             remainder="passthrough",
             )

    learning_rate = trial.suggest_float("learning_rate", 0.001, 0.3)
    max_iter = trial.suggest_int("max_iter", 100, 500)
    max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 31, 100)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 10, 50)
    l2_regularization = trial.suggest_float("l2_regularization", 0.0, 10.0)
    validation_fraction = trial.suggest_float("validation_fraction", 0.1, 0.3)
    n_iter_no_change = trial.suggest_int("n_iter_no_change", 5, 20)

    # Create the classifier pipeline
    model = HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        validation_fraction=validation_fraction,
        n_iter_no_change=n_iter_no_change,
    )
    pipeline = make_pipeline(
       preprocessor,
       model
       )

    # Train it
    pipeline.fit(X_train, y_train)
    # Get the score on the validation set
    y_pred = pipeline.predict(X_valid)

    # Let's have the submission to the right format to assess our model
    submission = pd.DataFrame(data = y_pred)
    submission = submission.reset_index()
    submission.columns = ["ID", "Risk"]

    #Now let's score it and return the score

    return score(y_valid, submission,"optuna")