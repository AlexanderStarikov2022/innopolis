!pip install catboost

!pip install optuna

import numpy as np
import optuna
from optuna.integration import CatBoostPruningCallback

import catboost as cb
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import pandas as pd

df = pd.read_csv("train_dataset_train (2).csv")
test = pd.read_csv("test_dataset_test (1).csv")

data = df.drop(["id",".geo", "crop"], axis = 1)
target = df[["crop"]]

# Best trial:
#   Value: 0.993006993006993
#   Params: 
#     colsample_bylevel: 0.09577110194372195
#     depth: 10
#     boosting_type: Ordered
#     bootstrap_type: Bernoulli
#     subsample: 0.9718309436226747

def objective(trial: optuna.Trial) -> float:
    data, target = load_breast_cancer(return_X_y=True)
    train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.25)

    param = {
        "objective": 'MultiClass',
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1, log=True),
        "depth": trial.suggest_int("depth", 1, 12),
        "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
        ),
        "used_ram_limit": "3gb",
        "eval_metric": "Accuracy",
    }

    if param["bootstrap_type"] == "Bayesian":
        param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    elif param["bootstrap_type"] == "Bernoulli":
        param["subsample"] = trial.suggest_float("subsample", 0.1, 1, log=True)

    gbm = cb.CatBoostClassifier(**param)

    pruning_callback = CatBoostPruningCallback(trial, "Accuracy")
    gbm.fit(
        train_x,
        train_y,
        eval_set=[(valid_x, valid_y)],
        verbose=0,
        early_stopping_rounds=100,
        callbacks=[pruning_callback],
    )

    # evoke pruning manually.
    pruning_callback.check_pruned()

    preds = gbm.predict(valid_x)
    pred_labels = np.rint(preds)
    accuracy = accuracy_score(valid_y, pred_labels)

    return accuracy


if __name__ == "__main__":
    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), direction="maximize"
    )
    study.optimize(objective, n_trials=100, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

#Score = 0.974069
params = {'objective': 'MultiClass', 
          'colsample_bylevel': 0.09577110194372195, 
          'depth': 10, 
          'boosting_type': 'Ordered', 
          'bootstrap_type': 'Bernoulli',
          'subsample': 0.9718309436226747 
          }

best_model = cb.CatBoostClassifier(**params)

pruning_callback = CatBoostPruningCallback(trial, "Accuracy")

train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.25)

best_model.fit(
        train_x,
        train_y,
        eval_set=[(valid_x, valid_y)],
        verbose=0,
        early_stopping_rounds=100
        # callbacks=[pruning_callback],
    )

from sklearn.metrics import f1_score

# best_model.fit(train_x, train_y)

predict_valid = best_model.predict(valid_x)
predict_train = best_model.predict(train_x)

print('Метрика F1 на тренировочной выборке:', f1_score(train_y, predict_train, average='macro', zero_division = 0))
print('Метрика F1 на валидационной выборке:', f1_score(valid_y, predict_valid, average='macro', zero_division = 0))

submission = pd.read_csv('sample_solution.csv')

test = test.drop('.geo', axis =1)

best_model = cb.CatBoostClassifier(**params)

# cb = CatBoostClassifier(**params)
best_model.fit(data, target)

predict_test = best_model.predict(test)

submission

submission["crop"] = predict_test
submission.head(4)

submission.to_csv("complete_solution.csv", index=False)
