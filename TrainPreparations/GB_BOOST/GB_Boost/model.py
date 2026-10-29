from xgboost import XGBClassifier


def create_model():

    model = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,

        subsample=0.8,
        colsample_bytree=0.8,

        min_child_weight=5,

        reg_lambda=2.0,

        objective="binary:logistic",

        eval_metric="logloss",

        scale_pos_weight=1.5,

        random_state=42,
    )

    return model