from xgboost import XGBClassifier


def create_model():

    model = XGBClassifier(

        # =================================================
        # trees
        # =================================================

        n_estimators=200,
        max_depth=4,

        learning_rate=0.03,

        # =================================================
        # regularization
        # =================================================

        subsample=0.8,
        colsample_bytree=0.8,

        reg_alpha=1.0,
        reg_lambda=2.0,

        min_child_weight=5,

        # =================================================
        # imbalance
        # =================================================

        scale_pos_weight=0.35,

        # =================================================
        # misc
        # =================================================

        objective="binary:logistic",

        eval_metric="logloss",

        random_state=42,

        tree_method="hist",
    )

    return model