from sklearn.tree import DecisionTreeClassifier


def create_model():
    model = DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=10,
        class_weight={0: 1.0, 1: 2.0}  # bias toward KEEP
    )
    return model