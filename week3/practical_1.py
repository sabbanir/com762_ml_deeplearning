from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
X,y = make_classification(n_samples=20000, random_state=1)
# print(X)
# print(y)
# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)


clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)


# Predicted probabilities P(y|x)
y_prob = clf.predict_proba(X_test)


# Accuracy score
accuracy = accuracy_score(y_test, y_pred)

# Display results
print("Predicted Probabilities (P(y|x)):\n", y_prob)
print("\nPredictions:\n", y_pred)
print("\nAccuracy Score:", accuracy)

# Loss function value
print("\nFinal Loss:", clf.loss_)

# Activation function used
print("\nActivation Function:", clf.activation)