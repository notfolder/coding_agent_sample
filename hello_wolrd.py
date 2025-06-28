from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def train_iris_classifier():
    iris = datasets.load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
    return model, accuracy, conf_matrix

if __name__ == "__main__":
    classifier, accuracy, conf_matrix = train_iris_classifier()
    print(f"Iris classifier trained successfully! Accuracy: {accuracy:.2f}")
    print("Confusion Matrix:\n", conf_matrix)