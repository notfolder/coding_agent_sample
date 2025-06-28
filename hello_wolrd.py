from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def train_iris_classifier():
    iris = datasets.load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    classifier = train_iris_classifier()
    print("Iris classifier trained successfully!")