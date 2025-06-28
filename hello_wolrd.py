from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def load_iris_data():
    iris = load_iris()
    X, y = iris.data, iris.target
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_evaluate():
    X_train, X_test, y_train, y_test = load_iris_data()
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return accuracy_score(y_test, predictions)

def main():
    accuracy = train_and_evaluate()
    print(f'Iris classification accuracy: {accuracy:.2f}')

if __name__ == "__main__":
    main()