import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():

    with open("data.pickle", "rb") as f:
        data = pickle.load(f)

    x_train, x_test, y_train, y_test = train_test_split(
        data["landmarks"], data["class"], test_size=0.1, shuffle=True, random_state=25
    )

    model = RandomForestClassifier(n_estimators=100, random_state=25)

    model.fit(x_train, y_train)

    # accuracy in my case 84.68%
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy score: {accuracy*100:.2f}%")

    with open("model.pickle", "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    main()
