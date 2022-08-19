import csv
import sys

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists of evidences

    labels should be the corresponding list of labels
    """

    csvFile = open(filename, newline='')
    reader = csv.reader(csvFile)
    next(reader)

    evidences = []
    labels = []

    for row in reader:
        evidence = []

        evidence.append(int(row[0]))  # Administrative
        evidence.append(float(row[1]))  # Administrative_Duration
        evidence.append(int(row[2]))  # Informational
        evidence.append(float(row[3]))  # Informational_Duration
        evidence.append(int(row[4]))  # ProductRelated
        evidence.append(float(row[5]))  # ProductRelated_Duration
        evidence.append(float(row[6]))  # BounceRates
        evidence.append(float(row[7]))  # ExitRates
        evidence.append(float(row[8]))  # PageValues
        evidence.append(float(row[9]))  # SpecialDay

        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        for i in range(len(months)):
            if row[10] == months[i]:
                evidence.append(i)  # Month
                break
        
        evidence.append(int(row[11]))  # OperatingSystems
        evidence.append(int(row[12]))  # Browser
        evidence.append(int(row[13]))  # Region
        evidence.append(int(row[14]))  # TrafficType
        evidence.append(1) if row[15] == 'Returning_Visitor' else evidence.append(0)  # VisitorType
        evidence.append(1) if row[16] == 'TRUE' else evidence.append(0)  # Weekend

        evidence = np.array(evidence)
        evidences.append(evidence)  # Evidence
        labels.append(1) if row[17] == 'TRUE' else labels.append(0)  # Label

    return (evidences, labels)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)

    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    sensitivity = 0.0
    specificty = 0.0

    totalTrues = 0
    positiveTrues = 0
    totalFalses = 0
    positiveFalses = 0

    for label, prediction in zip(labels, predictions):
        if label == 1:
            totalTrues += 1
            if prediction == 1:
                positiveTrues += 1
        else:
            totalFalses += 1
            if prediction == 0:
                positiveFalses += 1

    sensitivity = positiveTrues / totalTrues
    specificty = positiveFalses / totalFalses

    return (sensitivity, specificty)        


if __name__ == "__main__":
    main()
