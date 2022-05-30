import network as n
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import collections

# the layout of the code is the following: in network.py we created the neural network,in perceptron.py
# we have the single perceptron code you can run to see the plots from report in 2.2.1
# in the main.py all the methods are called from network so the splitting data, changing format to the desired one
# cross validation, training and in the end return_unknown_results is computing the prediction of the test set
# you just have to run the main.py class and after a lot of prints you can see in the end the results in the chosen file
# Some methods are commented as of now, so that the program trains, validates, tests and runs the predictions for
# the unknown set. If you want to just train and test, uncomment evaluate_no_validation. If you want to run the
# network again with the best result from validation, uncomment evaluate_best_result


def main():
    # Read provided data
    features, labels, unknown = read_data()

    # Split into train set and test set
    x_train, x_test, y_train, y_test = train_test_split(features, labels, train_size=0.9,
                                                        random_state=42, shuffle=True, stratify=labels)

    # Change format of the data, so that it matches the one expected by the network
    training, training_for_validate, testing, unknown = change_data_format(x_train, x_test, y_train, y_test, unknown)

    # evaluate_no_validation(training, testing)
    evaluate_cross_validation(training, training_for_validate, testing, unknown)


# Changes data format
def change_data_format(x_train, x_test, y_train, y_test, unknown):
    training = []
    testing = []
    training_for_validate = []
    unk = []

    # Change shape of input for training the network
    for i in range(np.shape(x_train)[0]):
        transformed_label = np.zeros(7)
        transformed_label[int(y_train[i]) - 1] = 1
        transformed_label = [[j] for j in transformed_label]

        x = [[j] for j in x_train[i]]
        x = np.array(x)
        x = x.astype("float32")
        training.append((x, np.array(transformed_label)))

    # Change shape of input for validation
    for i in range(np.shape(x_train)[0]):
        x = [[j] for j in x_train[i]]
        x = np.array(x)
        x = x.astype("float32")
        training_for_validate.append((x, int(y_train[i] - 1)))

    # Change shape of input for testing the network
    for i in range(np.shape(x_test)[0]):
        x = [[j] for j in x_test[i]]
        x = np.array(x)
        x = x.astype("float32")
        testing.append((x, int(y_test[i] - 1)))

    # Change shape of input for unknown set
    for i in range(np.shape(unknown)[0]):
        x = [[j] for j in unknown[i]]
        x = np.array(x)
        x = x.astype("float32")
        unk.append(x)

    return training, training_for_validate, testing, unk


def evaluate_no_validation(training, testing):
    results = []
    epochs = 30

    # Train 10 times
    for i in range(10):
        network = n.NeuralNetwork([10, 22, 7])
        result = network.training(training, epochs, 10, 0.1, evaluation_data=testing, monitor_evaluation_accuracy=True)
        results.append(result[epochs - 1])

    print(np.mean(results))

    # Plot the result
    plt.plot(results)
    plt.xlabel("Training for the ... time")
    plt.ylabel("Percentage of predictions that are right")
    plt.show()


# Training the network using cross validation
def evaluate_cross_validation(training, training_for_validate, testing, unknown):
    no_folds = 10
    fold_size = len(training) / no_folds
    i = 0
    epochs = 30
    best_accuracy = -1
    best_no_hidden = -1
    current = random.randint(3, 30)
    tried = collections.OrderedDict()
    remaining = 28
    no_repetitions = 5
    best_training = []
    best_validation = []
    no = 0

    # Running the network with different amounts of neurons in the hidden layer
    while i < len(training) and remaining > 0 and no < no_repetitions:
        validation = training_for_validate[int(i): (int(i) + int(fold_size))]
        training2 = training[int(0): int(i)] + training[int(i) + int(fold_size):]

        results = []

        # Train 10 times
        for i in range(10):
            network = n.NeuralNetwork([10, current, 7])
            result = network.training(training2, epochs, 10, 0.1, evaluation_data=validation,
                                      monitor_evaluation_accuracy=True)
            results.append(result[epochs - 1])

        mean = np.mean(results)
        tried[current] = mean

        # Compare current accuracy with best accuracy
        if best_accuracy == -1 or mean > best_accuracy:
            best_accuracy = mean
            best_no_hidden = current
            best_training = training2
            best_validation = validation

        # Make sure we don't choose the same number of neurons again
        while current in tried:
            current = random.randint(3, 30)

        remaining -= 1
        i += fold_size
        no += 1

    # Sort by key
    for key in sorted(tried):
        tried.move_to_end(key)

    values = tried.values()
    keys = tried.keys()

    print(keys, values)

    # Plot keys and values
    plt.plot(keys, values)
    plt.xlabel("Number of perceptrons in the hidden layer")
    plt.ylabel("Mean percentage of predictions that are right")
    plt.show()

    print("The optimal number of perceptrons in the hidden layer is " + str(best_no_hidden) +
          ", with accuracy " + str(best_accuracy))

    # evaluate_best_result(best_no_hidden, best_training, best_validation)
    return_unknown_results(best_no_hidden, training, testing, unknown)


# Output the predicted results for the unknown data and save them in a file
def return_unknown_results(no_hidden, training, testing, unknown):
    epochs = 30

    # Train and get predictions
    network = n.NeuralNetwork([10, no_hidden, 7])
    network.training(training, epochs, 10, 0.1, evaluation_data=testing, monitor_evaluation_accuracy=True)
    result = network.compute_unknown(unknown)

    f = open("../data/Group_26_classes.txt", "w")

    output = str(result[0] + 1)

    # Build output
    for i in range(len(result) - 1):
        output += ","
        output += str(result[i + 1] + 1)

    f.write(output)
    f.close()


def evaluate_best_result(no_hidden, training, validation):
    epochs = 30
    results = np.zeros(epochs)

    # Train 10 times
    for i in range(10):
        network = n.NeuralNetwork([10, no_hidden, 7])
        result = network.training(training, epochs, 10, 0.1, evaluation_data=validation,
                                  monitor_evaluation_accuracy=True)
        for j in range(epochs):
            results[j] += result[j]

    # Compute mean
    for j in range(epochs):
        results[j] = results[j] / 10

    # Plot the result
    plt.plot(results)
    plt.xlabel("Epoch")
    plt.ylabel("Percentage of predictions that are right")
    plt.show()


# this method is reading the input files from the specified paths
def read_data():
    features = np.array([line.strip().split(",") for line in open("../data/features.txt")]).astype(float)
    targets = np.array([line.strip().split(",") for line in open("../data/targets.txt")]).astype(float)
    unknown = np.array([line.strip().split(",") for line in open("../data/unknown.txt")]).astype(float)

    return features, targets, unknown


if __name__ == '__main__':
    main()
