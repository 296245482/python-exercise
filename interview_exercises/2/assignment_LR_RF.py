import numpy as np
import matplotlib.pyplot as plt
import csv
import random
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier


# Load the data
def load_data(file_name):
    data_list = []
    with open(file_name, 'rb') as tsv:
        tsv = csv.reader(tsv, delimiter='\t')
        for row in tsv:
            # Eliminatin the missing value
            if row[2] != '-':
                data_list.append(row)
    return data_list


# Draw the histogram using pyplot
def draw_histogram(data):
    # Init lists to record number in each age group
    male_ages = [0, 0, 0, 0, 0, 0]
    female_ages = [0, 0, 0, 0, 0, 0]

    # Calculate number of gender in each age
    for subject in data[1:]:
        if subject[0] == 'M':
            male_ages[int(subject[1])/10 - 1] += 1
        else:
            female_ages[int(subject[1])/10 - 1] += 1
    # Divide ages into 6 group
    n_group = 6
    fig, ax = plt.subplots()

    # Define the index for the order
    index = np.arange(n_group)

    # The histogram bar width
    bar_width = 0.35

    # .75 transparent
    opacity = 0.25

    # Plot two histograms for male and female in the same chart
    rects1 = plt.bar(index, male_ages, bar_width, alpha=opacity, color='b', label='Men')
    rects2 = plt.bar(index + bar_width, female_ages, bar_width, alpha=opacity, color='r', label='Women')

    # Set the labels, tile, and ticks
    plt.xlabel('Groups')
    plt.ylabel('Numbers')
    plt.title('Numbers by group and gender')
    plt.xticks(index + bar_width, ('10-20', '20-30', '30-40', '40-50', '50-60', '60-70'))

    # Set the X, Y range
    plt.ylim(0, 120)
    plt.xlim(-0.5, 6.5)

    # Show the legend
    plt.legend()
    plt.tight_layout()

    # Save the histogram
    plt.savefig("Histogram.PNG")


def draw_scatter(data):
    # Init the data
    ages = []
    y_lists = []

    # Read the data
    for subject in data[1:]:
        # Both of them added a suitable random value to make points not obscured
        ages.append(int(subject[1])+random.uniform(-1, 1))
        # Use the value of (OGrade - Igrade) for Y-axis data
        y_lists.append(int(subject[2])-int(subject[3])+random.uniform(-0.5, 0.5))

    # Convert into array for sort function
    ages_array = np.array(ages)
    i = 0

    # Init a new list for sort
    y_sorted = list(range(len(ages)))

    # Sort the correspond data the same as the sorted ages
    for index in np.argsort(ages_array):
        y_sorted[i] = y_lists[index]
        i += 1

    # Sort the ages to make it order for scatter
    ages.sort()

    # Set the value of the chart
    plt.figure(figsize=(9, 7), dpi=200)

    # Set the labels and title
    plt.xlabel('Ages')
    plt.ylabel('Differences')
    plt.title('Generosity VS Age')

    # Get the color map
    cm = plt.cm.get_cmap('RdYlBu')

    # Draw the scatter and set the size/color to make the figure look better
    plt.scatter(ages, y_sorted, s=15, c=y_sorted, cmap=cm, marker='o', linewidths=0.1, alpha=0.5)

    # Draw the line and set the size/color to make the figure look better
    plt.plot(ages, y_sorted, 'k', linewidth=0.3)

    # Draw the grid line and set the parameter to make the figure look better
    plt.grid(True, linestyle='--', color="g", linewidth="0.3")

    # Show the color bar
    plt.colorbar()

    # Save the figure in the root path
    plt.savefig("Scatter.PNG")


# Split the data into training set and testing set, with the 70% ratio for training
def data_split(data):

    # Make the random split reproducible
    random.seed(1)
    
    # Change the data's order randomly
    random.shuffle(data)

    # Set the ratio of training set we need
    number = int(len(data)*0.7)

    # Specific the training and testing set
    training = data[:number]
    testing = data[number:]

    # Return the data set
    return training, testing


# Make the gender into 1 and 0, 1 for male and 0 for female
def dummy_indicator(data):
    # Convert data into dummy variables
    for subject in data:
        if subject[0] == 'M':
            subject[0] = 1
        else:
            subject[0] = 0
    # Return the data with dummy indicator
    return data


# Save the training and testing set
def save_data(train_set, test_set):
    # Set the file name
    train_path = 'trainDataSet.csv'
    test_path = 'testDataSet.csv'
    # Write the data into train.csv
    with open(train_path, 'wb') as csvfile:
        writer = csv.writer(csvfile)
        for row in train_set:
            writer.writerow(row)
    # Write the data into test.csv
    with open(test_path, 'wb') as csvfile2:
        writer = csv.writer(csvfile2)
        for row in test_set:
            writer.writerow(row)
    # Return these data file path
    return train_path, test_path


# Apply the logistic regression and test the model with testing data
def logistic_regression(train, test):
    # Training part
    df = pd.read_csv(train)

    # Specify the columns for our data
    df.columns = ["gender", "age", "ograde", "igrade"]
    cols_to_keep = ['gender', 'age', 'ograde', 'igrade']
    train_data = df[cols_to_keep]

    # Add the intercept value which needed in logistic regression
    train_data['intercept'] = 1.0

    # Specify the training columns, without 'gender' column
    train_cols = train_data.columns[1:]

    # Use the 'statsmodels' to train our model
    logit = sm.Logit(train_data['gender'], train_data[train_cols])
    result = logit.fit()

    # Testing part
    df2 = pd.read_csv(test)
    # Specify the columns for testing data
    df2.columns = ["gender", "age", "ograde", "igrade"]

    # Add the intercept value which needed in logistic regression
    df2['intercept'] = 1.0

    # Specify the training columns, without 'gender' column
    predict_cols = df2.columns[1:]

    # Save the predict result of our model
    df2['predict'] = result.predict(df2[predict_cols])

    # Init some value
    total = 0
    hit = 0

    # The list to record predict
    dummy_predict = []
    for value in df2.values:
        # The gender calculate by our model
        predict = value[-1]
        # The actual gender value
        gender = int(value[0])
        # If > 0.5 we think it is reliable to believed that this subject is male
        if predict > 0.5:
            # Append the male prediction
            dummy_predict.append(1)
            # Record number of males we recognized
            total += 1
            if gender == 1:
                # Record number of the correct predicts we have
                hit += 1
        else:
            # Append the female prediction
            dummy_predict.append(0)
    # Print the precision of LR
    print '\n\n*1.1 LR - PredictedMaleInTestingSet: %d, Hit: %d, Precision: %.2f' % (total, hit, 100.0*hit/total)

    # Print the confidence interval
    # print '\n %s' % (result.conf_int())

    # Using the pandas to draw the confusion matrix
    print '\n*1.2 The confusion matrix of LR is:\n'
    print pd.crosstab(df2['gender'], np.array(dummy_predict), rownames=['actual'], colnames=['predicts'])


def random_forest(train, test):
    # Training part
    # Load the training and testing data
    train_df = pd.read_csv(train)
    train_df.columns = ["gender", "age", "ograde", "igrade"]
    test_df = pd.read_csv(test)
    test_df.columns = ["gender", "age", "ograde", "igrade"]

    # Point out the training set
    features = train_df.columns[1:4]
    features2 = test_df.columns[1:4]

    # Calculate the random forest model
    clf = RandomForestClassifier(n_jobs=2)
    y, _ = pd.factorize(train_df['gender'])
    clf.fit(train_df[features], y)

    # Testing part
    # Get the predict of testing set
    test_df['predict'] = clf.predict(test_df[features2])

    # Init some value
    total = 0
    hit = 0
    for value in test_df.values:
        # The gender calculate by our model
        predict = value[-1]
        # The actual gender value
        gender = int(value[0])
        # If predict = 1, we predict it a male
        if predict == 1:
            # Record number of males we recognized
            total += 1
            if gender == 1:
                # Record number of the correct predicts we have
                hit += 1
    # Print the precision of RF
    print '\n\n*2.1 RF - PredictedMaleInTestingSet: %d, Hit: %d, Precision: %.2f' % (total, hit, 100.0*hit/total)

    # Using the pandas to draw the confusion matrix
    print '\n*2.2 The confusion matrix of RF is:\n'
    print pd.crosstab(test_df['gender'], test_df['predict'], rownames=['actual'], colnames=['predicts'])

# Set the init data path and name
path = "data-subset.tsv"

# Load our data
data = load_data(path)

# Draw histogram
draw_histogram(data)

# Draw scatter
draw_scatter(data)

# Wipe out header
data = data[1:]

# Dummy gender as indicator
dummy_indicator(data)

# Divide training and testing set
training, testing = data_split(data)

# Save these data set
train_path, test_path = save_data(training, testing)

# Train and validate with logistic regression
logistic_regression(train_path, test_path)

# Train and validate with random forest
random_forest(train_path, test_path)