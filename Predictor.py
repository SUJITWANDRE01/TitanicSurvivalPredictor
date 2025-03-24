import math
import numpy as np
import pandas as pd
import seaborn as sns
from seaborn import countplot
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure,show
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

def TitanicLogistic():
    # Load data
    titanic_data = pd.read_csv('TitanicDataset.csv')

    print("First 5 entries from loaded dataset")
    print(titanic_data.head())

    print("Number of passangers are "+str(len(titanic_data)))

    # Analyze data
    print("Visualisation : Survived and non survied passangers")
    figure()
    target = "Survived"
    countplot(data=titanic_data,x=target).set_title("Survived and non survived passengers")
    show()
    
    print("Visualisation : Survived and non survied passangers based on Gender")
    figure()
    target = "Survived"
    countplot(data=titanic_data,x=target,hue="Sex").set_title("Survived and non survived passengers based on Gender")
    show()

    print("Visualisation : Survived and non survied passangers based on the Passanger class")
    figure()
    target = "Survived"
    countplot(data=titanic_data,x=target,hue="Pclass").set_title("Survived and non survived passengers based on the Passanger class")
    show()

    print("Visualisation : Survived and non survied passangers based on Age")
    figure()
    titanic_data["Age"].plot.hist().set_title("Survived and non survied passangers based on Age")
    show()

    print("Visualisation : Survived and non survied passangers based on the Fare")
    figure()
    titanic_data["Fare"].plot.hist().set_title("Survived and non survied passangers based on Fare")
    show()

    # Data Cleaning
    # Drop unnecessary columns
    titanic_data.drop("zero", axis=1, inplace=True)

    # Encode categorical variables
    le = LabelEncoder()
    titanic_data['Sex'] = le.fit_transform(titanic_data['Sex'])
    
    # One-hot encode Pclass
    Pclass = pd.get_dummies(titanic_data["Pclass"], prefix='Pclass', drop_first=True)
    
    # Prepare features
    titanic_data = pd.concat([titanic_data, Pclass], axis=1)
    
    # Select features
    features = ['Passengerid', 'Age', 'Fare', 'Sex', 'Pclass_2', 'Pclass_3']
    x = titanic_data[features]
    y = titanic_data["Survived"]

    # Ensure column names are strings
    x.columns = x.columns.astype(str)

    # Data Training
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.5, random_state=42)

    # Train Logistic Regression
    logmodel = LogisticRegression(max_iter=1000)
    logmodel.fit(xtrain, ytrain)

    # Data testing 
    prediction = logmodel.predict(xtest)

    # Calculate Accuracy
    print("Classification report of Logistic Regression is : ")
    print(classification_report(ytest, prediction))

    print("Confusion Matrix of Logistic Regression is : ")
    print(confusion_matrix(ytest, prediction))

    print("Accuracy of Logistic Regression is : ")
    print(accuracy_score(ytest, prediction))

def main():
    print("Logistic Regression on Titanic data set")
    TitanicLogistic()

if __name__ == "__main__":
    main()
