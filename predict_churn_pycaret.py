import pandas as pd
from pycaret.classification import ClassificationExperiment

filepath = '/home/plato/Documents/Regis/Week 5 - Automation and Data Science/new_churn_data.csv'

def load_data(filepath):
    # Loading our new churn data to be predicted
    df = pd.read_csv(filepath, index_col='customerID')
    return df


def make_predictions(df):
    # making churn predictions using the loaded data
    classifier = ClassificationExperiment()
    # retrieving our previously trained gbc model
    model = classifier.load_model('pycaret_model')           
    predictions = classifier.predict_model(model, data=df)
    # changing the default prediction_label column in predict_model method to our preferred column name and value name 
    predictions.rename({'prediction_label': 'Churn'}, axis=1, inplace=True)
    predictions['Churn'].replace({1: 'Churn', 0: 'No Churn'},
                                            inplace=True)
   
    return predictions


if __name__ == "__main__":
    df = load_data(filepath)
    predictions = make_predictions(df)
    print('predictions:')
    print(predictions)