
from flask import Flask, render_template, jsonify, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle

app = Flask(__name__)


class HousePriceModel:
    def __init__(self):
        self.model, self.original_columns = self.train_model()

    def train_model(self):
        # Read the dataset
        df = pd.read_csv('Entities.csv', usecols=[
                         'property_type', 'price', 'city', 'baths', 'bedrooms', 'Total_Area'])

        # Manually encode 'city' with specific numeric codes
        city_mapping = {'Islamabad': 0, 'Lahore': 1,
                        'Rawalpindi': 2, 'Karachi': 3, 'Faisalabad': 4}
        df['city'] = df['city'].map(city_mapping)

        # One-hot encode categorical variables
        df = pd.get_dummies(
            df, columns=['property_type'], drop_first=True)

        X = df.drop('price', axis=1)
        y = df['price']

        # Split the dataset into training and testing sets
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # Create a linear regression model
        model = LinearRegression()

        # Train the model
        model.fit(X_train, y_train)

        # Assess model accuracy on the test set
        y_pred = model.predict(X_test)
        r_squared = r2_score(y_test, y_pred)
        print(f'R-squared on test set: {r_squared}')

        return model, X.columns.tolist()

    def save_model(self, filename):
        with open(filename, 'wb') as model_file:
            pickle.dump(
                {'model': self.model, 'original_columns': self.original_columns}, model_file)

    def load_model(self, filename):
        with open(filename, 'rb') as model_file:
            data = pickle.load(model_file)
            self.model = data['model']
            self.original_columns = data['original_columns']

    def predict_price(self, input_data):
        input_df = pd.DataFrame([input_data])

        # Manually encode 'city' with specific numeric codes
        city_mapping = {'Islamabad': 0, 'Lahore': 1,
                        'Rawalpindi': 2, 'Karachi': 3, 'Faisalabad': 4}
        input_df['city'] = input_df['city'].map(city_mapping)

        # Convert property_type to upper case (as mentioned in the previous conversion logic)
        input_df['property_type'] = input_df['property_type'].str.upper()

        # Convert property_type and city to one-hot encoded format
        input_df = pd.get_dummies(
            input_df, columns=['property_type'], drop_first=True)

        # Ensure consistency in feature names
        missing_columns = set(self.original_columns) - set(input_df.columns)
        for col in missing_columns:
            input_df[col] = 0  # Add missing columns with zero values

        # Reorder columns to match the order during training
        input_df = input_df[self.original_columns]

        # Make prediction
        prediction = self.model.predict(input_df)[0]
        return prediction


# Create an instance of the HousePriceModel class
house_price_model = HousePriceModel()

# Train the model and save it to a file
house_price_model.save_model('linear_regression_model.pkl')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request form
        input_data = {
            'property_type': request.form['property_type'],
            'baths': float(request.form['baths']),
            'bedrooms': float(request.form['bedrooms']),
            'Total_Area': float(request.form['Total_Area']),
            'city': request.form['city']
        }

        # Make prediction using the model
        prediction = house_price_model.predict_price(input_data)

        # Return the prediction as JSON
        return jsonify({'prediction': prediction})

    except Exception as e:
        # Return an error message as JSON
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
