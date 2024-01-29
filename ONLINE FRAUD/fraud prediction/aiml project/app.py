from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open(r"C:\\Users\\rajab\\Downloads\\dataset\\best_model1.pkl", 'rb'))

app = Flask(__name__)

@app.route('/')
def about():
    return render_template('home.html')

@app.route('/home')
def about1():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def home1():
    if request.method == 'GET':
        return render_template('predict.html')

@app.route('/submit', methods=['POST', 'GET'])
def predict_up():
    if request.method == 'POST':
        form_values = [request.form[field] for field in request.form]
        print("Form values:", form_values)

        # Check if any form field is empty
        if any(value == '' for value in form_values):
            return render_template('submit.html', prediction_text="Please fill in all the fields")

        try:
            # Convert form values to float (excluding empty strings)
            form_values_float = [float(value) for value in form_values if value.strip()]
            print("Form values (float):", form_values_float)

            if len(form_values_float) > 0:
                x = np.array(form_values_float).reshape(1, -1)
                print("Input shape:", x.shape)

                pred = model.predict(x)
                print("Prediction:", pred[0])

                return render_template('submit.html', prediction_text=str(pred[0]))
            else:
                return render_template('submit.html', prediction_text="No valid input data provided")
        except ValueError:
            return render_template('submit.html', prediction_text="Invalid input data")
    else:
        return render_template('submit.html', prediction_text="No prediction available")
    


if __name__ == '__main__':
    app.run(debug=True)
 