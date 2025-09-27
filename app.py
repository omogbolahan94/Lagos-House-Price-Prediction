from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
imporrt os


app = Flask(__name__)


# Load saved pipeline (preprocessor + model)
model = joblib.load("house_price_pipeline_rf.joblib")

# Dropdown options
transaction_types = ["sale", "rent"]
locations = [
    'Lekki Phase 1', 'Ajah', 'Chevron Toll Gate', 'Lekki Phase 2',
    'Admiralty Way', 'Lekki County Homes', 'Sora Estste Chevron',
    'Off Chevron Drive', 'Ibeju Lekki', 'Orchid', 'Eleganza Gardens',
    'Maben Estate Chevron', 'Ologolo', 'Lekki', 'Ikate', 'Osapa',
    'Old Ikoyi', 'Banana Island', 'Ikoyi', 'Osborne Foreshore (Ikoyi)',
    'Parkview Estate', 'Ikeja Axis', 'Ikeja', 'Ogba', 'Ikeja GRA',
    'Ifako Ijaye', 'Allen (Ikeja)', 'Mobolaji Bank Anthony (Ikeja)',
    'Victoria Island', 'Ajose Adeogun', 'Oniru', 'Ahmadu Bello Way'
]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        bedrooms = int(request.form["Bedrooms"])
        bathrooms = int(request.form["Bathrooms"])
        toilets = int(request.form["Toilets"])
        parking_spaces = int(request.form["Parking_Spaces"])
        transaction_type = request.form["Transaction_Type"]
        location = request.form["Location"]

        # Convert input into dataframe (model expects same structure as training)
        input_data = pd.DataFrame([{
            "Bedrooms": bedrooms,
            "Bathrooms": bathrooms,
            "Toilets": toilets,
            "Parking_Spaces": parking_spaces,
            "Transaction_Type": transaction_type,
            "Location(2)": location
        }])

        # Predict
        prediction = model.predict(input_data)[0]
        prediction = f"₦ {prediction:,.0f}" 

    return render_template("index.html", 
                           prediction=prediction,
                           transaction_types=transaction_types,
                           locations=locations)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
    
