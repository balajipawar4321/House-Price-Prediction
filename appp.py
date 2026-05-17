from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model & encoder
model = pickle.load(open("property_price_model.pkl", "rb"))
area_encoder = pickle.load(open("area_encoder.pkl", "rb"))

@app.route("/")
def home():
    areas = list(area_encoder.classes_)
    return render_template("index.html", areas=areas)

@app.route("/predict", methods=["POST"])
def predict():

    # ---- BASIC DETAILS ----
    area = request.form["area"]
    area_sqft = float(request.form["area_sqft"])
    bhk = int(request.form["bhk"])
    bathrooms = int(request.form["bathrooms"])
    balcony = int(request.form["balcony"])
    floor = int(request.form["floor"])
    total_floors = int(request.form["total_floors"])
    age = int(request.form["age"])

    # ---- PROPERTY DETAILS ----
    furnishing = request.form["furnishing"]
    property_type = request.form["property_type"]

    # ---- LOCATION ----
    locality_rating = float(request.form["locality_rating"])
    distance = float(request.form["distance"])

    # ---- AMENITIES ----
    parking = 1 if request.form["parking"] == "Yes" else 0
    lift = 1 if request.form["lift"] == "Yes" else 0
    security = 1 if request.form["security"] == "Yes" else 0
    power_backup = 1 if request.form["power_backup"] == "Yes" else 0

    # ---- NEARBY ----
    school = float(request.form["school"])
    hospital = float(request.form["hospital"])
    market = float(request.form["market"])

    # ---- ENCODING ----
    furnishing_map = {"Unfurnished": 0, "Semi-Furnished": 1, "Fully-Furnished": 2}
    property_map = {"Apartment": 0, "Villa": 1, "Row House": 2}

    area_encoded = area_encoder.transform([area])[0]

    input_data = np.array([[
        area_encoded,
        area_sqft,
        bhk,
        bathrooms,
        balcony,
        floor,
        total_floors,
        age,
        furnishing_map[furnishing],
        property_map[property_type],
        locality_rating,
        distance,
        parking,
        lift,
        security,
        power_backup,
        school,
        hospital,
        market
    ]])

    price_per_sqft = model.predict(input_data)[0]
    total_price = price_per_sqft * area_sqft

    return render_template(
        "index.html",
        areas=list(area_encoder.classes_),
        price_sqft=int(price_per_sqft),
        total_price=int(total_price)
    )

if __name__ == "__main__":
    app.run(debug=True)