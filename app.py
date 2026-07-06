import pickle
from flask import Flask, render_template, request, jsonify

from src.pipline.predictpipline import predictpipline, CustomData

from src.utils import load_object
import sys
from src.exception import CustomException
from src.logger import logging

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    print("POSTMAN /predict HIT")

    data = CustomData(
        gender=request.form.get("gender"),
        race_ethnicity=request.form.get("race_ethnicity"),
        parental_level_of_education=request.form.get("parental_level_of_education"),
        lunch=request.form.get("lunch"),
        test_preparation_course=request.form.get("test_preparation_course"),
        reading_score=float(request.form.get("reading_score")),
        writing_score=float(request.form.get("writing_score")),
    )

    final_df = data.get_data_as_data_frame()
    logging.info(final_df.head(1).to_string())

    pipeline = predictpipline()
    prediction = pipeline.predict(final_df)

    return render_template(
        "index.html",
        prediction_text=f"Predicted Math Score: {round(prediction[0], 2)}"
    )

if __name__ == "__main__":
   app.run(host="0.0.0.0", port=5000)

