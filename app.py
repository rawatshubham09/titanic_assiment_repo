import numpy as np
import pickle

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from flask import Flask, render_template, url_for, request

# loading Pickle Models

dir_name = os.getcwd()
folder_name = "model"
pickle_model_name = "model.pkl"
pickle_trans_name = "Transformer.pkl"

model_path = os.path.join(dir_name, folder_name, pickle_model_name)
trans_path = os.path.join(dir_name, folder_name, pickle_trans_name)

# Pickle
model = pickle.load(open(model_path, "rb"))
Transformer = pickle.load(open(trans_path, "rb"))

app = Flask(__name__)


@app.route("/")
def homePage():
    return render_template("home.html")


@app.route("/predict", methods=["GET","POST"])
def predict():
    try:
        if request.method == "POST":
            PClass = int(request.form.get("PClass"))
            SibSp = int(request.form.get("SibSp"))
            Parch = float(request.form.get("Parch"))
            Sex = request.form.get("Sex").lower()
            AGE = int(request.form.get("AGE"))
            Fare = float(request.form.get("Fare"))

        #Dataframe
        family = int(SibSp + Parch)
        data = {'Pclass': [PClass],
                'Sex': [Sex],
                'Age': [AGE],
                'Fare': [Fare],
                'family': [family]
                }
        df = pd.DataFrame(data)
        print("DataFrame Created")
        transform_data = Transformer.transform(df)
        result = model.predict(transform_data)
        print(result[0])
        return render_template("result.html", result=result[0])
    except:
        render_template("home.html")

if __name__=="__main__":
    app.run(debug=True)

