import os
import pandas as pd
import flask
import yaml
from flask import jsonify, request
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash

from kbclean.detection.adhoc import AdhocDetector

app = flask.Flask(__name__)
auth = HTTPBasicAuth()

users = {
    "mint": generate_password_hash("asf12jkj!%&"),
}

print(app.root_path)
data = yaml.load(
    open(os.path.join(app.root_path, "config.yml")), Loader=yaml.FullLoader
)
app.config.from_mapping(data)
app.config["DEBUG"] = True

detector = AdhocDetector(
    {"host": app.config["ES_HOST"], "port": app.config["ES_PORT"]}
)


@auth.verify_password
def verify_password(username, password):
    if username in users and check_password_hash(users.get(username), password):
        return username


@app.route("/detect", methods=["POST"])
@auth.login_required
def detect():
    df = pd.DataFrame.from_dict(request.json["table"], orient="index").transpose()
    result_df = detector.detect(df)
    for index, row in result_df.iterrows():
        for column in result_df.columns:
            if not row[column]:
                df.at[index, column] =  f"[[[{df.at[index, column]}]]]"
    return jsonify(table=df.to_dict(orient="list"))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=10011)
