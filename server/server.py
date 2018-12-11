from compute import compute_mean_std as compute_function

from flask import Flask, render_template, request
from model import Average
from werkzeug import secure_filename
import os

from model import DensePoseInferenceModel

# Application object
app = Flask(__name__)

# Path to the web application
@app.route('/', methods=['GET', 'POST'])
def index():
    form = Average(request.form)
#     print(form)
    filename = None  # default
    if request.method == 'POST':
        print(request.files)
        dense_pose_inference_model = DensePoseInferenceModel()
        result = dense_pose_inference_model.run(image)
    else:
        result = None

    return render_template("index.html", form=form, result=result)

if __name__ == '__main__':
    app.run(debug=False)