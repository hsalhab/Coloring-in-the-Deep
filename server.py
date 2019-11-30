from flask import Flask, send_file, jsonify
from flask_cors import CORS, cross_origin
import numpy as np
import boto3
import os
import base64

app = Flask(__name__)
CORS(app)


@app.route('/get-image', methods=['GET'])
def get_image():
    idx = np.random.randint(10830, 10840)
    img_name = "{}.png".format(idx)
    s3 = boto3.client('s3', aws_access_key_id=os.environ['S3_KEY'], aws_secret_access_key=os.environ['S3_SECRET'])
    s3.download_file('tcs-img', img_name, "img.png")
    with open("img.png", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    base64_string = encoded_string.decode('utf-8')
    response = jsonify("image", base64_string)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/')
def index():
    return "hello"

if __name__=="__main__":
    app.run(debug=True)
