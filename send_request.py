import requests

img_path = "/media/ml/data_ml/dogs-vs-cats/test1/132.jpg"

headers = {
    'Content-Type': 'application/octet-stream'
}

image_data = open(img_path, "rb").read()

r = requests.post("http://0.0.0.0:5000/predict", headers=headers, data=image_data)

print(r.text)