from glob import glob

import pandas as pd
import numpy as np  # linear algebra
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

from models import get_model_classif_nasnet
from utils import data_gen, chunker, read_image


labeled_files = glob('/media/ml/data_ml/dogs-vs-cats/train/*.jpg')
test_files = glob('/media/ml/data_ml/dogs-vs-cats/test1/*.jpg')

train, val = train_test_split(labeled_files, test_size=0.1, random_state=101010)

model = get_model_classif_nasnet()

batch_size = 32
h5_path = "model.h5"
checkpoint = ModelCheckpoint(h5_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

_ = model.fit_generator(
    data_gen(train, batch_size),
    validation_data=data_gen(val, batch_size),
    epochs=10, verbose=1,
    callbacks=[checkpoint],
    steps_per_epoch=len(train) // batch_size,
    validation_steps=len(val) // batch_size)

model.load_weights(h5_path)

preds = []
ids = []

for batch in chunker(test_files, batch_size):
    X = [preprocess_input(read_image(x)) for x in batch]
    X = np.array(X)
    preds_batch = model.predict(X).ravel().tolist()
    preds += preds_batch

df = pd.DataFrame({'id': test_files, 'label': preds})
df.to_csv("baseline_nasnet.csv", index=False)
df.head()
