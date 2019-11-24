from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, Input, Dropout, Concatenate, GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def get_model_classif_nasnet():
    inputs = Input((64, 64, 3))
    base_model = MobileNet(include_top=False, input_shape=(64, 64, 3)) #, weights=None
    x = base_model(inputs)
    out1 = GlobalMaxPooling2D()(x)
    out2 = GlobalAveragePooling2D()(x)
    out3 = Flatten()(x)
    out = Concatenate(axis=-1)([out1, out2, out3])
    out = Dropout(0.5)(out)
    out = Dense(1, activation="sigmoid", name="3_")(out)
    model = Model(inputs, out)
    model.compile(optimizer=Adam(0.001), loss=binary_crossentropy, metrics=['acc'])
    model.summary()

    return model