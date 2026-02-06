import glob
import numpy as np
from os import path
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import backend as K
from tensorflow import keras
import warnings
from dataPre import loadCsv, dataset_pre
import pickle
import os
import matplotlib.pyplot as plt   # <-- ADDED
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=Warning)

TL = 4

trainPath_201 = "data/UNSW_NB15_Train201.csv"
trainPath_202 = "data/UNSW_NB15_Train202.csv"
trainPath_203 = "data/UNSW_NB15_Train203.csv"
trainPath_204 = "data/UNSW_NB15_Train204.csv"
trainPath_205 = "data/UNSW_NB15_Train205.csv"

testPath_2 = "data/UNSW_NB15_TestBin.csv"

trainData_201 = loadCsv(trainPath_201)
trainData_202 = loadCsv(trainPath_202)
trainData_203 = loadCsv(trainPath_203)
trainData_204 = loadCsv(trainPath_204)
trainData_205 = loadCsv(trainPath_205)

testData_2 = loadCsv(testPath_2)

trainData01_scaler = trainData_201[:, 0:196]
trainData02_scaler = trainData_202[:, 0:196]
trainData03_scaler = trainData_203[:, 0:196]
trainData04_scaler = trainData_204[:, 0:196]
trainData05_scaler = trainData_205[:, 0:196]
testData_scaler = testData_2[:, 0:196]

scaler = MinMaxScaler()
trainData01_scaler = scaler.fit_transform(trainData01_scaler)
trainData02_scaler = scaler.fit_transform(trainData02_scaler)
trainData03_scaler = scaler.fit_transform(trainData03_scaler)
trainData04_scaler = scaler.fit_transform(trainData04_scaler)
trainData05_scaler = scaler.fit_transform(trainData05_scaler)
testData_scaler = scaler.fit_transform(testData_scaler)

x_train01 = np.reshape(dataset_pre(trainData01_scaler, TL), (-1, TL, 196))
x_train02 = np.reshape(dataset_pre(trainData02_scaler, TL), (-1, TL, 196))
x_train03 = np.reshape(dataset_pre(trainData03_scaler, TL), (-1, TL, 196))
x_train04 = np.reshape(dataset_pre(trainData04_scaler, TL), (-1, TL, 196))
x_train05 = np.reshape(dataset_pre(trainData05_scaler, TL), (-1, TL, 196))
x_test = np.reshape(dataset_pre(testData_scaler, TL), (-1, TL, 196))

y_train01 = trainData_201[:, 196]
y_train02 = trainData_202[:, 196]
y_train03 = trainData_203[:, 196]
y_train04 = trainData_204[:, 196]
y_train05 = trainData_205[:, 196]
y_test = testData_2[:, 196]

shape = x_train01.shape[2]

# ===== METRIC STORAGE (ADDED) =====
rounds_list = []
accuracy_list = []
loss_list = []

def build_base_model():
    model = keras.Sequential([
        keras.layers.Input(shape=(TL, shape)),
        keras.layers.Conv1D(32, 3, activation="relu", padding="same"),
        keras.layers.Conv1D(64, 3, activation="relu", padding="same"),
        keras.layers.Conv1D(128, 3, activation="relu", padding="same"),
        keras.layers.Conv1D(128, 3, strides=2, activation="relu", padding="same"),
        keras.layers.Conv1D(128, 3, strides=2, activation="relu", padding="same"),
        keras.layers.Flatten(),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

def load_or_create_model():
    if path.exists("CentralServer/fl_model.h5"):
        model = keras.models.load_model("CentralServer/fl_model.h5", compile=False)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
    else:
        model = build_base_model()
    return model

def train_and_save(x_train, y_train, server_id):
    model = load_or_create_model()
    model.fit(x_train, y_train, batch_size=500, epochs=1,
              validation_data=(x_test, y_test), verbose=0, shuffle=True)

    os.makedirs("Server", exist_ok=True)
    with open(f"Server/Server{server_id}.pkl", "wb") as f:
        pickle.dump(model.get_weights(), f)

def fl_average():
    weights = []
    for f in glob.glob("Server/*.pkl"):
        with open(f, "rb") as file:
            weights.append(pickle.load(file))
    return [np.mean(w, axis=0) for w in zip(*weights)]

def model_fl(round_id):
    avg = fl_average()
    model = build_base_model()
    model.set_weights(avg)

    loss, acc = model.evaluate(x_test, y_test, verbose=0)

    rounds_list.append(round_id)
    accuracy_list.append(acc)
    loss_list.append(loss)

    print(f"[Round {round_id}] Global Model -> Accuracy: {acc:.4f}, Loss: {loss:.4f}")

    os.makedirs("CentralServer", exist_ok=True)
    model.save("CentralServer/fl_model.h5")


# ===== FEDERATED TRAINING =====
for i in range(3):
    train_and_save(x_train01, y_train01, 1)
    train_and_save(x_train02, y_train02, 2)
    train_and_save(x_train03, y_train03, 3)
    train_and_save(x_train04, y_train04, 4)
    train_and_save(x_train05, y_train05, 5)

    model_fl(i + 1)
    print("Federated Round:", i + 1)
    K.clear_session()

# ===== PLOTTING (ADDED) =====
plt.figure()
plt.plot(rounds_list, accuracy_list)
plt.xlabel("Federated Rounds")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Federated Rounds")
plt.show()

plt.figure()
plt.plot(rounds_list, loss_list)
plt.xlabel("Federated Rounds")
plt.ylabel("Loss")
plt.title("Loss vs Federated Rounds")
plt.show()


def plot_confusion_matrix():
    print("\nGenerating Confusion Matrix for Final Global Model...")

    model = keras.models.load_model("CentralServer/fl_model.h5")

    y_pred_prob = model.predict(x_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).reshape(-1)

    cm = confusion_matrix(y_test[:len(y_pred)], y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Attack"]
    )

    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix – Federated IDS (Final Model)")
    plt.show()


plot_confusion_matrix()
