import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

plt.margins(0, 0)

df = pd.read_csv('heart.csv')

# Setting up x and y
x = df.drop('target', axis=1).values
y = df['target'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.95)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Setting up model
model = Sequential()

model.add(Dense(13, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(9, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(7, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(1, activation='sigmoid'))  # For binary classification
model.compile(loss='binary_crossentropy', optimizer='adam')

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

# model = load_model("model")

model.fit(x_train, y_train, epochs=600, validation_data=(x_test, y_test), callbacks=[early_stop])

pd.DataFrame(model.history.history).plot()
plt.show()

# To use best model so far:
# model = load_model('cancer_classifier.h5')

preds = model.predict_classes(x_test)

classif_rep = pd.DataFrame(classification_report(y_test, preds, output_dict=True))

print(classif_rep)

savemodel = input("Save model? [Y/N]")

if savemodel.upper() == "Y":
    model.save("model")
