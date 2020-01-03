import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import Adam
import math as math
import time





start_time = time.time()

#load data from csv file
df1 = pd.read_csv("data_monbug/anoSC1_v11_nokey.csv")
# df1 = pd.read_csv("home/matt/Dekstop/ML/data_monbug/anoSC1_v11_nokey.csv")
#check to make sure its okay
# print(df1.head(50))

#split into train and test set
dfTest = df1[df1['Train'] == 0]
dfTrain = df1[df1['Train'] == 1]

#make list of patient ids in train set
patientList = dfTrain["SampleID"].tolist()


#read large text file

df2 = pd.read_csv("data_monbug/eset_HTA20.txt", sep='\s')
# df2 = pd.read_csv("home/matt/Dekstop/ML/data_monbug/eset_HTA20.txt", sep='\s')

#print data
# print(df2.head(10).to_string())



#df2t is the transpose of df2

df2T = df2.T
#make df2 only include patients from trainset stored in df2Train this is our X values

df2Train = df2T[df2T.index.isin(patientList)]

#df2Test predict is dataset for patients who don't have GA but have gene info
df2_Predict = df2T[~df2T.index.isin(patientList)]

#gestiation age Y value
GA_df2Train = [dfTrain[dfTrain['SampleID']==i]['GA'].iloc[0] for i in df2Train.index]


x_train1 = df2Train.to_numpy()
y_train1 = np.asarray(GA_df2Train)

print(x_train1.shape)
print(y_train1.shape)

x_train, x_test, y_train, y_test = train_test_split(x_train1, y_train1, test_size=0.25, random_state=42)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)



y_train=np.reshape(y_train, (-1,1))
y_test=np.reshape(y_test, (-1,1))

scaler_x = MinMaxScaler(feature_range = (0,1)).fit(x_train)
scaler_y = MinMaxScaler(feature_range = (0,1)).fit(y_train)



xscale=scaler_x.transform(x_train)

yscale=scaler_y.transform(y_train)

xscaleTest=scaler_x.transform(x_test)

yscaleTest=scaler_y.transform(y_test)

x_train=xscale
y_train = yscale

x_test = xscaleTest
y_test = yscaleTest







#build model
model = Sequential()
model.add(Dense(128, input_dim=32830,kernel_initializer='glorot_normal',activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()




# model = Sequential()
# model.add(Dense(128, input_shape=(32830,kernel_initializer='glorot_normal',activation='relu'))
# model.summary()

#compile model

adam = Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-8, amsgrad=False)
model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])

history = model.fit(x_train, y_train, epochs=250, batch_size=164, verbose=1)


#predict




# x_test= scaler_x.transform(x_test)
predictions= model.predict(x_test)
# #invert normalize
predictions = scaler_y.inverse_transform(predictions)

y_test = scaler_y.inverse_transform(y_test)



print("X=%s, Predicted=%s" % (x_test[0], predictions[0]))
print("compare from y test set: ",y_test[0])



ig, ax = plt.subplots()
ax.scatter(y_test, predictions)

ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
plt.show()







rmseTrue= predictions-y_test
rmseTrue = rmseTrue*rmseTrue
rmseTrue = rmseTrue.mean()
rmseTrue = math.sqrt(rmseTrue)
print("RMSE TRUE: ",rmseTrue)

elapsed_time = time.time() - start_time
print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))