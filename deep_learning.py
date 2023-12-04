# -*- coding: utf-8 -*-
import os 
import pandas as pd
import cv2
import numpy as np
import pickle
from keras.layers import Conv2D,MaxPooling2D,Activation,Dense,Flatten
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from keras.layers import Dropout
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
durus = pd.read_excel("G-3 2023 Duruş.xlsx",sheet_name="B07-1 Final")
data=pd.read_excel("final.xlsx")

tox_alarm=data.iloc[:,[34]]



tox_alarm_0=tox_alarm[tox_alarm["TOX_ALARM"]==0]

tox_alarm_1=tox_alarm[tox_alarm["TOX_ALARM"]==1]

hid1=data.iloc[:,[2,3]]

hid2=data.iloc[:,[4,5]]

hid3=data.iloc[:,[6,7]]

hid4=data.iloc[:,[8,9]]

hid5=data.iloc[:,[10,11]]

lıne_alarm=data.iloc[:,[14]]



hid6=data.iloc[:,[12,13]]

tank6_bas=hid6.iloc[:,[0]]
tank6_sic=hid6.iloc[:,[1]]




"""

x=hid6
y=tox_alarm
"""
"""
pressure_values = tank6_bas["HYRAULIC_TANK_6_YAG_BASINCI"].values
pressure_min = 190 - 10
pressure_max = 190 + 10

normalized_pressure = (pressure_values - pressure_min) / (pressure_max - pressure_min)
  

temperature_min = 50 - 5
temperature_max = 50 + 5


temperature_values = tank6_sic["HYRAULIC_TANK_6_YAG_SICAKLIGI"].values
normalized_temperature = (temperature_values - temperature_min) / (temperature_max - temperature_min)

normalized_features = np.column_stack((normalized_pressure, normalized_temperature))

  

x=normalized_features
y=tox_alarm
"""
# OVER-UNDER SAMPLE METODLARI

x1 = hid1
x2 = hid2
x3 = hid3
x4 = hid4
x5 = hid5
y=lıne_alarm













"""
min_pressure = -10
max_pressure =190
min_temperature = -5
max_temperature = 50


hid1['normalized_pressure'] = (hid1['HYRAULIC_TANK_1_YAG_BASINCI'] - min_pressure) / (max_pressure - min_pressure)
hid1['normalized_temperature'] = (hid1['HYRAULIC_TANK_1_YAG_SICAKLIGI'] - min_temperature) / (max_temperature - min_temperature)
hid1.drop('HYRAULIC_TANK_1_YAG_BASINCI', axis=1, inplace=True)# yukarıdaki işlemin aynısını burada da yaptım
hid1.drop('HYRAULIC_TANK_1_YAG_SICAKLIGI', axis=1, inplace=True)# yukarıdaki işlemin aynısını burada da yaptım


hid2['normalized_pressure'] = (hid2['HYRAULIC_TANK_2_YAG_BASINCI'] - min_pressure) / (max_pressure - min_pressure)
hid2['normalized_temperature'] = (hid2['HYRAULIC_TANK_2_YAG_SICAKLIGI'] - min_temperature) / (max_temperature - min_temperature)
hid2.drop('HYRAULIC_TANK_2_YAG_BASINCI', axis=1, inplace=True)# yukarıdaki işlemin aynısını burada da yaptım
hid2.drop('HYRAULIC_TANK_2_YAG_SICAKLIGI', axis=1, inplace=True)# yukarıdaki işlemin aynısını burada da yaptım


hid3['normalized_pressure'] = (hid3['HYRAULIC_TANK_3_YAG_BASINCI'] - min_pressure) / (max_pressure - min_pressure)
hid3['normalized_temperature'] = (hid3['HYRAULIC_TANK_3_YAG_SICAKLIGI'] - min_temperature) / (max_temperature - min_temperature)
hid3.drop('HYRAULIC_TANK_3_YAG_BASINCI', axis=1, inplace=True)# yukarıdaki işlemin aynısını burada da yaptım
hid3.drop('HYRAULIC_TANK_3_YAG_SICAKLIGI', axis=1, inplace=True)# yukarıdaki işlemin aynısını burada da yaptım


hid4['normalized_pressure'] = (hid4['HYRAULIC_TANK_4_YAG_BASINCI'] - min_pressure) / (max_pressure - min_pressure)
hid4['normalized_temperature'] = (hid4['HYRAULIC_TANK_4_YAG_SICAKLIGI'] - min_temperature) / (max_temperature - min_temperature)
hid4.drop('HYRAULIC_TANK_4_YAG_BASINCI', axis=1, inplace=True)# yukarıdaki işlemin aynısını burada da yaptım
hid4.drop('HYRAULIC_TANK_4_YAG_SICAKLIGI', axis=1, inplace=True)# yukarıdaki işlemin aynısını burada da yaptım




hid5['normalized_pressure'] = (hid5['HYRAULIC_TANK_5_YAG_BASINCI'] - min_pressure) / (max_pressure - min_pressure)
hid5['normalized_temperature'] = (hid5['HYRAULIC_TANK_5_YAG_SICAKLIGI'] - min_temperature) / (max_temperature - min_temperature)
hid5.drop('HYRAULIC_TANK_5_YAG_BASINCI', axis=1, inplace=True)# yukarıdaki işlemin aynısını burada da yaptım
hid5.drop('HYRAULIC_TANK_5_YAG_SICAKLIGI', axis=1, inplace=True)# yukarıdaki işlemin aynısını burada da yaptım
"""








def under_sampling(x,y):
    from imblearn.under_sampling import RandomUnderSampler
    # Random Undersampling bu metotla azınlık olan sınıfımı rastgele örneklem çıkarımı yaparak çoğunluğa uyarladım
    rus = RandomUnderSampler(sampling_strategy="auto") # bu yöntemi de denedim fakat istedğim sonucu yine elde edemedim. önceki sonuca göre accuracy düşse de confusion daha iyi..
    x_resampled, y_resampled = rus.fit_resample(x, y)
    return x_resampled,y_resampled

def random_over_sampl():
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler()
    x_resampled, y_resampled = ros.fit_resample(x,y)
    return x_resampled,y_resampled

def smote_over_sampl(x,y):    
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(sampling_strategy=0.5,random_state=42)# *sampling strategy ! ! !(yöntemimiz çok sayıda sentetik "yapay" veri üretiyor. bizim çekindiğimiz şey makinemizin bu sentetik verilere göre öğrenmesi bundan dolayı),
   # strategy_sampling ile makineye çoğunluk sınıfın % 50 sinin örneğini al ve bunlardan veri üret
    x_resampled, y_resampled = smote.fit_resample(x,y)
    return x_resampled,y_resampled  


def Adasyn_over_sampl(x,y):
    from imblearn.over_sampling import ADASYN #Adasyn over_sampling modelinde decision tree en yüksek accurac'i verdi.
    adasyn = ADASYN(sampling_strategy=0.5)
    x_resampled, y_resampled = adasyn.fit_resample(x, y)
    return x_resampled,y_resampled


def borderline_smote():
    from imblearn.over_sampling import BorderlineSMOTE
    smote = BorderlineSMOTE(sampling_strategy=0.5)
    x_resampled, y_resampled = smote.fit_resample(x, y)
    return x_resampled,y_resampled





x_resampled = pd.concat([x1, x2, x3, x4, x5], axis=1)


x_resampled, y_resampled = smote_over_sampl(x_resampled,y)


x_resampled,y_resampled=under_sampling(x_resampled, y)

timestep=5
X=[]
Y=[]


for i in range(len(x_resampled)-timestep):
    X.append(x_resampled[i:i+timestep])
    Y.append(x_resampled[i+timestep])
    
X=np.array(X)
Y=np.array(Y)


"""
x_resampled, y_resampled = smote_over_sampl()


x_resampled,y_resampled=under_sampling(x_resampled, y_resampled)



x_train,x_test,y_train,y_test=train_test_split(x_resampled,y_resampled,test_size=0.2,random_state=42)

"""
x_train,x_test,y_train,y_test=train_test_split(x_resampled,y_resampled,test_size=0.2,random_state=42)


def full_connected():# burada units diye verdiğim özellikler feature sayısı yani makinenin öğrenme sürecindeki çıkaracağı yeni özellik sayısı.
    model = Sequential()
    # Giriş katmanı (Input Layer)
    model.add(Dense(units=64, activation='relu', input_dim=x_train.shape[1])) 
    
    # İlk gizli katman. Yine 64 nöron içerir ve ReLU aktivasyonu kullanır.
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(0.2))
    
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(0.2))
    
    # Dropout, aşırı öğrenmeyi azaltmaya yardımcı olur. Bu katmanda %20'sini devre dışı bırakırız.ularız.
    
    # İkinci gizli katman. 32 nöron içerir ve yine ReLU aktivasyonunu kullanır.
    # Yine %20'lik bir dropout uygularız.
    # Çıkış Katmanı (Output Layer)
    model.add(Dense(units=1, activation='sigmoid'))
    # Çıkış katmanında sadece 1 nöron kullanırız, çünkü bu ikili sınıflandırma problemi için uygundur.
    # Aktivasyon fonksiyonu olarak sigmoid kullanarak sonucun 0 ile 1 arasında bir olasılık değeri olmasını sağlarız.
    # Modeli derleyelim
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # - optimizer: Adam optimizer kullanılıyor, genellikle etkili bir seçenektir.
    # - loss: Çıkış katmanında sigmoid aktivasyonu kullandığımız için "binary_crossentropy" kullanıyoruz.
    # - metrics: Modelin doğruluk metriği olarak "accuracy" kullanılır.
    
    # Model özetini yazdıralım
    model.summary()
    
    # Modeli derledikten sonra eğitim verileriyle fit edin
    history = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)
    
    # Eğitim sonucunda test verileri üzerinde modeli değerlendirin
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Kaybı: {test_loss}, Test Doğruluk: {test_accuracy}")
    
    
    
    
    
    # Eğitim sırasında kaydedilen loss ve accuracy değerlerini alın
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    
    # Loss değerlerini çizdirin
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend()
    plt.title('Loss')
    plt.xlabel("epoch")
    
    # Accuracy değerlerini çizdirin
    plt.figure()
    plt.plot(train_accuracy, label='Training Accuracy')
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy')
    
    plt.show()
    
    predictions = model.predict(x_test)
    threshold = 0.5
    binary_predictions = (predictions > threshold).astype(int)
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    
    cm=confusion_matrix(y_test,binary_predictions)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Yüzdelik hesaplama
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized)
    disp.plot(include_values=True, cmap='Blues', xticks_rotation='horizontal', values_format='.2%')
    plt.show()


def lstm():
    model = Sequential()
    model.add(LSTM(units=32, input_shape=(x_train.shape[1],1), return_sequences=True))
    
    model.add(LSTM(units=32, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=32, return_sequences=True))
    model.add(Dense(units=1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # - optimizer: Adam optimizer kullanılıyor, genellikle etkili bir seçenektir.
    # - loss: Çıkış katmanında sigmoid aktivasyonu kullandığımız için "binary_crossentropy" kullanıyoruz.
    # - metrics: Modelin doğruluk metriği olarak "accuracy" kullanılır.
    
    # Model özetini yazdıralım
    model.summary()
    
    # Modeli derledikten sonra eğitim verileriyle fit edin
    history = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)
    
    # Eğitim sonucunda test verileri üzerinde modeli değerlendirin
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Kaybı: {test_loss}, Test Doğruluk: {test_accuracy}")
    
        
    
    # Eğitim sırasında kaydedilen loss ve accuracy değerlerini alın
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    
    # Loss değerlerini çizdirin
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend()
    plt.title('Loss')
    plt.xlabel("epoch")
    
    # Accuracy değerlerini çizdirin
    plt.figure()
    plt.plot(train_accuracy, label='Training Accuracy')
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy')
    
    plt.show()
    
    
    
    
    
    
    predictions = model.predict(x_test)
    threshold = 0.5
    binary_predictions = (predictions > threshold).astype(int)
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    
    cm=confusion_matrix(y_test,binary_predictions)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Yüzdelik hesaplama
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized)
    disp.plot(include_values=True, cmap='Blues', xticks_rotation='horizontal', values_format='.2%')
    plt.show()
    
    
    print(f'CNN LSTM:\n{cm}\n')
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    
    plt.xlabel("Tahmin Edilen Sınıf")
    plt.ylabel("Gerçek Sınıf")
    plt.title("Karışıklık Matrisi")
    plt.show()


def cnn_lstm(): # hibrit model. %92 acc aldım. confusion %94-%94
    model = Sequential()
    
    # CNN Katmanı
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(x_train.shape[1],1)))
    model.add(MaxPooling1D(pool_size=2))
    
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(x_train.shape[1],1)))
    model.add(MaxPooling1D(pool_size=2))
    
    model.add(Dropout(0.2))
    
    
    # LSTM Katmanı
    model.add(LSTM(units=64, return_sequences=True))
    model.add(LSTM(units=64))  # İsterseniz bu katmanı kullanmayabilirsiniz
    
    # Tam Bağlantı Katmanları
    model.add(Dense(units=32, activation='relu'))
    model.add(Dropout(0.5))  # Aşırı öğrenmeyi önlemek için Dropout
    
    # Çıkış Katmanı
    model.add(Dense(units=1, activation='sigmoid'))
    
    
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
      # Modeli derledikten sonra eğitim verileriyle fit edin
    history = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)
      
      # Eğitim sonucunda test verileri üzerinde modeli değerlendirin
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Kaybı: {test_loss}, Test Doğruluk: {test_accuracy}")
      
          
    # Eğitim sırasında kaydedilen loss ve accuracy değerlerini alın
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    
    # Loss değerlerini çizdirin
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend()
    plt.title('Loss')
    plt.xlabel("epoch")
    
    # Accuracy değerlerini çizdirin
    plt.figure()
    plt.plot(train_accuracy, label='Training Accuracy')
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy')
    
    plt.show()  
    
      
    model.save('egitilmis_model.h5')
    predictions = model.predict(x_test)
    threshold = 0.5
    binary_predictions = (predictions > threshold).astype(int)
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    
    cm=confusion_matrix(y_test,binary_predictions)
    
    print(f'CNN LSTM:\n{cm}\n')
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    
    plt.xlabel("Tahmin Edilen Sınıf")
    plt.ylabel("Gerçek Sınıf")
    plt.title("Karışıklık Matrisi")
    plt.show()
    

    """
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Yüzdelik hesaplama
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized)
    disp.plot(include_values=True, cmap='Blues', xticks_rotation='horizontal', values_format='.2%')
    plt.show()
    """


# rnn

# RNN modelini oluşturalım
def rnn():
    from tensorflow.keras.layers import SimpleRNN, Dense
    model = Sequential()
    
    
    model.add(SimpleRNN(64, activation='relu', input_shape=(x_train.shape[1],1,),return_sequences=True)) 
    model.add(Dropout(0.2))
    
    model.add(SimpleRNN(units=64, activation='relu',return_sequences=True)) 
    model.add(Dropout(0.2))
    
    model.add(Dense(units=1, activation='sigmoid'))  # İkili sınıflandırma için sigmoid aktivasyonu
    
    
    
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
      # Modeli derledikten sonra eğitim verileriyle fit edin
    history = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)
      
      # Eğitim sonucunda test verileri üzerinde modeli değerlendirin
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Kaybı: {test_loss}, Test Doğruluk: {test_accuracy}")
      
          
    # Eğitim sırasında kaydedilen loss ve accuracy değerlerini alın
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    
    # Loss değerlerini çizdirin
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend()
    plt.title('Loss')
    plt.xlabel("epoch")
    
    # Accuracy değerlerini çizdirin
    plt.figure()
    plt.plot(train_accuracy, label='Training Accuracy')
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy')
    
    plt.show()  
    
      
      
      
    predictions = model.predict(x_test)
    threshold = 0.5
    binary_predictions = (predictions > threshold).astype(int)
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    
    cm=confusion_matrix(y_test,binary_predictions)
    
    print(f'naive bayes Karmaşıklık Matrisi:\n{cm}\n')
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    
    plt.xlabel("Tahmin Edilen Sınıf")
    plt.ylabel("Gerçek Sınıf")
    plt.title("Karışıklık Matrisi")
    plt.show()
    
    
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Yüzdelik hesaplama
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized)
    disp.plot(include_values=True, cmap='Blues', xticks_rotation='horizontal', values_format='.2%')
    plt.show()



















































