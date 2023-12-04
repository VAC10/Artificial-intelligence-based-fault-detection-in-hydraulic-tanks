import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.preprocessing import StandardScaler

durus = pd.read_excel("G-3 2023 Duruş.xlsx",sheet_name="B07-1 Final")
final_data=pd.read_excel("final.xlsx")



tox_alarm=final_data.iloc[:,[34]]

lıne_alarm=final_data.iloc[:,[14]]


tox_alarm_0=tox_alarm[tox_alarm["TOX_ALARM"]==0]

tox_alarm_1=tox_alarm[tox_alarm["TOX_ALARM"]==1]

hid1=final_data.iloc[:,[2,3]]

hid2=final_data.iloc[:,[4,5]]

hid3=final_data.iloc[:,[6,7]]

hid4=final_data.iloc[:,[8,9]]

hid5=final_data.iloc[:,[10,11]]

hid6=final_data.iloc[:,[12,13]]

tank6_bas=hid6.iloc[:,[0]]
tank6_sic=hid6.iloc[:,[1]]


# Aşağıdaki fonksiyonlar tox alarmın 0 ve 1 lere ayrılıp hangi sıcaklık ve basınç aralığında 1 aldığını görmek için oluşturuldu 
# ve ardından tox_visulation ile görselleştirildi

def tox_0_basinc():
    tox_0_conc_bas=pd.concat([tank6_bas,tox_alarm_0],axis=1) #bu kısımda tank 6 nın alarm gelmeyen sıcaklıklarını filtreledik
    tox_0_conc_bas.dropna(inplace=True)
    return tox_0_conc_bas
    



def tox_1_basinc():
    tox_1_conc_bas=pd.concat([tank6_bas,tox_alarm_1],axis=1) #bu kısımda tank 6 nın alarm gelmeyen sıcaklıklarını filtreledik
    tox_1_conc_bas.dropna(inplace=True)
    return tox_1_conc_bas


def tox_0_sicaklik():   
    tox_0_conc_sic=pd.concat([tank6_sic,tox_alarm_0],axis=1) #bu kısımda tank 6 nın alarm gelmeyen sıcaklıklarını filtreledik
    tox_0_conc_sic.dropna(inplace=True)
    return tox_0_conc_sic
    tox_1_conc_sic=pd.concat([tank6_sic,tox_alarm_1],axis=1) #bu kısımda tank 6 nın alarm gelmeyen sıcaklıklarını filtreledik
    tox_1_conc_sic.dropna(inplace=True)
    


def tox_1_sicaklik():   
    tox_1_conc_sic=pd.concat([tank6_sic,tox_alarm_1],axis=1) #bu kısımda tank 6 nın alarm gelmeyen sıcaklıklarını filtreledik
    tox_1_conc_sic.dropna(inplace=True)
    return tox_1_conc_sic



"""
bu kısımda excelden bazı operasyonlar yapıp b-07 arıza türlerini filtreledik, başlangıç ve bitiş tarihlerini
tespit ettik. bu süreçte elimde hata olmayan verileride bulundurmam lazımdı 
yani kısacası elimde b-07 bulununan hatalar ve hata olmayan verileri aldım ve bunu final_xslsx dosyasına 
kaydettim veri sayım neredeyse yarı yarıya düştü.
"""
"""


def b071_to_zaman(raw: pd.DataFrame, durus: pd.DataFrame):
    zaman["B07-1"] = ""
    zaman["CREATED_DATE_TIME      "] = pd.to_datetime(raw["CREATED_DATE_TIME      "])
    for i in range(len(durus)):
        bas = durus.loc[i, "Başlangıç Tarih Saat"]
        bit = durus.loc[i, "Bitiş Tarih Saat"]
        dur = durus.loc[i, "Durus dakika"]
        raw.loc[(raw["CREATED_DATE_TIME      "]>=bas) & (raw["CREATED_DATE_TIME      "]<=bit), "B07-1"] += str(i) + f"-({dur})"
    return raw



new_df =b071_to_zaman(df.copy(), durus)
new_df = new_df[~((new_df["B07-1"] == "") & ((new_df["LINE_ALARM"] == 1) | (new_df["TOX_ALARM"] == 1)))]
new_df.sort_values(by="ID     ").to_excel("final_dummy.xlsx", index=False)

"""


def tox_visulation():


    plt.scatter(tox_0_sicaklik(),tox_0_basinc(),color='green', label='Alarm Çalmıyor', marker='o',alpha=0.4)
    
    
    plt.scatter(tox_1_sicaklik(),tox_1_basinc(), color='red', label='Alarm Çalıyor', marker='o',alpha=0.4)
    
    # bu kısımdan itibaren veri keşfi, görselleştirmesi bitti...
    
    
    """
    
    """
    
    # Eksen etiketlerini ve başlığı ekleyin
    plt.xlabel('Sicaklik (°C)')
    plt.ylabel('Basinc (kPa)')
    plt.title('Alarm Durumu ve Basinc-Sicaklik İlişkisi')
    
    # Lejantı ekleyin
    plt.legend()
    
    # Görseli görüntüleyin
    plt.show()

tox_visulation()

x=hid6
y=tox_alarm

# Bu kısımda basınç ve sıcaklık değerlerini scale(ölçeklendirme) işlemi uyguladık bakalım
# accurcay' e ne kadar etki edecek (çok bir etkisi olmadı)


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
 
x_scaled=scaler.fit_transform(x)
print(x_scaled)
 
 
x_scaled_to_df=pd.DataFrame(x_scaled) # ölçeklendirdiğim veriler float türündeler bu yüzden dataframe dönüştürdüm
tank6_bas_scale=x_scaled_to_df.iloc[:,[0]] # basınç ve sıcaklıkları scale ettik
tank6_sic_scale=x_scaled_to_df.iloc[:,[1]]#    "    "     "           "      "  

def scale_0_bas():

    tox_0_scale_bas=pd.concat([tank6_bas_scale,tox_alarm_0],axis=1) #bu kısımda tank 6 nın alarm gelmeyen sıcaklıklarını filtreledik
    tox_0_scale_bas.dropna(inplace=True)
    return tox_0_scale_bas 
    
def scale_1_bas():    
    tox_1_scale_bas=pd.concat([tank6_bas_scale,tox_alarm_1],axis=1) #bu kısımda tank 6 nın alarm gelmeyen sıcaklıklarını filtreledik
    tox_1_scale_bas.dropna(inplace=True)
    return tox_1_scale_bas
    
    
   
def scale_0_sic(): 
    tox_0_scale_sic=pd.concat([tank6_sic_scale,tox_alarm_0],axis=1) #bu kısımda tank 6 nın alarm gelmeyen sıcaklıklarını filtreledik
    tox_0_scale_sic.dropna(inplace=True)
    return tox_0_scale_sic
    
def scale_1_sic():
    tox_1_scale_sic=pd.concat([tank6_sic_scale,tox_alarm_1],axis=1) #bu kısımda tank 6 nın alarm gelmeyen sıcaklıklarını filtreledik
    tox_1_scale_sic.dropna(inplace=True)
    return tox_1_scale_sic



def visual_scale_Sic_bas():# scale edilmiş sıcaklık ve basınç değerlerinin görselleştirilmesi

    plt.scatter(scale_0_bas(),scale_0_sic(), color='green', label='Alarm Çalmıyor', marker='o',alpha=0.4)
    
    
    plt.scatter(scale_1_bas(),scale_1_sic(), color='red', label='Alarm Çalıyor', marker='o',alpha=0.4)
    
    plt.xlabel('Sicaklik (°C)')
    plt.ylabel('Basinc (kPa)')
    plt.title('Alarm Durumu ve Basinc-Sicaklik İlişkisi')
    
    # Lejantı ekleyin
    plt.legend()
    
    # Görseli görüntüleyin
    plt.show()
    
visual_scale_Sic_bas()    
# bu kısımdan itibaren veri keşfi, görselleştirmesi bitti...


# Eksen etiketlerini ve başlığı ekleyin


# Görseli görüntüleyin
plt.show()







#makine öğrenmesi modelleri deneme
"""
x=hid6 
y=tox_alarm # tox hattı için
"""

x1 = hid1
x2 = hid2
x3 = hid3
x4 = hid4
x5 = hid5
y=lıne_alarm

# OVER-UNDER SAMPLE METODLARI
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
    adasyn = ADASYN(sampling_strategy=0.5,random_state=42)
    x_resampled, y_resampled = adasyn.fit_resample(x, y)
    return x_resampled,y_resampled


def borderline_smote():
    from imblearn.over_sampling import BorderlineSMOTE
    smote = BorderlineSMOTE(sampling_strategy=0.5,random_state=42)
    x_resampled, y_resampled = smote.fit_resample(x, y)
    return x_resampled,y_resampled

x_resampled = pd.concat([x1, x2, x3, x4, x5], axis=1)


x_resampled, y_resampled = smote_over_sampl(x_resampled,y)


x_resampled,y_resampled=under_sampling(x_resampled, y)






"""
x_resampled,y_resampled=smote_over_sampl() # önce over sample yapıp 
x_resampled,y_resampled=under_sampling(x_resampled, y_resampled)# ardından under sample yaptım ki üretilen sentetik verilere göre karar verilmesin, küme küçülsün.
print(x_resampled.shape)
print(y_resampled.shape)
"""


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
x_train,x_test,y_train,y_test=train_test_split(x_resampled,y_resampled,test_size=0.33,random_state=0) # test ve traini ayırdık
x_train.shape,x_test.shape,y_train.shape,y_test.shape

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

"""
yaptığımız örneklem arttırma-azaltma 
tekniklerini bir de grafik üzerinde görelim. Bakalım nasıl yapılmış

"""

resample_sic=x_resampled.iloc[:,[1]]
resample_bas=x_resampled.iloc[:,[0]]

resample_tox_0=y_resampled[y_resampled["TOX_ALARM"]==0]
resample_tox_1=y_resampled[y_resampled["TOX_ALARM"]==1]

def res_0_basinc():
    res_0_conc_bas=pd.concat([resample_bas,resample_tox_0],axis=1) 
    res_0_conc_bas.dropna(inplace=True)
    return res_0_conc_bas
    



def res_1_basinc():
    res_1_conc_bas=pd.concat([resample_bas,resample_tox_1],axis=1) 
    res_1_conc_bas.dropna(inplace=True)
    return res_1_conc_bas


def res_0_sicaklik():   
    res_0_conc_sic=pd.concat([resample_sic,resample_tox_0],axis=1) 
    res_0_conc_sic.dropna(inplace=True)
    return res_0_conc_sic
    


def res_1_sicaklik():   
    res_1_conc_sic=pd.concat([resample_sic,resample_tox_1],axis=1) 
    res_1_conc_sic.dropna(inplace=True)
    return res_1_conc_sic


def visulation_over_under_sample():
    plt.scatter(res_0_sicaklik(),res_0_basinc(),color='green', label='Alarm Çalmıyor', marker='o',alpha=0.4)
    
    
    plt.scatter(res_1_sicaklik(),res_1_basinc(), color='red', label='Alarm Çalıyor', marker='o',alpha=0.4)
    
    # bu kısımdan itibaren veri keşfi, görselleştirmesi bitti...
    
    
    """
    
    """
    
    # Eksen etiketlerini ve başlığı ekleyin
    plt.xlabel('Sicaklik (°C)')
    plt.ylabel('Basinc (kPa)')
    plt.title('Alarm Durumu ve Basinc-Sicaklik İlişkisi')
    
    # Lejantı ekleyin
    plt.legend()
    
    # Görseli görüntüleyin
    plt.show()
    

visulation_over_under_sample()


def logistic_Reg():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    lr=LogisticRegression()
    lr.fit(x_train,y_train)
    y_pred=lr.predict(x_test)
    print("Accuracy score {}".format(accuracy_score(y_test,y_pred)))
    cm7=confusion_matrix(y_test, y_pred)
    print(f'logistic regresyon Karmaşıklık Matrisi:\n{cm7}\n')
    # algoritmaları test ettiğimde dengesiz(imbalanced data) sıkıntısıyla karşı karşıya kaldım.
    cm_normalized = cm7.astype('float') / cm7.sum(axis=1)[:, np.newaxis]  # Yüzdelik hesaplama
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized)
    disp.plot(include_values=True, cmap='Blues', xticks_rotation='horizontal', values_format='.2%')
    plt.show()
    """
    sns.heatmap(cm7, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Tahmin Edilen Sınıf")
    plt.ylabel("Gerçek Sınıf")
    plt.title("Karışıklık Matrisi")
    plt.show()
    """

def knn_neighbors(): 
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score,roc_auc_score
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    clf=KNeighborsClassifier(n_neighbors=20)
    clf.fit(x_train,y_train)
    y_pred=clf.predict(x_test)
    print("Accuracy score {}".format(accuracy_score(y_test,y_pred)))
    print("ROC AUC score {}".format(roc_auc_score(y_test,y_pred)))
    #pd.DataFrame(data={"Y_Actual":y_test,"Y_Predict":y_pred})
    cm1=confusion_matrix(y_test, y_pred)
    cm_normalized = cm1.astype('float') / cm1.sum(axis=1)[:, np.newaxis]  # Yüzdelik hesaplama
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized)
    disp.plot(include_values=True, cmap='Blues', xticks_rotation='horizontal', values_format='.2%')
    plt.show()
    
    """
    print(f'knn Karmaşıklık Matrisi:\n{cm}\n')   # algoritmaları test ettiğimde dengesiz(imbalanced data) sıkıntısıyla karşı karşıya kaldım.
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Tahmin Edilen Sınıf")
    plt.ylabel("Gerçek Sınıf")
    plt.title("Karışıklık Matrisi")
    plt.show()
    """


def naive_bayes():
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    clf=GaussianNB()
    clf.fit(x_train,y_train)
    y_pred=clf.predict(x_test)
    print("Accuracy score {}".format(accuracy_score(y_test,y_pred)))
    print("ROC AUC score {}".format(roc_auc_score(y_test,y_pred)))
    cm1=confusion_matrix(y_test, y_pred)
    print(f'naive bayes Karmaşıklık Matrisi:\n{cm1}\n')
    cm_normalized = cm1.astype('float') / cm1.sum(axis=1)[:, np.newaxis]  # Yüzdelik hesaplama
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized)
    disp.plot(include_values=True, cmap='Blues', xticks_rotation='horizontal', values_format='.2%')
    plt.show()
    """
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    
    plt.xlabel("Tahmin Edilen Sınıf")
    plt.ylabel("Gerçek Sınıf")
    plt.title("Karışıklık Matrisi")
    plt.show()
    """



def random_forest():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    rf=RandomForestClassifier(n_estimators=60)
    rf.fit(x_train, y_train)
    y_pred=rf.predict(x_test)
    print("Accuracy score {}".format(accuracy_score(y_test,y_pred)))
    print("ROC AUC score {}".format(roc_auc_score(y_test,y_pred)))
    cm1=confusion_matrix(y_test, y_pred)
    cm_normalized = cm1.astype('float') / cm1.sum(axis=1)[:, np.newaxis]  # Yüzdelik hesaplama
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized)
    disp.plot(include_values=True, cmap='Blues', xticks_rotation='horizontal', values_format='.2%')
    plt.show()
    """
    print(f'naive bayes Karmaşıklık Matrisi:\n{cm1}\n')
    sns.heatmap(cm1, annot=True, fmt="d", cmap="Blues")
    
    plt.xlabel("Tahmin Edilen Sınıf")
    plt.ylabel("Gerçek Sınıf")
    plt.title("Karışıklık Matrisi")
    plt.show()
   """





def decision_tree():
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    dt=DecisionTreeClassifier()
    dt.fit(x_train,y_train)
    y_pred=dt.predict(x_test)
    print("Accuracy score {}".format(accuracy_score(y_test,y_pred)))
    cm=confusion_matrix(y_test, y_pred)
    print(f'decision tree Karmaşıklık Matrisi:\n{cm}\n')
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Yüzdelik hesaplama 
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized)
    disp.plot(include_values=True, cmap='Blues', xticks_rotation='horizontal', values_format='.2%')  
    plt.show()
    
    """
    print(f'decision tree Karmaşıklık Matrisi:\n{cm}\n')
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    
    plt.xlabel("Tahmin Edilen Sınıf")
    plt.ylabel("Gerçek Sınıf")
    plt.title("Karışıklık Matrisi")
    plt.show()
    """
    
def Gradient_boosting():
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    clf=GradientBoostingClassifier(n_estimators=50,learning_rate=0.02)
    clf.fit(x_train,y_train)
    y_pred=clf.predict(x_test)
    print("Accuracy score {}".format(accuracy_score(y_test,y_pred)))
    print("ROC AUC score {}".format(roc_auc_score(y_test,y_pred)))
    #pd.DataFrame(data={"Y_Actual":y_test,"Y_Predict":y_pred})
   
    cm = confusion_matrix(y_test, y_pred)
    print(f'gradient boosting Karmaşıklık Matrisi:\n{cm}\n')
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Yüzdelik hesaplama
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized)
    disp.plot(include_values=True, cmap='Blues', xticks_rotation='horizontal', values_format='.2%')
    plt.show()
    
    
    """
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    
    
    plt.xlabel("Tahmin Edilen Sınıf")
    plt.ylabel("Gerçek Sınıf")
    plt.title("Karışıklık Matrisi")
    plt.show()
    """

# populer olan xgb machine learning algoritmasını kullanalım
def xgboost_algoritması(): # Not: bu algoritmayı ilk defa görüyorum.
    
    from xgboost import XGBClassifier
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    xgb = XGBClassifier().fit(x_train, y_train)
    y_pred=xgb.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'xgb Accuracy: {round(accuracy, 3)}')
    
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Yüzdelik hesaplama
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized)
    disp.plot(include_values=True, cmap='Blues', xticks_rotation='horizontal', values_format='.2%')
    plt.show()
   
    """
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    
    
    plt.xlabel("Tahmin Edilen Sınıf")
    plt.ylabel("Gerçek Sınıf")
    plt.title("Karışıklık Matrisi")
   
    plt.show()
 """
    
 
def svm(): #smote la %84 acc aldım
    from sklearn.svm import SVC
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    
    svm_classifier = SVC(kernel='linear', random_state=42)
    svm_classifier.fit(x_train, y_train)  
    y_pred=svm_classifier.predict(x_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Svm Accuracy: {round(accuracy, 3)}')
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Yüzdelik hesaplama
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized)
    disp.plot(include_values=True, cmap='Blues', xticks_rotation='horizontal', values_format='.2%')
    plt.show()
    """
    sns.heatmap(cm55, annot=True, fmt="d", cmap="Blues")
     
    plt.xlabel("Tahmin Edilen Sınıf")
    plt.ylabel("Gerçek Sınıf")
    plt.title("Karışıklık Matrisi")
   
    plt.show()
    """
 
    

logistic_Reg()
xgboost_algoritması()
knn_neighbors()
naive_bayes() # çok vasat kaldı.
decision_tree()
Gradient_boosting()
random_forest()
svm()






"""
algoritmaları test ettiğimde dengesiz(imbalanced data) sıkıntısıyla karşı karşıya kaldım.
yani 0 gelen veriler 1 gelen verilerden 40.000 daha fazla olduğundan model bir yerden sonra çoğunluk olan sınıfı tahmin ediyor ve bu da
yüksek accuracy vermesine rağmen sonuç istenilen gibi olmuyor ve yapılan tahminler çoğunluk olan veriye göre yapılmış
"""









# 




# smote kütüphanesi, normalization, data modele nasıl verilir, hangi modeli kullanmalıyım, under sample, over sample 

# önce undersampling'i dene

#1.2. SMOTE (Synthetic Minority Oversampling Technique)

#Bu teknik, azınlık sınıfındaki örneklerin kopyalarını oluşturmanın yanı sıra, bu örneklerin **bazı özelliklerinde rastgele değişiklikler yaparak yeni örnekler oluşturur**. Bu sayede, veri setindeki azınlık sınıfına ait örneklerin sayısını artırırken, yeni ve çeşitli verilerin de oluşturulmasını sağlar.




































    





