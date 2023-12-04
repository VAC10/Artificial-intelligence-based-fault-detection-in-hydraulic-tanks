import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

data=pd.read_excel("raw.xlsx")
"""
data_time=pd.read_excel("G-3 2023 Duruş.xlsx")



data_time=data_time.dropna()# nan olan verileri kaldırdık.
print(data.isnull().values.any())
"""
data.head()
print(data.isnull().values.any())

print(data.describe())

data.info()

dizi=data.iloc[:,2:14]

lıne_alarm=data.iloc[:,[14]]

basinc=dizi.iloc[:, [0, 2, 4, 6, 8,]]

sicaklik=dizi.iloc[:,[1,3,5,7,9,]]

m=pd.concat([basinc,sicaklik],axis=1)

all_concate=pd.concat([m,lıne_alarm],axis=1)





corr_matrix = dizi.corr()
plt.figure(figsize=(20,15))
sns.heatmap(corr_matrix, annot=True)
plt.show()


"""Korelasyon matrisine line_auto eklenince basınç 
ve sıcaklık değerleriyle doğru orantılı olduğu görülüyor 
bu bizim başta bulduğumuz ters orantı ilkesine aykırı.. 
bence sütündan kaldırılması gerek
"""

# istasyonda üretim süresinin alarma bir etkisi varmı?

hid1=data.iloc[:,[2,3,26,14]]

hid2=data.iloc[:,[4,5,27,14]]

hid3=data.iloc[:,[6,7,28,14]]

hid4=data.iloc[:,[8,9,29,30,31,14]]

hid5=data.iloc[:,[10,11,32,33,20,19,18,17,14]]

hid6=data.iloc[:,[12,13,34]]




corr_matrix = hid1.corr()
plt.figure(figsize=(20,15))
sns.heatmap(corr_matrix, annot=True)
plt.show()


corr_matrix = hid2.corr()
plt.figure(figsize=(20,15))
sns.heatmap(corr_matrix, annot=True)
plt.show()

corr_matrix = hid3.corr()
plt.figure(figsize=(20,15))
sns.heatmap(corr_matrix, annot=True)
plt.show()


corr_matrix = hid4.corr()
plt.figure(figsize=(20,15))
sns.heatmap(corr_matrix, annot=True)
plt.show()

corr_matrix = hid5.corr()
plt.figure(figsize=(20,15))
sns.heatmap(corr_matrix, annot=True)
plt.show()

corr_matrix = hid6.corr()
plt.figure(figsize=(20,15))
sns.heatmap(corr_matrix, annot=True)
plt.show()




tox_alarm=hid6.iloc[:,[2]]

tox_1=tox_alarm[tox_alarm['TOX_ALARM']==1]


tox_0=tox_alarm[tox_alarm['TOX_ALARM']==0]


alarm_df=lıne_alarm[lıne_alarm['LINE_ALARM']==1]
alarm_df_0=lıne_alarm[lıne_alarm['LINE_ALARM']==0]

basinc_0=pd.concat([basinc,alarm_df_0],axis=1)
sicaklik_0=pd.concat([sicaklik,alarm_df_0],axis=1)

basinc_1=pd.concat([basinc,alarm_df],axis=1)
sicaklik_1=a=pd.concat([sicaklik,alarm_df],axis=1)

tank1_sic=sicaklik_1.iloc[:,[0]]
tank1_bas=basinc_1.iloc[:,[0]]

tank2_sic=sicaklik_1.iloc[:,[1]]
tank2_bas=basinc_1.iloc[:,[1]]

tank3_sic=sicaklik_1.iloc[:,[2]]
tank3_bas=basinc_1.iloc[:,[2]]


tank4_sic=sicaklik_1.iloc[:,[3]]
tank4_bas=basinc_1.iloc[:,[3]]


tank5_sic=sicaklik_1.iloc[:,[4]]
tank5_bas=basinc_1.iloc[:,[4]]

tank6_sic=hid6.iloc[:,[1]]
tank6_bas=hid6.iloc[:,[0]]

ayir2_=data.iloc[:,[14]]


sicaklik1_drop=sicaklik_1.drop('LINE_ALARM', axis=1)
alarm_11=sicaklik_1.iloc[:,[5]]
alarm_00=basinc_0.iloc[:,[5]]

birlestir=pd.concat([tank1_sic,alarm_11],axis=1)


sicaklik_1=a=pd.concat([sicaklik,alarm_df],axis=1)


tox_1_concate_sic=pd.concat([tank6_sic,tox_1],axis=1)
tox_1_concate_sic.dropna(inplace=True)

tox_1_concate_sic.drop('TOX_ALARM', axis=1, inplace=True) # Burada sicaklik kaydım 14000 idi alarm verilerim ise 49 bin idi boyutlar uyuşmadığı için plot çizilmedi 
# bu yüzden sicaklikta tox alarımda 1 e denk gelen satırları aldım ve boyutu 49 bine indirdim artık çizdirebilirim


tox_1_concate_bas=pd.concat([tank6_bas,tox_1],axis=1)
tox_1_concate_bas.dropna(inplace=True)
tox_1_concate_bas.drop('TOX_ALARM', axis=1, inplace=True)# yukarıdaki işlemin aynısını burada da yaptım



tox_0_concate_sic=pd.concat([tank6_sic,tox_0],axis=1)
tox_0_concate_sic.dropna(inplace=True)

tox_0_concate_sic.drop('TOX_ALARM', axis=1, inplace=True) # Burada sicaklik kaydım 14000 idi alarm verilerim ise 49 bin idi boyutlar uyuşmadığı için plot çizilmedi 
# bu yüzden sicaklikta tox alarımda 0'a denk gelen satırları aldım ve boyutu 49 bine indirdim artık çizdirebilirim


tox_0_concate_bas=pd.concat([tank6_bas,tox_0],axis=1)
tox_0_concate_bas.dropna(inplace=True)
tox_0_concate_bas.drop('TOX_ALARM', axis=1, inplace=True)# yukarıdaki işlemin aynısını burada da yaptım












plt.figure(figsize=(16, 16))
plt.scatter(tank1_sic,ayir2_, c='blue', label='alarm-tank1 sıcaklığı')
plt.xlabel('Sıcaklık')
plt.ylabel('alarm değeri')
plt.title('alarm-tank1 sıcaklığı')
plt.legend()
plt.grid()
plt.show()


min_sicaklik = min(tank1_sic['HYRAULIC_TANK_1_YAG_SICAKLIGI'])
max_sicaklik = max(tank1_sic['HYRAULIC_TANK_1_YAG_SICAKLIGI'])

print(min_sicaklik)
print(max_sicaklik)






plt.figure(figsize=(16, 16))
plt.scatter(tank3_sic,ayir2_, label='alarm-tank3 sıcaklığı')
plt.xlabel('Sıcaklık')
plt.ylabel('alarm değeri')
plt.title('alarm-tank3 sıcaklığı')
plt.legend()
plt.grid()
plt.show()



corr_matrix = all_concate.corr() 
plt.figure(figsize=(20,15))
sns.heatmap(corr_matrix, annot=True)
plt.show()

# hidrolik 6 ile tox alarm arasındaki kolerasyon matrisi..
hidrolik_6=data.iloc[:,[12,13,34]]

corr_matrix_tox=hidrolik_6.corr()
plt.figure(figsize=(20,15))
sns.heatmap(corr_matrix_tox,annot=True)
plt.show()












plt.scatter(tox_0_concate_sic,tox_0_concate_bas, color='green', label='Alarm Çalmıyor', marker='o',alpha=0.4)


plt.scatter(tox_1_concate_sic,tox_1_concate_bas, color='red', label='Alarm Çalıyor', marker='o',alpha=0.4)

"""

"""

# Eksen etiketlerini ve başlığı ekledik
plt.xlabel('Sicaklik (°C)')
plt.ylabel('Basinc (kPa)')
plt.title('Alarm Durumu ve Basinc-Sicaklik İlişkisi')

# Lejantı ekledik
plt.legend()

# Görseli görüntüle
plt.show()
















