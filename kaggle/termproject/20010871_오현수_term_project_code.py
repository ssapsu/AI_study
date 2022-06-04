# !kaggle competitions download -c titanic -p "./dataset/"
# !unzip ./dataset/titanic.zip -d ./dataset && rm ./dataset/titanic.zip
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
train_filepath = "./dataset/train.csv"
test_filepath = "./dataset/test.csv"
train = pd.read_csv(train_filepath)
test = pd.read_csv(test_filepath)
train.info()
결측치 보정
train.describe()
train.describe(include=['O'])
<h1>Pclass와 생존</h1>
cor_Pclass_Survived = train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print(cor_Pclass_Survived)
# plt.figure(figsize=(6,6))
plt.title("correlation between Pclass and Survived")
sns.barplot(x=cor_Pclass_Survived["Pclass"], y=cor_Pclass_Survived["Survived"]);
#groupby에 as_index를 False로 하면 Pclass를 Index로 사용하지 않음
#ascending : 오름차순
cor_Sex_Survived = train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by="Survived", ascending=False)
plt.title("correlation between sex and survived")
sns.barplot(x=cor_Sex_Survived["Sex"], y=cor_Sex_Survived["Survived"]);
sns.swarmplot(x=train["Survived"], y=train["Fare"]);
cor_SibSp_Survived = train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values("Survived", ascending=False)
plt.title("correlation between SibSp and Survived")
sns.barplot(x = cor_SibSp_Survived["SibSp"], y=cor_SibSp_Survived["Survived"]);
cor_Parch_Survived = train[["Parch","Survived"]].groupby(["Parch"], as_index=False).mean().sort_values("Survived", ascending=False)
plt.title("correlation between Parch and Survived")
sns.barplot(x=cor_Parch_Survived["Parch"], y=cor_Parch_Survived["Survived"]);
g = sns.FacetGrid(train, col="Survived")
g.map(plt.hist, 'Fare', bins=20);
g = sns.FacetGrid(train, col="Survived")
g.map(plt.hist, 'Age', bins=20);
train["Title"] = train["Name"].str.extract('([A-Za-z]+)\.')
    
train.head(5)
g = sns.FacetGrid(train, col="Sex", row="Survived")
g.map(plt.hist, 'Age', bins=20);
grid = sns.FacetGrid(train, col="Survived", row="Pclass",hue="Pclass",height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5,bins=20)
grid.add_legend();
grid = sns.FacetGrid(train, row="Pclass", col="Sex", height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();
df = train.corr()
df
# heatmap by seaborn
ax = sns.heatmap(df, annot=True)
plt.title('Heatmap of Flight by seaborn', fontsize=20)
plt.show()
print(train.columns)
print(train.shape)
train.isnull().sum()
Cabin 칼럼의 경우 결측치의 비중이 너무 높으므로 남은 데이터로 추정하면 노이즈만 증가할 것으로 보이므로 제거
data = train,test
print(type(data))
for element in data:
    element.drop(['Cabin'], axis=1, inplace=True)
for element in data:
    element[element["Embarked"].isnull()].fillna("S")
for element in data:
    element['Title'] = element.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
for element in data:
    print(pd.crosstab(element["Title"], element["Sex"]))
for element in data:
    element['Title'] = element['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
        'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    element['Title'] = element['Title'].replace('Mlle', 'Miss')
    element['Title'] = element['Title'].replace('Ms', 'Miss')
    element['Title'] = element['Title'].replace('Mme', 'Mrs')
for element in data:
    element.groupby('Title')["Age"].mean()
for element in data:
    element.loc[(element.Age.isnull()) & (element.Title=='Master'), 'Age'] = 4.5742
    element.loc[(element.Age.isnull()) & (element.Title=='Miss'), 'Age'] = 21.846
    element.loc[(element.Age.isnull()) & (element.Title=='Mr'), 'Age'] = 32.368
    element.loc[(element.Age.isnull()) & (element.Title=='Mrs'), 'Age'] = 35.789
    element.loc[(element.Age.isnull()) & (element.Title=='Rare'), 'Age'] = 45.545
a,b = data
a = pd.get_dummies(a, columns=["Title", "Embarked"]).copy()
b = pd.get_dummies(b, columns=["Title", "Embarked"]).copy()
data=a,b
for element in data:
    element['Age'] = element['Age'].astype(float)/100
    print(element)
# for element in data:
#     element['AgeBand'] = pd.cut(element['Age'], 5)
# for element in data:
#     element.loc[ element['Age'] <= 16, 'Age'] = 0
#     element.loc[(element['Age'] > 16) & (element['Age'] <= 32), 'Age'] = 1
#     element.loc[(element['Age'] > 32) & (element['Age'] <= 48), 'Age'] = 2
#     element.loc[(element['Age'] > 48) & (element['Age'] <= 64), 'Age'] = 3
#     element.loc[ element['Age'] > 64, 'Age']
#     element = element.drop(['AgeBand'], axis=1,inplace=True)
for element in data:
    element['FamilySize'] = element['SibSp'] + element['Parch'] + 1
# train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for element in data:
    element['IsAlone'] = 0
    element.loc[element['FamilySize'] == 1, 'IsAlone'] = 1
# train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
for element in data:
    element['Sex'] = element["Sex"].astype('category')
    print(element.Sex.cat.categories)
    print(element.Sex.cat.codes.head(5))

    element["Sex"] = element.Sex.cat.codes
for element in data:
    element['Fare'].fillna(element['Fare'].dropna().median(), inplace=True)
    print(element.head())

for element in data:
    element['FareBand'] = pd.qcut(element['Fare'], 4)
    # train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
for element in data:
    element.loc[ element['Fare'] <= 7.91, 'Fare'] = 0
    element.loc[(element['Fare'] > 7.91) & (element['Fare'] <= 14.454), 'Fare'] = 1
    element.loc[(element['Fare'] > 14.454) & (element['Fare'] <= 31), 'Fare']   = 2
    element.loc[ element['Fare'] > 31, 'Fare'] = 3
    element['Fare'] = element['Fare'].astype(int)

    element = element.drop(['FareBand'], axis=1,inplace=True)
a,b = data
a.info()
data[1].head()
for element in data:
    element = element.drop(['PassengerId', 'Name', "SibSp", "Parch", "Ticket"],axis = 1, inplace=True)
df = data[0].corr()
plt.figure(figsize=(15,15))
sns.heatmap(df, annot=True)
print(df)
#Pclass, IsAlone, Rare는 개선이 필요, Age도 개선 필요, Pclass, Sex는 왜 이상하냐 ㅅㅂ
for element in data:
    element = element.drop(['FamilySize'],axis = 1, inplace=True)
train, test = data

X_train = train.drop("Survived",axis = 1)
Y_train = train["Survived"]
X_train.shape, Y_train.shape
X_train
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train,Y_train, test_size=0.3, shuffle=True)
model = Sequential()
model.add(Dense(128, input_dim=13, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics="accuracy")
model.summary()
#모델 최적화 진행
MODEL_DIR = './model/'
if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)

modelpath="./model/{epoch:02d}-{val_loss:.4f}.hdf5"
checkpointer = ModelCheckpoint(filepath=modelpath, monitor="val_loss", verbose=1, save_best_only=True)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)


history = model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid),epochs=500, batch_size=100, callbacks=[early_stopping_callback, checkpointer])

y_vloss = history.history['val_loss']

y_loss = history.history['loss']

# 그래프로 표현
x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c="red", label='Testset_loss')
plt.plot(x_len, y_loss, marker='.', c="blue", label='Trainset_loss')

# 그래프에 그리드를 주고 레이블을 표시
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
print("\n Accuracy: %.4f" % (model.evaluate(X_valid, Y_valid)[1]))
pred = model.predict(test)
pred = pd.DataFrame(pred, columns=['Survived'])
print(pred.head())
series = pd.Series([i for i in range(892, 1310)])
pred.loc[:,'PassengerID'] = series
pred = pred[["PassengerID", "Survived"]]
pred["Survived"] = pred[["Survived"]].round().astype(int)
pred.to_csv('./predict.csv', index = False)