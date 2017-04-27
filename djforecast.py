from pandas import read_csv, DataFrame, Series
from sklearn import cross_validation, svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import pylab as pl


#US30- Index of Dow Jones. 1- growth, 0- falling

#PriceOIL- Oil price

#PriceAXP- American Express stock price

#PriceBAC- Bank Of America stock price

#PriceMCD- McDonalds stock price

#PriceHPQ- Hewlett Packard stock price

#PriceINTC- Intel stock price

#PricePG- Prockter And Gamble stock price

#PriceIBM- International Business Machines stock price

#PoliticSTBL- Political stabilty in country(us) p- positive, n- negative

#ConsBR- Consumer Behavior. Shows buying activity of americans. a- active, na- non active

#All prices are in US Dollars


data = read_csv('dowjonesforecast/data/train.csv')
data.pivot_table('Day', 'PriceOIL', 'US30', 'count').plot(kind='bar', stacked=True)

fig, axes = plt.subplots(ncols=2)
data.pivot_table('Day', ['PriceHPQ'], 'US30', 'count').plot(ax=axes[0], title='PriceHPQ')
data.pivot_table('Day', ['PriceINTC'], 'US30', 'count').plot(ax=axes[1], title='PriceINTC')

data.Day[data.PolSTBL.notnull()].count()
data.Day[data.PriceMCD.notnull()].count()
data.PriceMCD = data.PriceMCD.median()
data[data.ConsBR.isnull()]

MaxPassEmbarked = data.groupby('ConsBR').count()['Day']
data.ConsBR[data.ConsBR.isnull()] = MaxPassEmbarked[MaxPassEmbarked == MaxPassEmbarked.max()].index[0]

data.Day[data.Fare.isnull()]
data = data.drop(['Day','PriceAXP','PricePG','PolSTBL'],axis=1)


label = LabelEncoder()
dicts = {}

label.fit(data.PriceBAC.drop_duplicates()) 
dicts['PriceBAC'] = list(label.classes_)
data.PriceBAC = label.transform(data.PriceBAC) 

label.fit(data.ConsBR.drop_duplicates())
dicts['ConsBR'] = list(label.classes_)
data.ConsBR = label.transform(data.ConsBR)

test = read_csv('dowjonesforecast/data/test.csv')
test.PriceMCD[test.PriceMCD.isnull()] = test.PriceMCD.mean()
test.Fare[test.Fare.isnull()] = test.Fare.median() 
MaxPassConsBR = test.groupby('ConsBR').count()['Day']
test.ConsBR[test.ConsBR.isnull()] = MaxPassEmbarked[MaxPassEmbarked == MaxPassEmbarked.max()].index[0]
result = DataFrame(test.Day)
test = test.drop(['PriceAXP','PricePG','PolSTBL','Day'],axis=1)

label.fit(dicts['PriceBAC'])
test.PriceBAC = label.transform(test.PriceBAC)

label.fit(dicts['ConsBR'])
test.ConsBR = label.transform(test.ConsBR)


target = data.US30
train = data.drop(['US30'], axis=1) 
kfold = 5 
itog_val = {} 


ROCtrainTRN, ROCtestTRN, ROCtrainTRG, ROCtestTRG = cross_validation.train_test_split(train, target, test_size=0.25) 

model_rfc = RandomForestClassifier(n_estimators = 70) 
model_knc = KNeighborsClassifier(n_neighbors = 18) 
model_lr = LogisticRegression(penalty='l1', tol=0.01) 
model_svc = svm.SVC()


scores = cross_validation.cross_val_score(model_rfc, train, target, cv = kfold)
itog_val['RandomForestClassifier'] = scores.mean()
scores = cross_validation.cross_val_score(model_knc, train, target, cv = kfold)
itog_val['KNeighborsClassifier'] = scores.mean()
scores = cross_validation.cross_val_score(model_lr, train, target, cv = kfold)
itog_val['LogisticRegression'] = scores.mean()
scores = cross_validation.cross_val_score(model_svc, train, target, cv = kfold)
itog_val['SVC'] = scores.mean()


DataFrame.from_dict(data = itog_val, orient='index').plot(kind='bar', legend=False)

pl.clf()
plt.figure(figsize=(8,6))

model_svc.probability = True
probas = model_svc.fit(ROCtrainTRN, ROCtrainTRG).predict_proba(ROCtestTRN)
fpr, tpr, thresholds = roc_curve(ROCtestTRG, probas[:, 1])
roc_auc  = auc(fpr, tpr)
pl.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('SVC', roc_auc))

probas = model_rfc.fit(ROCtrainTRN, ROCtrainTRG).predict_proba(ROCtestTRN)
fpr, tpr, thresholds = roc_curve(ROCtestTRG, probas[:, 1])
roc_auc  = auc(fpr, tpr)
pl.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('RandonForest',roc_auc))

probas = model_knc.fit(ROCtrainTRN, ROCtrainTRG).predict_proba(ROCtestTRN)
fpr, tpr, thresholds = roc_curve(ROCtestTRG, probas[:, 1])
roc_auc  = auc(fpr, tpr)
pl.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('KNeighborsClassifier',roc_auc))

probas = model_lr.fit(ROCtrainTRN, ROCtrainTRG).predict_proba(ROCtestTRN)
fpr, tpr, thresholds = roc_curve(ROCtestTRG, probas[:, 1])
roc_auc  = auc(fpr, tpr)
pl.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('LogisticRegression',roc_auc))
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.legend(loc=0, fontsize='small')
pl.show()

model_rfc.fit(train, target)
result.insert(1,'US30', model_rfc.predict(test))
result.to_csv('dowjonesforecast/data/train.csv', index=False)