# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 20:24:56 2021

@author: AM4
"""
import pandas as pd
import numpy as np
from neural import MLP

df = pd.read_csv('data.csv')

df = df.iloc[np.random.permutation(len(df))]
y = df.iloc[0:100, 4].values
yf = np.eye(100,3)
for i in range(0,100):
    if y[i] == "Iris-setosa":
        yf[i] = np.array([1,0,0])
    if y[i] == "Iris-versicolor":
        yf[i] = np.array([0,1,0])
    if y[i] == "Iris-virginica":
        yf[i] = np.array([0,0,1])
X = df.iloc[0:100, 0:4].values

inputSize = X.shape[1] # РєРѕР»РёС‡РµСЃС‚РІРѕ РІС…РѕРґРЅС‹С… СЃРёРіРЅР°Р»РѕРІ СЂР°РІРЅРѕ РєРѕР»РёС‡РµСЃС‚РІСѓ РїСЂРёР·РЅР°РєРѕРІ Р·Р°РґР°С‡Рё 
hiddenSizes = 10 # Р·Р°РґР°РµРј С‡РёСЃР»Рѕ РЅРµР№СЂРѕРЅРѕРІ СЃРєСЂС‹С‚РѕРіРѕ (Рђ) СЃР»РѕСЏ 
outputSize = yf.shape[1] # РєРѕР»РёС‡РµСЃС‚РІРѕ РІС‹С…РѕРґРЅС‹С… СЃРёРіРЅР°Р»РѕРІ СЂР°РІРЅРѕ РєРѕР»РёС‡РµСЃС‚РІСѓ РєР»Р°СЃСЃРѕРІ Р·Р°РґР°С‡Рё

iterations = 50
learning_rate = 0.1

net = MLP(inputSize, outputSize, learning_rate, hiddenSizes)

# РѕР±СѓС‡Р°РµРј СЃРµС‚СЊ (С„Р°РєС‚РёС‡РµСЃРєРё СЃРµС‚СЊ СЌС‚Рѕ РІРµРєС‚РѕСЂ РІРµСЃРѕРІ weights)
for i in range(iterations):
    net.train(X, yf)

    if i % 10 == 0:
        print("РќР° РёС‚РµСЂР°С†РёРё: " + str(i) + ' || ' + "РЎСЂРµРґРЅСЏСЏ РѕС€РёР±РєР°: " + str(np.mean(np.square(yf - net.predict(X)))))

# СЃС‡РёС‚Р°РµРј РѕС€РёР±РєСѓ РЅР° РѕР±СѓС‡Р°СЋС‰РµР№ РІС‹Р±РѕСЂРєРµ
pr = net.predict(X)
print(sum(abs(yf-(pr>0.5))))