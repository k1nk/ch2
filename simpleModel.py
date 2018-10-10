# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 14:53:54 2018

@author: nakatanikenichi
"""

import numpy as np
import chainer.functions as F
import chainer.links as L
from chainer import Variable, optimizers

# グラフ表示を行うため、グラフ表示用のモジュールmatplotlibをpltとして読み込みます。
import matplotlib.pyplot as plt

def plotmodel(model):   
    x_for_plot = Variable(np.array([[1],[2],[7]],dtype=np.float32))   
    y_for_plot = model(x_for_plot)    
    #plt.plot(x_for_plot.data,y_for_plot.data,"r-")
    plt.plot(x_for_plot.data,y_for_plot.data,marker='o',ls='-')
    plt.xlabel("x")
    plt.ylabel("y")
    
#モデルを定義します。
model = L.Linear(1,1)

#パラメータの最適化の方法を選択し、モデルと関連付けます。
optimizer = optimizers.SGD()
optimizer.setup(model)

#入力するデータ[1],[2],[7]を作成します。
x = Variable(np.array([[1],[2],[7]],dtype=np.float32))

#答えとなるデータ[2],[4],[14]を作成します。
t = Variable(np.array([[2],[4],[14]], dtype=np.float32))

#学習の回数を設定します。
times = 50

for i in range(0,times):
    #モデルの勾配データを初期化します。
    model.cleargrads()
    #optimizer.zero_grads()

    #予測します。
    y = model(x)

    #モデルの出力を表示します。
    #print(y.data)

    #予測と答えとの誤差を計算します。
    loss = F.mean_squared_error(y,t)

    #誤差逆伝搬を行い、勾配を計算します。
    #print "誤差逆伝搬前の勾配　model.W.grad :", model.W.grad
    loss.backward()
    #print "誤差逆伝搬後の勾配　model.W.grad :", model.W.grad
    
    #パラメータの更新を行います。
    #print "更新前のウエイト　model.W.data :", model.W.data
    optimizer.update()
    #print "更新後のウエイト　model.W.data :", model.W.data
    
    #モデルをグラフ表示します。
    plotmodel(model)

plt.show()