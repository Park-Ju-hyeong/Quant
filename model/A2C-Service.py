#!/usr/bin/env python
# coding: utf-8

import os
os.environ["CUDA_VISIBLE_DEVICES"]="9"

import gym
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko


from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D
from tqdm import tqdm
from glob import glob


import codecs
import time
import logging, logging.config
from datetime import datetime, timedelta

class Help(object):
    """유틸함수"""
    
    def getNow(self):
        """오늘날짜"""
        return datetime.now().strftime("%Y%m%d")
    
    def getNowDiff(self, N=1):
        """N일 이전 날짜"""
        return (datetime.now() - timedelta(N)).strftime("%Y%m%d")
    
    def getTime(self):
        """현재시간"""
        return datetime.now().strftime("%H%M")
    
    def getWeek(self):
        return datetime.today().weekday()
    
    def checkTime(self, timer="0100"):
        """실행 시간 체크 함수
        2분 단위로 시간을 체크한다. 
        함수 실행시간이 2분 이내라면, time.sleep(60*2) 를 통해서 중복 실행되는걸 막아야 한다.
        """
#         now = int(self.getTime()) // 2
#         run = int(timer) // 2
#         return now == run
        while int(int(self.getTime()) if int(self.getTime()) > 700 else int(self.getTime()) + 2400) < int(timer):
            time.sleep(1)
    
    def getTimer(self):
        
        timer = []
        
        for mm in range(9, 16):
    
            for ss in range(60):

                mmss = "{}{}".format(str(mm).zfill(2), str(ss).zfill(2))

                if int(mmss) >= 1520:

                    break
                    
                timer.append(mmss)
                
        return timer
                    
    def pushGithub(self):
        subprocess.call("/Users/juhy9/Documents/GitHub/System/Github/src/bat/GitAutoPush.bat")
        time.sleep(60*2)
        self.logger.info("</pushGithub>")

        
    def get_processes_running(self):
        """대신증권 PID 가져오기"""
        tasks = subprocess.check_output(['tasklist']).decode('cp949', 'ignore').split("\r\n")
        p = []
        for task in tasks:
            m = re.match("(.+?) +(\d+) (.+?) +(\d+) +(\d+.* K).*",task)
            if m is not None:
                image = str(m.group(1))
                pid = str(m.group(2))

                if "DibServer.exe" == image:

                    return pid
        return None


utils = Help()


columns = [
 'A Ratio_20',
 'B Ratio_20',
 'Aroon_20 Up',
 'Down',
 'Aroon Osillator_20',
 'BB-RSI_종가_10',
 'BB-RSI_20_2.00 상한',
 'BB-RSI 하한',
 'Chande Momentum Oscillator_10',
 "Chaikin's Volatility_14,14",
 'CompuTrac Volatility_10',
 '+DI_14',
 '-DI_14',
 '+DI(simple)_14',
 '-DI(simple)_14',
 'Energy+_14',
 'Energy-_14',
 'High Low Oscillator_3',
 'Inertia_20,14,10',
 'Linear Trend Oscillator_10,20',
 'Open Difference_15',
 'Random Walk Index_15,3',
 'RCI_5',
 'RCI_9',
 'RCI_13',
 'RCI_18',
 'Relative Volatility Index_단순,14,10',
 'Reverse 단기_12',
 ' 장기_24',
 'RSI_종가,14',
 'RSI(simple)_종가,14',
 'Sigma_종가,20',
 'SMI_5,3,3',
 'Fast %K_5',
 'Fast %D_3',
 'Slow %K_5,3',
 'Slow %D_3',
 'Slow(Simple) %K_5,3',
 'Slow(Simple) %D_3',
 'StochOsc_5,3',
 'TRIX기울기_종가,14',
 'TSF Oscillator_종가,14,28',
 'VHF_14',
 'CCI_14',
]



scaler = pd.read_csv("/notebooks/jpark/Documents/GitHub/Quant/model/scaler_v1.csv")

scaler.index = scaler["Unnamed: 0"]

del scaler["Unnamed: 0"]

class KRX():
    
    def __init__(self):
        self.config()
        self.dataload()
        
    def config(self):
        """configuration"""
        
        self.action_space = 2
        self.observation_space = (1 + len(columns), )

    def dataload(self):
        """github에 있는 주가데이터 가져오기"""
        
        self.KRX_ETF_2020_LIST = pd.read_csv("/notebooks/jpark/Documents/GitHub/Quant/data/KRX_ETF_2020_LIST.csv")
        self.DF = pd.DataFrame([])
        self.OBS = pd.DataFrame([])
        
        for idx, (name, code, _, _) in tqdm(self.KRX_ETF_2020_LIST.iterrows(), total=len(self.KRX_ETF_2020_LIST)):
            
            try:
                filepath = "/notebooks/jpark/Documents/GitHub/Quant/data/{}/Index_{}.txt".format(code, code)
                STOCK_DATA = pd.read_table(filepath, delimiter=" ")
                STOCK_DATA["종목"] = idx
                STOCK_DATA = STOCK_DATA.loc[(STOCK_DATA.날짜 == STOCK_DATA.날짜.max())]
                # STOCK_DATA = STOCK_DATA.loc[(STOCK_DATA.날짜 > 20191200 )]
                STOCK_DATA[columns] = STOCK_DATA[columns].astype(float)    
                STOCK_DATA = STOCK_DATA.loc[STOCK_DATA[columns].apply(lambda x : np.sum(x) < 1e+100, 1)]

                if len(STOCK_DATA) > 0:
                    STOCK_OHLC = STOCK_DATA[["날짜", "종목", "시가", "고가", "저가", "종가"]]
                    STOCK_INDEX = STOCK_DATA[["날짜", "종목"] + columns]

                    if len(STOCK_OHLC) > 0:
                        self.DF = pd.concat([self.DF, STOCK_OHLC])
                        self.OBS = pd.concat([self.OBS, STOCK_INDEX])
                        
            except:
                print(name, code)
                # 한화시스템 A272210
                # 롯데리츠 A330590

        self.DF = self.DF.sort_values(["날짜", "종목"])
        self.OBS = self.OBS.sort_values(["날짜", "종목"])
        
        self.DF = self.DF.reset_index(drop=True)
        self.OBS = self.OBS.reset_index(drop=True)
        
        self.day = self.DF.날짜.unique()
        self.item = self.DF.종목.unique()
        self.meanP = scaler.meanP
        self.stdP = scaler.stdP
        
        self.NEXTOBS = pd.concat([
            self.OBS.iloc[:, 1:2],
            (self.OBS.iloc[:, 2:] - self.meanP) / self.stdP
        ], axis=1)
        
        
        self.DFLIST = self.DF.values.tolist()
        self.NEXTLIST = np.array(self.NEXTOBS.values.tolist())
            
    def mywallet(self):
        """계좌관리"""
        
        self.account = {}
        
        # 천만원부터 시작하자 
        self.account["account"] = 10000000
        # 주식평가가치 
        self.account["eval_stock"] = 0
        
        # 일자별 잔고 
        self.dayaccount = {}
        for day in self.day:
            self.dayaccount[str(day)] = 0
        
        # 종목별 잔고 
        for idx, (name, code, _, _) in self.KRX_ETF_2020_LIST.iterrows():
            self.account[str(idx)] = {}
            self.account[str(idx)]["name"] = name
            self.account[str(idx)]["code"] = code
            self.account[str(idx)]["nowP"] = 0
            self.account[str(idx)]["buyP"] = 0
            self.account[str(idx)]["amount"] = 0            
            self.account[str(idx)]["eval_stock"] = 0
            
            # 종목별 일자별 잔고
            self.account[str(idx)]["day"] = {}
            for day in self.day:
                self.account[str(idx)]["day"][str(day)] = 0
            
    def calcEval(self):
        """주식 평가가치 계산하기"""
        
        self.account["eval_stock"] = 0
        
        for idx, (name, code, _, _) in self.KRX_ETF_2020_LIST.iterrows():
            evalP = int(self.account[str(idx)]["nowP"] * self.account[str(idx)]["amount"])
            self.account["eval_stock"] += evalP
            
        return int(self.account["account"] + self.account["eval_stock"])
        
        
    def calcReward(self, item, ystrd, tod):
        """리워드 계산"""
        
        td_evalP = int(self.account[str(item)]["day"][str(tod)])
        ystd_evalP = int(self.account[str(item)]["day"][str(ystrd)])
        return  (td_evalP - ystd_evalP) / ystd_evalP * 100
            
        
    def trade_history(self):
        """거래기록"""
        
        self.history = []

    
    def reset(self):
        self.dones = False
        self.mywallet()
        self.trade_history()
        self.envidx = 0
        self.ystd = 0
        return self.NEXTLIST[self.envidx, :]
        
        
    def step(self, action, show=False):
        
        if self.dones:
            raise
        
        [date, item, openP, highP, lowP, closeP] = self.DFLIST[self.envidx]
        
        
        # 월초 시작잔고 뿌려주기 
        if self.ystd // 100 != date // 100:
            
            self.dayaccount[str(date)] = self.calcEval()
            print("[{}] 시작 잔고 : {}".format(date, format(self.dayaccount[str(date)], ",")))
            self.ystd = date
        
        # 리워드 계산을 위한 잔고 계산 
        startAccount = self.calcEval()
        
        ################################################################################
        # 장시작
        ################################################################################
        if action == 0:
            """pass"""
            self.account[str(item)]["nowP"] = closeP
            
        elif action == 1:
            """매수"""
            self.account[str(item)]["nowP"] = closeP
            self.account[str(item)]["buyP"] += openP
            self.account[str(item)]["amount"] += 1
            self.account["account"] -= openP
            
            
            msg = "매수 -- 시가 -- 날짜:{} -- 종목:{}/{} -- 산가격:{} -- 판가격:0 -- 종가:{}"
            out = msg.format(
                date, 
                self.account[str(item)]["name"], 
                self.account[str(item)]["code"],
                openP,
                closeP,
            )
            
            self.history.append(out)

            if show:
                print(out)
                
        else:
            raise print("ERROR")
        
        
        ################################################################################
        # 자동 매도 
        ################################################################################
        if self.account[str(item)]["amount"] > 0:
            
            avgP = int(self.account[str(item)]["buyP"] / self.account[str(item)]["amount"])
            if highP > int(avgP * (1 + 0.02)):

                self.account["account"] += int(self.account[str(item)]["buyP"] * (1 + 0.02))
                self.account[str(item)]["buyP"] = 0
                self.account[str(item)]["amount"] = 0

                msg = "매도 -- 익절 -- 날짜:{} -- 종목:{}/{} -- 산가격:{} -- 판가격:{} -- 종가:{}"
                out = msg.format(
                    date, 
                    self.account[str(item)]["name"], 
                    self.account[str(item)]["code"],
                    avgP,
                    int(avgP * (1 + 0.02)),
                    closeP,
                )
                
                self.history.append(out)

                if show:
                    print(out)

                
        ################################################################################
        # 로스컷 설정
        ################################################################################
        if self.account[str(item)]["amount"] > 0:
            
            avgP = int(self.account[str(item)]["buyP"] / self.account[str(item)]["amount"])
            if lowP < int(avgP * (1 - 0.03)):

                self.account["account"] += int(self.account[str(item)]["buyP"] * (1 - 0.03))
                self.account[str(item)]["buyP"] = 0
                self.account[str(item)]["amount"] = 0

                msg = "매도 -- 손절 -- 날짜:{} -- 종목:{}/{} -- 산가격:{} -- 판가격:{} -- 종가:{}"
                out = msg.format(
                    date,
                    self.account[str(item)]["name"], 
                    self.account[str(item)]["code"],
                    avgP,
                    int(avgP * (1 - 0.03)),
                    closeP,
                )
                
                self.history.append(out)
                
                if show:
                    print(out)

        ################################################################################
        # 수익률 정리
        ################################################################################
        
        # 현재가격(종가) * 주식 수 
        eval_stock = self.account[str(item)]["nowP"] * self.account[str(item)]["amount"]
        self.account[str(item)]["eval_stock"] = eval_stock
        self.account[str(item)]["day"][str(date)] = eval_stock
        
        # 매일 계산하다보면 제일 마지막 종목에서 계산되서
        # 궂이 매일 마지막 종목에서만 계산하라고 짤 필요가 없음 
        self.dayaccount[str(date)] = self.calcEval()
        
        ################################################################################
        # 일과 마무리 
        ################################################################################
        
        # 리워드
        # 현재 평잔 - 아침 평잔 / 단가 
        self.rewards = ( self.dayaccount[str(date)] - startAccount) / self.account[str(item)]["nowP"] * 100
        
        if show:
            print("[reward:{:>.2f}] {} ==> {}".format(self.rewards, format(startAccount, ","), format(self.dayaccount[str(date)], ",")))
        
        
        ################################################################################
        # 다음날 
        ################################################################################
        self.envidx = self.envidx + 1
        
        if self.envidx == len(self.NEXTLIST):
            self.dones = True
        else:
            self.next_obs = self.NEXTLIST[self.envidx, :]
        
        return self.next_obs, self.rewards, self.dones, None


# In[15]:


env = KRX()


# In[16]:


class ProbabilityDistribution(tf.keras.Model):
    def call(self, logits):
        # sample a random categorical action from given logits
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)

class Model(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__('mlp_policy')
        # no tf.get_variable(), just simple Keras API
        
        self.embedding = kl.Embedding(300, 5)
        self.flatten = kl.Flatten()
        
        self.rnn_layer_1 = kl.LSTM(1024, return_sequences=True)
        self.rnn_layer_2 = kl.LSTM(512, return_sequences=True)
        self.rnn_layer_3 = kl.LSTM(256)
        
        self.action_layer_1 = kl.Dense(128, activation='relu', name = "action")
        self.logits = kl.Dense(num_actions, name='policy_logits')
        
        self.value_layer_1 = kl.Dense(128, activation='relu', name = "value")
        self.value = kl.Dense(1, name='value')
        
        self.dist = ProbabilityDistribution()

    def call(self, inputs):
        # inputs is a numpy array, convert to Tensor
        x_item = tf.convert_to_tensor(inputs[:, :1])
        x_item = tf.cast(x_item, tf.int32)
        x_index = tf.convert_to_tensor(inputs[:, 1:])
        
        # separate hidden layers from the same input tensor
        self.embedding_item = self.embedding(x_item)
        self.embedding_flatten = self.flatten(self.embedding_item)
        
        # concatenate embeddinglayer and hiddne layer
        self.concat = kl.concatenate([self.embedding_flatten, x_index])
        self.timestep = tf.reshape(self.concat, (-1, 1, 49))
        
        
        self.lstm_1 = self.rnn_layer_1(self.timestep)
        self.lstm_2 = self.rnn_layer_2(self.lstm_1)
        self.lstm_3 = self.rnn_layer_3(self.lstm_2)
        
        # action
        self.action_out = self.action_layer_1(self.lstm_3)
        
        # values
        self.value_out = self.value_layer_1(self.lstm_3)
        
        return self.logits(self.action_out), self.value(self.value_out)

    def action_value(self, obs):
        # executes call() under the hood
        logits, value = self.predict(obs)
        action = self.dist.predict(logits)
        # a simpler option, will become clear later why we don't use it
        # action = tf.random.categorical(logits, 1)
        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)
        


# In[17]:


model = Model(num_actions=env.action_space)


# In[19]:


model.load_weights("/notebooks/jpark/Documents/GitHub/TensorFlow-2.x-Tutorials/17-A2C/ckpt/v1/QUANT_A2C_201912_19472140.ckpt")


# # Test

# In[20]:


df = []

next_obs = env.reset()

while not env.dones:
    
    item = int(next_obs[0])
    
    action, _ = model.action_value(next_obs[None, :])
    
    if action == 1:
    
        df.append([env.account[str(item)]["code"], env.account[str(item)]["name"]])
        
    next_obs, _, _, _ = env.step(action)


# In[21]:


buylist = pd.DataFrame(df)


# In[22]:


buylist


# In[23]:


buylist.to_csv("/notebooks/jpark/Documents/GitHub/Quant/buysell/{}.csv".format(utils.getNow()), index=None)
