# Quant

## 작성중 .. [2019-04-01]
## 5월부터 오픈 예정 



<p align="center">
    <a target="_blank" href="https://www.creontrade.com/">
        <img src="https://raw.githubusercontent.com/Park-Ju-hyeong/Quant/master/images/%ED%81%AC%EB%A0%88%EC%98%A8.png" />
    </a>
</p>

대신증권 크레온 플러스를 이용해서 기업별 주식 데이터를 가져오고 매매를 진행합니다.  
윈도우 예약시스템을 이용하려 했으나, 가끔 실행이 안될때가 있어 파이썬 스크립트로 작성하게 됨.  


## 데이터

2019.03.17. 기준으로 `시가총액` 1조원 이상인 기업들의 주가데이터를 `Quant/data/*code*` 에 저장했습니다.  
221개 종목 (`OTM ETN` 포함)

## 데이터 구조

<table>
    <thead>
        <tr>
            <th> 종목 코드 </th>
            <th> 파일 명 </th>
            <th> 내용 </th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=5> A005930 </td>
            <td rowspan=2>DAY_A005930.txt</td>
            <td>전년도 일단위 데이터</td>
        </tr>
        <tr>
            <td>변수 : 날짜, 시가, 고가, 저가, 종가, 거래량, <br>거래대금, 상장주식수, 시가총액, 외국인현보유수량, 기관순매수</td>
        </tr>
        <tr>
            <td rowspan=3>MIN_YYYY_A005930.txt</td>
            <td>연도별 분단위 데이터</td>
        </tr>
        <tr>
            <td>변수 : 날짜, 시간, 시가, 고가, 저가, 종가, 거래량</td>
        </tr>
        <tr>
            <td><U>대신증권에서 2017년 이전 분단위 데이터를 제공하지 않음.</U></td>
        </tr>
    </tbody>
</table>

## 데이터 읽기

```python
import pandas as pd
data = pd.read_table("/notebooks/JuHyeong/JuIng/Stock/data/A005930/DAY_A005930.txt", delimiter=" ")
data.tail()
```  


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>날짜</th>
      <th>시가</th>
      <th>고가</th>
      <th>저가</th>
      <th>종가</th>
      <th>거래량</th>
      <th>거래대금</th>
      <th>상장주식수</th>
      <th>시가총액</th>
      <th>외국인현보유수량</th>
      <th>기관순매수</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10558</th>
      <td>20190325</td>
      <td>45300</td>
      <td>45650</td>
      <td>44800</td>
      <td>45500</td>
      <td>8699728</td>
      <td>394433000000</td>
      <td>5969782000</td>
      <td>271625081000000</td>
      <td>3391523</td>
      <td>-1029563</td>
    </tr>
    <tr>
      <th>10559</th>
      <td>20190326</td>
      <td>45500</td>
      <td>45700</td>
      <td>44900</td>
      <td>45250</td>
      <td>9729811</td>
      <td>440020000000</td>
      <td>5969782000</td>
      <td>270132635000000</td>
      <td>3393714</td>
      <td>-1332399</td>
    </tr>
    <tr>
      <th>10560</th>
      <td>20190327</td>
      <td>44750</td>
      <td>45600</td>
      <td>44250</td>
      <td>45350</td>
      <td>9568081</td>
      <td>430843000000</td>
      <td>5969782000</td>
      <td>270729613000000</td>
      <td>3391456</td>
      <td>1031230</td>
    </tr>
    <tr>
      <th>10561</th>
      <td>20190328</td>
      <td>44950</td>
      <td>45200</td>
      <td>44300</td>
      <td>44850</td>
      <td>6821306</td>
      <td>306038000000</td>
      <td>5969782000</td>
      <td>267744722000000</td>
      <td>3391293</td>
      <td>-864538</td>
    </tr>
    <tr>
      <th>10562</th>
      <td>20190329</td>
      <td>44500</td>
      <td>44900</td>
      <td>44200</td>
      <td>44650</td>
      <td>11491713</td>
      <td>511624000000</td>
      <td>5969782000</td>
      <td>266550766000000</td>
      <td>3392055</td>
      <td>325024</td>
    </tr>
  </tbody>
</table>




## 코드 예시
 

| 기준 | 주피터 노트북 | 내용 |
|:----:|:--------:|:---------:|
| 초기 세팅 | <a target="_blank" href="https://github.com/Park-Ju-hyeong/Quant/blob/master/ipynb/%EC%9D%BC%EB%B3%84%EB%8D%B0%EC%9D%B4%ED%84%B0(%EC%B4%88%EA%B8%B0%EC%84%B8%ED%8C%85).ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> Jupyter </a> | 1000.01.01. ~ 2019.12.31.(현재) 까지 주가 일별 데이터를 수집합니다. |
| 초기 세팅 | <a target="_blank" href="https://github.com/Park-Ju-hyeong/Quant/blob/master/ipynb/%EB%B6%84%EB%8B%A8%EC%9C%84%EB%8D%B0%EC%9D%B4%ED%84%B0(%EC%B4%88%EA%B8%B0%EC%84%B8%ED%8C%85).ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> Jupyter </a> | 1000.01.01. ~ 2019.12.31.(현재) 까지 주가 분단위 데이터를 수집합니다. |  

<!-- | 매일 실행 | <a target="_blank" href="https://github.com/Park-Ju-hyeong/Quant/blob/master/ipynb/%EC%9D%BC%EB%B3%84%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%88%98%EC%A7%91(%EB%A7%A4%EC%9D%BC%EC%8B%A4%ED%96%89).ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> Jupyter </a> | 매일 01:00 ~ 05:00 사이에 실행 시켜 전날 일별 주가 데이터를 가져옵니다.|
| 매일 실행 | <a target="_blank" href="https://github.com/Park-Ju-hyeong/Quant/blob/master/ipynb/%EB%B6%84%EB%8B%A8%EC%9C%84%EB%8D%B0%EC%9D%B4%ED%84%B0(%EB%A7%A4%EC%9D%BC%EC%8B%A4%ED%96%89).ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> Jupyter </a> | 매일 01:00 ~ 05:00 사이에 실행 시켜 전날 주가 분단위 데이터를 가져옵니다.| -->


## 함수 

| 함수 | 내용 | 실행 시간  | 비고 | 
|:--------:|:----:|:----:|:----:|
| `getDayStock` | 전날 일별 주가를 수집합니다. | 02:00 | - |
| `getMinStock` | 전날 분별 주가를 수집합니다. | 03:00 | - |
| `-` | - | - | - |
| `stockBuy` | 매수 요청을 날립니다. | - | `price=-1` 이면 시장가 매매 |
| `stockSell` | 매도 요청을 날립니다. | - | `price=-1` 이면 시장가 매매 |
| `-` | - | - | - |
| `checkLogin` | 크레온 플러스에 연결여부를 판단합니다. | - | 접속이 안돼있다면, 로그인합니다. |
 `creonPlusDisconnect` | 크레온 플러스에 연결을 끊습니다. | 05:00 | 동시에 프로그램도 종료됩니다. |
 | `-` | - | - | - |
| `pushGithub` | `Github/Quant` 에 push 합니다. | - | auto commit |
| `-` | - | - | - |

## 프로세스

```
while 1:

    ################################################################################
    #     00:00 ~ 05:00 [데이터 수집]
    ################################################################################
    
    if 0 < creon.getWeek() < 6:

        if creon.checkTime("0200"):
            creon.checkLogin()
            creon.getDayStock()
            creon.pushGithub()

        if creon.checkTime("0300"):
            creon.checkLogin()
            creon.getMinStock()
            creon.pushGithub()

    ################################################################################
    #     05:45 ~ 06:00 [서버 점검]
    ################################################################################

    if creon.checkTime("0500"):
        creon.checkLogin()
        creon.creonPlusDisconnect()

    if creon.checkTime("0700"):
        creon.checkLogin()

    ################################################################################
    #     09:00 ~ 15:30 [거래]
    ################################################################################    

    if creon.getWeek() < 5:

        if creon.checkTime("0850"):
            creon.checkLogin()

            for code, price, volume in [["A005930", -1, 1]]:

                creon.stockBuy(code, price, volume)


        if creon.checkTime("1520"):
            creon.checkLogin()

            for code, price, volume in [["A005930", -1, 1]]:

                creon.stockSell(code, price, volume)

    ################################################################################
    #     16:00 ~ 24:00 [휴식]
    ################################################################################    

    if creon.checkTime("1600"):
        time.sleep(60*60*8)

    time.sleep(100)
```