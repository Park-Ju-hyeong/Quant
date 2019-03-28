# Quant

<p align="center">
    <img src="https://raw.githubusercontent.com/Park-Ju-hyeong/Quant/master/images/%ED%81%AC%EB%A0%88%EC%98%A8.png">
</p>

대신증권 크레온 플러스를 이용해서 기업별 주식 데이터를 가져오고 매매를 진행합니다.  
윈도우 예약시스템을 이용하려 했으나, 가끔 실행이 안될때가 있어 파이썬 스크립트로 작성하게 됨.

## 작성중 .. [2019-03-28]

## 실행 방법
 

| 기준 | 일단위 데이터 | 분단위 데이터| 내용 |
|:----:|:--------:|:---------:|:---------:|
| 초기 세팅 | <a target="_blank" href="https://github.com/Park-Ju-hyeong/Quant/blob/master/ipynb/%EC%9D%BC%EB%B3%84%EB%8D%B0%EC%9D%B4%ED%84%B0(%EC%B4%88%EA%B8%B0%EC%84%B8%ED%8C%85).ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> Jupyter </a> | <a target="_blank" href="https://github.com/Park-Ju-hyeong/Quant/blob/master/ipynb/%EB%B6%84%EB%8B%A8%EC%9C%84%EB%8D%B0%EC%9D%B4%ED%84%B0(%EC%B4%88%EA%B8%B0%EC%84%B8%ED%8C%85).ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> Jupyter </a> | 1000.01.01. ~ 2019.12.31.(현재) 까지 주가 데이터를 수집합니다. |
| 매일 실행 | <a target="_blank" href="https://github.com/Park-Ju-hyeong/Quant/blob/master/ipynb/%EC%9D%BC%EB%B3%84%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%88%98%EC%A7%91(%EB%A7%A4%EC%9D%BC%EC%8B%A4%ED%96%89).ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> Jupyter </a> | <a target="_blank" href="https://github.com/Park-Ju-hyeong/Quant/blob/master/ipynb/%EB%B6%84%EB%8B%A8%EC%9C%84%EB%8D%B0%EC%9D%B4%ED%84%B0(%EB%A7%A4%EC%9D%BC%EC%8B%A4%ED%96%89).ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> Jupyter </a> | 매일 01:00 ~ 05:00 사이에 실행 시켜 전날 주가 데이터를 가져옵니다.|

## 함수 



| 함수 | 내용 | 실행 시간  | 비고 | 
|:--------:|:----:|:----:|:----:|
| `getDayStock` | 전날 일별 주가 수집 | 02:00 | - |
| `getMinStock` | 전날 분별 주가 수집 | 03:00 | - |
| `checkLogin` | 크레온 플러스에 연결여부를 판단합니다. | - | 접속이 안돼있다면, 로그인합니다. |
| `creonPlusDisconnect` | 크레온 플러스에 연결을 끊습니다. | - | 동시에 프로그램도 종료됩니다. |
| `pushGithub` | `https://github.com/Park-Ju-hyeong/Quant` <br> 에 push 합니다. | - | auto commit |
| `-` | - | - | - |


## 알고리즘 

```
while 1:
    
    ################################################################################
    #     00:00 ~ 05:00 [데이터 수집]
    ################################################################################
    
    creon.checkLogin()

    if creon.CheckTime("0200"):
        creon.getDayStock()

    if creon.CheckTime("0300"):
        creon.getMinStock()
    
    creon.pushGithub()
    
    ################################################################################
    #     05:45 ~ 06:00 [서버 점검]
    ################################################################################
    
    creon.checkLogin()
    
    if creon.CheckTime("0500"):
        creon.creonPlusDisconnect()
        
    if creon.CheckTime("0700"):
        creon.checkLogin()
        
    ################################################################################
    #     09:00 ~ 15:30 [거래]
    ################################################################################    
    
    creon.checkLogin()
    
    ################################################################################
    #     16:00 ~ 24:00 [휴식]
    ################################################################################    
        
    time.sleep(60)
```