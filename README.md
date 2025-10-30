# 교보디티에스 사내게시판 검색 챗봇 프로그램

교보디티에스 사내게시판 내용을 검색기반 챗봇 프로그램

## 준비

소스를 다운받은후 

```
 pip install -r requirements.txt 
```
콘솔내에서 해당 문장 실행하여 관련라이브러리 다운로드 

## 구조

```
vectorDbTest/
├── chatbot.py # 게시판 문서를 분석·검색하는 AI 챗봇 메인 로직
├── make_vector_db.py # 텍스트 데이터를 벡터화하여 ChromaDB에 저장하는 스크립트
├── chroma_db2/ # 벡터디비 생성폴더
│
├── commonUtil/
│ └── timeCheck.py # 실행 시간, 성능 측정 등 유틸리티 함수
│
├── crawlingSource/
│ └── startCrawling.py # 회사 게시판 등 외부 소스에서 문서를 크롤링하는 스크립트
├── data/
│ └── 공지/ # 사내 공지사항 게시판 크롤링파일 보관폴더
│ └── 사우소식/ # 사내 사우소식 게시판 크롤링파일 보관폴더
│ └── 회사소식/ # 사내 회사소식 게시판 크롤링파일 보관폴더
```


## 흐름도
 게시판 크롤링 -> 백터db 생성 -> 검색 -> llm 모델 전송 -> 리턴 


## 실제실행 예제 
 1. startCrawling.py 
``` 
  bbs_Id = "B0000004"  # 추출하고 싶은 게시판 입력 이후 다른 변수 자동 셋팅됨
  boardTotalPg = 50; # 추출하고 싶은 페이지 만큼 입력 
```
원하는 게시판 과 페이지 수 입력 하고 실행

파일 생성 확인 

<img width="436" height="215" alt="image" src="https://github.com/user-attachments/assets/555533e9-647c-4bd9-b7bb-3c9de85a5edf" />

2.make_vector_db.py 실행
 벡터 디비 생성 확인 
 
 <img width="549" height="117" alt="image" src="https://github.com/user-attachments/assets/e231071e-8a98-407a-bf2e-7aa08945c91a" />

3.chatbot.py 실행 
 질문 프롬프트 뜨면 정상
 
<img width="892" height="356" alt="image" src="https://github.com/user-attachments/assets/bc31a238-b3a9-49ff-9a2c-cc7515d5fb8f" />
