# 교보디티에스 사내게시판 검색 챗봇 프로그램

교보디티에스 사내게시판 내용을 검색기반 챗봇 프로그램

## 준비

소스를 다운받은후 

```
 pip install -r requirements.txt 
```
콘솔내에서 해당 문장 실행하여 관련라이브러리 다운로드 

### 구조

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


### 흐름도
 게시판 크롤링 -> 백터db 생성 -> 검색 -> llm 모델 전송 -> 리턴 

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
