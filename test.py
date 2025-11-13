from sentence_transformers import CrossEncoder

# model = CrossEncoder('./custom_kyoboDTS_bbs_reranker')
model = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2')
# query = "승진자 명단"
# docs = [
#     "인사발령 ( 승진 ) 을 다음과 같이 공지합니다.",
#     "신규입사자를 소개합니다.",
#     "FY2024 임금 및 단체협약 체결 결과 공지"
# ]


query = "최근 득남/득녀 소식"
docs = [
    "경조휴가 및 경조금 지급기준 공지",
    "※ 자녀장학금 신청공지(대학교 1학기, 고등학교/중학교/초등학교/유치원 年1회)※",
    "♥[축] 인프라사업1팀 장성훈 대리 득녀 ♥"
]

scores = model.predict([(query, doc) for doc in docs])
for d, s in zip(docs, scores):
    print(f"{s:.3f} → {d}")
