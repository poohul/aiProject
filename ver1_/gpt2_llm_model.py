import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import warnings

warnings.filterwarnings('ignore')


class GPT2LLMSystem:
    def __init__(self,
                 model_name: str = "skt/kogpt2-base-v2",  # 한국어 GPT-2 모델 사용
                 db_path: str = "./chroma_db",
                 collection_name: str = "my_documents",
                 max_length: int = 512):
        """
        GPT-2 기반 LLM 시스템 초기화

        Args:
            model_name: 사용할 GPT-2 모델 이름 (한국어 모델로 변경)
            db_path: ChromaDB 경로
            collection_name: 사용할 컬렉션 이름
            max_length: 생성할 최대 토큰 길이
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # GPT-2 모델 및 토크나이저 로드
        print(f"GPT-2 모델 로딩 중... ({model_name})")
        try:
            # 한국어 GPT-2 모델 시도
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        except:
            # 실패시 영어 GPT-2 모델로 폴백
            print("한국어 모델 로드 실패, 기본 GPT-2 모델을 사용합니다.")
            model_name = "gpt2"
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.model = GPT2LMHeadModel.from_pretrained(model_name)

        # 패딩 토큰 설정
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.to(self.device)
        print(f"모델이 {self.device}에 로드되었습니다.")

        # 벡터 DB 연결
        self.setup_vector_db(db_path, collection_name)

    def setup_vector_db(self, db_path: str, collection_name: str):
        """벡터 DB 설정"""
        try:
            self.client = chromadb.PersistentClient(path=db_path)
            self.collection = self.client.get_collection(collection_name)
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            print(f"벡터 DB 연결 완료: {collection_name}")
        except Exception as e:
            print(f"벡터 DB 연결 실패: {e}")
            self.collection = None
            self.embedding_model = None

    def search_relevant_context(self, query: str, n_results: int = 3) -> str:
        """
        관련 컨텍스트 검색

        Args:
            query: 검색 쿼리
            n_results: 반환할 결과 수

        Returns:
            관련 문서들을 합친 컨텍스트 문자열
        """
        if not self.collection or not self.embedding_model:
            return ""

        try:
            # 쿼리 임베딩 생성
            query_embedding = self.embedding_model.encode([query])

            # 유사 문서 검색
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=n_results
            )

            # 검색된 문서들을 컨텍스트로 합치기
            documents = results.get('documents', [[]])[0]
            context = "\n".join(documents)

            return context

        except Exception as e:
            print(f"컨텍스트 검색 중 오류: {e}")
            return ""

    def generate_response(self,
                          prompt: str,
                          use_context: bool = True,
                          temperature: float = 0.8,
                          top_p: float = 0.9,
                          num_return_sequences: int = 1) -> List[str]:
        """
        GPT-2를 사용하여 텍스트 생성

        Args:
            prompt: 입력 프롬프트
            use_context: 벡터 DB에서 관련 컨텍스트 사용 여부
            temperature: 생성 다양성 조절 (0.1~1.0)
            top_p: 누적 확률 임계값
            num_return_sequences: 생성할 응답 수

        Returns:
            생성된 텍스트 리스트
        """
        try:
            # 관련 컨텍스트 검색 및 추가
            context = ""
            if use_context and self.collection:
                context = self.search_relevant_context(prompt, n_results=2)

            # 프롬프트 구성 개선
            if context:
                full_prompt = f"""다음은 참고 자료입니다:
{context}

질문: {prompt}
답변:"""
            else:
                # 기본 한국어 프롬프트 템플릿
                full_prompt = f"""질문: {prompt}
답변:"""

            # 토큰화 (입력 길이 제한)
            inputs = self.tokenizer.encode(
                full_prompt,
                return_tensors="pt",
                max_length=300,  # 더 짧게 제한해서 생성 공간 확보
                truncation=True
            )
            inputs = inputs.to(self.device)

            # 어텐션 마스크 생성
            attention_mask = torch.ones(inputs.shape, device=self.device)

            # 텍스트 생성 파라미터 개선
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    attention_mask=attention_mask,
                    max_new_tokens=100,  # 더 짧게 생성
                    temperature=temperature,
                    top_p=top_p,
                    top_k=50,  # top_k 추가로 품질 개선
                    num_return_sequences=num_return_sequences,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    repetition_penalty=1.2,  # 반복 방지 강화
                    eos_token_id=self.tokenizer.eos_token_id,
                    early_stopping=True,  # 조기 종료 활성화
                    no_repeat_ngram_size=3  # 3-gram 반복 방지
                )

            # 결과 디코딩 및 정리
            responses = []
            for output in outputs:
                # 원본 프롬프트 제거하고 생성된 부분만 추출
                generated_text = self.tokenizer.decode(output[len(inputs[0]):], skip_special_tokens=True)

                # 텍스트 정리
                generated_text = generated_text.strip()

                # 불완전한 문장 제거
                if generated_text:
                    # 첫 번째 완전한 문장만 추출
                    sentences = generated_text.split('.')
                    if len(sentences) > 1 and sentences[0].strip():
                        generated_text = sentences[0].strip() + '.'

                    # 너무 짧은 응답 필터링
                    if len(generated_text.strip()) < 10:
                        generated_text = "죄송합니다. 적절한 답변을 생성하지 못했습니다."

                responses.append(generated_text if generated_text else "답변을 생성할 수 없습니다.")

            return responses

        except Exception as e:
            print(f"텍스트 생성 중 오류: {e}")
            return ["죄송합니다. 답변 생성 중 오류가 발생했습니다."]

    def chat(self):
        """대화형 채팅 인터페이스"""
        print("GPT-2 기반 LLM 시스템이 준비되었습니다.")
        print("'quit' 또는 'exit'을 입력하면 종료됩니다.\n")

        while True:
            user_input = input("사용자: ").strip()

            if user_input.lower() in ['quit', 'exit', '종료']:
                print("채팅을 종료합니다.")
                break

            if not user_input:
                continue

            print("AI가 응답을 생성 중...")
            responses = self.generate_response(user_input)

            print(f"\nAI: {responses[0]}\n")

    def batch_generate(self, prompts: List[str]) -> List[str]:
        """
        여러 프롬프트에 대해 일괄 생성

        Args:
            prompts: 프롬프트 리스트

        Returns:
            생성된 응답 리스트
        """
        responses = []
        for i, prompt in enumerate(prompts):
            print(f"프롬프트 {i + 1}/{len(prompts)} 처리 중...")
            response = self.generate_response(prompt)[0]
            responses.append(response)

        return responses

    def get_db_info(self) -> Dict:
        """벡터 DB 정보 반환"""
        if not self.collection:
            return {"error": "벡터 DB가 연결되지 않음"}

        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection.name,
                "document_count": count,
                "db_path": self.db_path
            }
        except Exception as e:
            return {"error": str(e)}


def main():
    """메인 실행 함수"""
    # LLM 시스템 초기화
    llm_system = GPT2LLMSystem()

    # DB 정보 출력
    db_info = llm_system.get_db_info()
    print(f"데이터베이스 정보: {db_info}")

    # 테스트 질문들
    test_queries = [
        "안녕하세요",
        "이 문서의 주요 내용은 무엇인가요?",
        "요약해주세요"
    ]

    print("\n=== 테스트 질문 결과 ===")
    for query in test_queries:
        print(f"\n질문: {query}")
        response = llm_system.generate_response(query)[0]
        print(f"답변: {response}")

    # 대화형 모드 시작
    print("\n=== 대화형 모드 ===")
    llm_system.chat()


if __name__ == "__main__":
    main()