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
                 model_name: str = "gpt2",
                 db_path: str = "./chroma_db",
                 collection_name: str = "my_documents",
                 max_length: int = 512):
        """
        GPT-2 기반 LLM 시스템 초기화

        Args:
            model_name: 사용할 GPT-2 모델 이름
            db_path: ChromaDB 경로
            collection_name: 사용할 컬렉션 이름
            max_length: 생성할 최대 토큰 길이
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # GPT-2 모델 및 토크나이저 로드
        print(f"GPT-2 모델 로딩 중... ({model_name})")
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
                          temperature: float = 0.7,
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
            if use_context and self.collection:
                context = self.search_relevant_context(prompt)
                if context:
                    full_prompt = f"Context: {context}\n\nQuestion: {prompt}\nAnswer:"
                else:
                    full_prompt = f"Question: {prompt}\nAnswer:"
            else:
                full_prompt = prompt

            # 토큰화
            inputs = self.tokenizer.encode(full_prompt, return_tensors="pt", max_length=512, truncation=True)
            inputs = inputs.to(self.device)

            # 어텐션 마스크 생성
            attention_mask = torch.ones(inputs.shape, device=self.device)

            # 텍스트 생성
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    attention_mask=attention_mask,
                    max_length=min(len(inputs[0]) + 150, self.max_length),
                    temperature=temperature,
                    top_p=top_p,
                    num_return_sequences=num_return_sequences,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    repetition_penalty=1.1
                )

            # 결과 디코딩
            responses = []
            for output in outputs:
                # 원본 프롬프트 제거하고 생성된 부분만 추출
                generated_text = self.tokenizer.decode(output[len(inputs[0]):], skip_special_tokens=True)
                responses.append(generated_text.strip())

            return responses

        except Exception as e:
            print(f"텍스트 생성 중 오류: {e}")
            return ["오류가 발생했습니다."]

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