import os
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict
import re


class VectorDBCreator:
    def __init__(self, db_path: str = "./chroma_db", model_name: str = "all-MiniLM-L6-v2"):
        """
        벡터 DB 생성기 초기화

        Args:
            db_path: ChromaDB 저장 경로
            model_name: 임베딩 모델 이름
        """
        self.db_path = db_path
        self.client = chromadb.PersistentClient(path=db_path)
        self.embedding_model = SentenceTransformer(model_name)
        self.collection = None

    def read_text_file(self, file_path: str) -> str:
        """
        텍스트 파일 읽기

        Args:
            file_path: 읽을 텍스트 파일 경로

        Returns:
            파일 내용 문자열
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            print(f"파일 '{file_path}' 읽기 완료 (길이: {len(content)} 문자)")
            return content
        except FileNotFoundError:
            print(f"파일 '{file_path}'을 찾을 수 없습니다.")
            return ""
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='cp949') as file:
                    content = file.read()
                print(f"파일 '{file_path}' 읽기 완료 (cp949 인코딩, 길이: {len(content)} 문자)")
                return content
            except UnicodeDecodeError:
                print(f"파일 '{file_path}' 인코딩 오류")
                return ""

    def split_text_into_chunks(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        텍스트를 청크 단위로 분할

        Args:
            text: 분할할 텍스트
            chunk_size: 청크 크기 (문자 단위)
            overlap: 청크 간 겹치는 부분 크기

        Returns:
            분할된 텍스트 청크 리스트
        """
        # 문장 단위로 먼저 분할
        sentences = re.split(r'[.!?]\s+', text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "

        # 마지막 청크 추가
        if current_chunk:
            chunks.append(current_chunk.strip())

        print(f"텍스트를 {len(chunks)}개 청크로 분할 완료")
        return chunks

    def create_vector_db(self, collection_name: str, text_chunks: List[str]) -> bool:
        """
        벡터 DB 생성

        Args:
            collection_name: 컬렉션 이름
            text_chunks: 텍스트 청크 리스트

        Returns:
            성공 여부
        """
        try:
            # 기존 컬렉션이 있다면 삭제
            try:
                self.client.delete_collection(collection_name)
                print(f"기존 컬렉션 '{collection_name}' 삭제됨")
            except:
                pass

            # 새 컬렉션 생성
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )

            # 임베딩 생성
            print("임베딩 생성 중...")
            embeddings = self.embedding_model.encode(text_chunks, show_progress_bar=True)

            # 문서 ID 생성
            ids = [f"doc_{i}" for i in range(len(text_chunks))]

            # 메타데이터 생성
            metadatas = [{"chunk_id": i, "length": len(chunk)} for i, chunk in enumerate(text_chunks)]

            # 벡터 DB에 추가
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=text_chunks,
                metadatas=metadatas,
                ids=ids
            )

            print(f"벡터 DB 생성 완료: {len(text_chunks)}개 문서 저장됨")
            return True

        except Exception as e:
            print(f"벡터 DB 생성 중 오류 발생: {e}")
            return False

    def search_similar_documents(self, query: str, n_results: int = 3) -> Dict:
        """
        유사한 문서 검색

        Args:
            query: 검색 쿼리
            n_results: 반환할 결과 수

        Returns:
            검색 결과 딕셔너리
        """
        if not self.collection:
            print("컬렉션이 초기화되지 않았습니다.")
            return {}

        try:
            # 쿼리 임베딩 생성
            query_embedding = self.embedding_model.encode([query])

            # 유사 문서 검색
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=n_results
            )

            return results

        except Exception as e:
            print(f"검색 중 오류 발생: {e}")
            return {}


def main():
    # 사용 예시
    vector_db = VectorDBCreator()

    # 텍스트 파일 읽기
    file_path = "sample.txt"  # 실제 파일 경로로 변경하세요
    text_content = vector_db.read_text_file(file_path)

    if text_content:
        # 텍스트 청크 분할
        chunks = vector_db.split_text_into_chunks(text_content)

        # 벡터 DB 생성
        success = vector_db.create_vector_db("my_documents", chunks)

        if success:
            # 검색 테스트
            query = "검색하고 싶은 내용"  # 실제 검색어로 변경하세요
            results = vector_db.search_similar_documents(query)

            print(f"\n검색 결과:")
            for i, doc in enumerate(results.get('documents', [[]])[0]):
                print(f"{i + 1}. {doc[:100]}...")


if __name__ == "__main__":
    main()