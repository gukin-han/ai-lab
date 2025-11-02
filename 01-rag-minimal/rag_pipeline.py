# rag_pipeline.py
import os
import glob
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI


@dataclass
class DocChunk:
    doc_id: str
    text: str
    embedding: np.ndarray


class RAGPipeline:
    """
    아주 단순한 in-memory RAG 파이프라인
    - docs 폴더의 .txt/.md 읽어서
    - chunk로 자르고
    - OpenAI 임베딩으로 바꿔서
    - 질의 들어오면 코사인 유사도로 top_k 뽑고
    - 그걸 컨텍스트로 LLM에 보내서 답변 생성
    """

    def __init__(
        self,
        docs_path: str = "docs",
        top_k: int = 3,
        embedding_model: str = "text-embedding-3-small",
        chat_model: str = "gpt-4o-mini",
    ):
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY 가 .env 에 없습니다.")
        self.client = OpenAI(api_key=api_key)

        self.docs_path = docs_path
        self.top_k = top_k
        self.embedding_model = embedding_model
        self.chat_model = chat_model

        self.chunks: List[DocChunk] = []

    # ---------- public API ----------
    def build_index(self) -> None:
        """docs/ 폴더의 파일들을 읽어 in-memory index를 만든다."""
        texts = self._load_docs()
        chunks = self._chunk_all(texts)
        embedded = self._embed_chunks(chunks)
        self.chunks = embedded
        print(f"[index] {len(self.chunks)} 개의 chunk 로 인덱스 구성 완료")

    def query(self, question: str) -> str:
        """질문 하나를 받아서 RAG 로 답한다."""
        q_emb = self._embed_query(question)
        top_chunks = self._search(q_emb, top_k=self.top_k)
        context = self._build_context(top_chunks)
        answer = self._call_llm(question, context)
        return answer

    # ---------- internal ----------
    def _load_docs(self) -> Dict[str, str]:
        """docs/ 이하의 .txt, .md 파일을 전부 읽어온다."""
        os.makedirs(self.docs_path, exist_ok=True)
        paths = glob.glob(os.path.join(self.docs_path, "*.txt")) + glob.glob(
            os.path.join(self.docs_path, "*.md")
        )
        docs = {}
        for p in paths:
            with open(p, "r", encoding="utf-8") as f:
                docs[os.path.basename(p)] = f.read()
        return docs

    def _chunk_all(self, docs: Dict[str, str]) -> List[Tuple[str, str]]:
        """아주 단순한 chunking: 글자 수 기준으로 자름."""
        all_chunks: List[Tuple[str, str]] = []
        chunk_size = 500  # 글자 수 기준
        overlap = 100

        for doc_id, text in docs.items():
            start = 0
            while start < len(text):
                end = start + chunk_size
                chunk = text[start:end]
                all_chunks.append((doc_id, chunk))
                start += chunk_size - overlap

        return all_chunks

    def _embed_chunks(self, chunks: List[Tuple[str, str]]) -> List[DocChunk]:
        embedded: List[DocChunk] = []
        for doc_id, text in chunks:
            emb = self._embed_text(text)
            embedded.append(DocChunk(doc_id=doc_id, text=text, embedding=emb))
        return embedded

    def _embed_query(self, query: str) -> np.ndarray:
        return self._embed_text(query)

    def _embed_text(self, text: str) -> np.ndarray:
        # OpenAI 임베딩 호출
        resp = self.client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )
        vec = np.array(resp.data[0].embedding, dtype=np.float32)
        return vec

    def _search(self, query_emb: np.ndarray, top_k: int = 3) -> List[DocChunk]:
        """코사인 유사도 기반 top_k 검색"""
        scored = []
        for chunk in self.chunks:
            score = self._cosine_similarity(query_emb, chunk.embedding)
            scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    def _build_context(self, chunks: List[DocChunk]) -> str:
        parts = []
        for i, c in enumerate(chunks, start=1):
            parts.append(f"[출처 {i} - {c.doc_id}]\n{c.text.strip()}\n")
        return "\n".join(parts)

    def _call_llm(self, question: str, context: str) -> str:
        system_prompt = (
            "너는 업로드된 문서를 바탕으로만 대답하는 RAG 어시스턴트다. "
            "모르면 모른다고 말해. 출처에 없는 내용은 추측하지 마."
        )
        user_prompt = (
            f"다음은 너에게 주어진 참고 문서이다:\n\n{context}\n\n"
            f"위 문서를 근거로 다음 질문에 한국어로 답해라:\n{question}"
        )

        resp = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
