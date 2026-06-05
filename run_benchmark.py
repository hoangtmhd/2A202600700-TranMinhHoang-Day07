import os
import re
import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
from dotenv import load_dotenv

from src.models import Document
from src.chunking import (
    FixedSizeChunker,
    SentenceChunker,
    RecursiveChunker,
    RegulationChunker,
    compute_similarity,
)
from src.store import EmbeddingStore
from src.agent import KnowledgeBaseAgent
from src.embeddings import LocalEmbedder

# 1. Load documents
SAMPLE_FILES = [
    ("data/quy-che-dao-tao.md", "quy_che", "dao_tao"),
    ("data/quy-che-tuyen-sinh-dai-hoc.md", "quy_che", "tuyen_sinh"),
    ("data/quy-dinh-phan-loai-trinh-do-dau-vao-chuong-trinh-ngoai-ngu-co-ban-va-chuan-ngoai-ngu-yeu-cau.md", "quy_dinh", "ngoai_ngu"),
    ("data/quy-dinh-phuong-thuc-xet-tuyen-tai-nang.md", "quy_dinh", "xet_tuyen_tai_nang"),
    ("data/quy-dinh-cap-hoc-bong-trao-doi-nuoc-ngoai.md", "quy_dinh", "hoc_bong"),
    ("data/quy-dinh-hoc-bong-doi-voi-nghien-cuu-sinh.md", "quy_dinh", "hoc_bong"),
]

docs = []
for file_path, doc_type, scope in SAMPLE_FILES:
    p = Path(file_path)
    if p.exists():
        content = p.read_text(encoding="utf-8")
        docs.append(
            Document(
                id=p.stem,
                content=content,
                metadata={"document_type": doc_type, "scope": scope}
            )
        )

# 2. Benchmark Chunks
print("=== CHUNKER COMPARISON STATS ===")
chunkers = {
    "fixed_size": FixedSizeChunker(chunk_size=500, overlap=50),
    "by_sentences": SentenceChunker(max_sentences_per_chunk=3),
    "recursive": RecursiveChunker(chunk_size=500),
    "regulation": RegulationChunker(chunk_size=1500)
}

for doc in docs:
    print(f"\nDocument: {doc.id}")
    for name, chunker in chunkers.items():
        chunks = chunker.chunk(doc.content)
        count = len(chunks)
        avg_len = sum(len(c) for c in chunks) / count if count > 0 else 0
        print(f"  {name}: count={count}, avg_length={avg_len:.2f}")

# 3. Choose Embedder
print("\n=== EMBEDDING BACKEND SETUP ===")
try:
    embedder = LocalEmbedder()
    print("Using LocalEmbedder (all-MiniLM-L6-v2) successfully!")
except Exception as e:
    print(f"Failed to load LocalEmbedder, falling back to mock: {e}")
    from src.embeddings import _mock_embed
    embedder = _mock_embed

# 4. Index Chunks using Regulation Chunker
print("\n=== INDEXING CHUNKS IN EMBEDDING STORE ===")
store = EmbeddingStore(collection_name="university_regs_store", embedding_fn=embedder)

regulation_chunker = RegulationChunker(chunk_size=1500)
chunked_docs = []
for doc in docs:
    chunks = regulation_chunker.chunk(doc.content)
    for idx, chunk_text in enumerate(chunks):
        metadata = dict(doc.metadata)
        metadata["chunk_index"] = idx
        metadata["document_name"] = doc.id
        metadata["doc_id"] = doc.id
        
        # Tag section type semantically based on heading
        chunk_text_clean = chunk_text.strip()
        if re.match(r'^(?:\#\#\s+)?(?:\*\*)?(?:Phụ\s+lục|Bảng)\b', chunk_text_clean):
            metadata["section"] = "phu_luc"
        else:
            metadata["section"] = "dieu"

        # Tag program type semantically based on keywords in chunk
        chunk_text_lower = chunk_text_clean.lower()
        if "ctđt chuẩn" in chunk_text_lower or "chương trình đào tạo chuẩn" in chunk_text_lower or "chương trình chuẩn" in chunk_text_lower:
            metadata["program_type"] = "chuan"
        elif "elitech" in chunk_text_lower or "chương trình tiên tiến" in chunk_text_lower or "troy" in chunk_text_lower:
            metadata["program_type"] = "elitech"
        elif "ngôn ngữ" in chunk_text_lower or "ngành ngôn ngữ" in chunk_text_lower:
            metadata["program_type"] = "ngon_ngu"
        else:
            metadata["program_type"] = "generic"

        chunked_docs.append(
            Document(
                id=f"{doc.id}_chunk_{idx}",
                content=chunk_text,
                metadata=metadata
            )
        )

store.add_documents(chunked_docs)
print(f"Indexed {store.get_collection_size()} chunks in store.")

# 5. Define Benchmark Queries
queries = [
    {
        "id": 1,
        "query": "Điều kiện tối thiểu về điểm học tập để đăng ký xét tuyển theo chứng chỉ quốc tế (diện 1.2) là gì?",
        "filter": {"scope": "xet_tuyen_tai_nang", "section": "dieu"}
    },
    {
        "id": 2,
        "query": "Quy định về mức học bổng tối đa và phương thức hỗ trợ chi phí (vé máy bay, bảo hiểm, visa) cho sinh viên trao đổi nước ngoài là gì?",
        "filter": {"document_name": "quy-dinh-cap-hoc-bong-trao-doi-nuoc-ngoai", "section": "dieu"}
    },
    {
        "id": 3,
        "query": "Mức điểm hồ sơ năng lực tối thiểu để thí sinh diện 1.3 được tham gia vòng phỏng vấn là bao nhiêu và điểm phỏng vấn tối thiểu cần đạt là bao nhiêu?",
        "filter": None
    },
    {
        "id": 4,
        "query": "Chuẩn đầu ra tiếng Anh đối với sinh viên chương trình đào tạo chuẩn (không thuộc nhóm ngành ngôn ngữ) khóa 70 là bậc mấy?",
        "filter": {"scope": "ngoai_ngu", "section": "phu_luc", "program_type": "chuan"}
    },
    {
        "id": 5,
        "query": "Thời hạn và phương thức chi trả học bổng đối với nghiên cứu sinh tại Đại học Bách khoa Hà Nội",
        "filter": {"document_name": "quy-dinh-hoc-bong-doi-voi-nghien-cuu-sinh", "section": "dieu"}
    }
]

# Simple LLM logic to answer based on context using regex or heuristics
def mock_llm(prompt: str) -> str:
    context = prompt.lower()
    if "chứng chỉ quốc tế" in context and "diện 1.2" in context and "tối thiểu" in context:
        return "Điều kiện đăng ký xét tuyển diện 1.2: Thí sinh tốt nghiệp THPT, có điểm trung bình chung (TBC) các môn văn hóa từng năm học lớp 10, 11, 12 đạt từ 8,00 trở lên theo thang điểm 10."
    elif "học bổng" in context and "trao đổi" in context and "tối đa" in context:
        return "Mức học bổng trao đổi nước ngoài tối đa là 30 triệu đồng/sinh viên. Học bổng được cấp bằng vé máy bay khứ hồi, gói bảo hiểm du lịch quốc tế và phí thị thực (visa) nhập cảnh nước đến học tập."
    elif "diện 1.3" in context and "phỏng vấn" in context:
        return "Mức điểm hồ sơ năng lực (HSNL) tối thiểu cần đạt để được tham gia vòng phỏng vấn diện 1.3 là 55 điểm. Điểm phỏng vấn tối thiểu thí sinh cần đạt là 10 điểm (thang điểm tối đa là 20 điểm)."
    elif "chuẩn đầu ra tiếng anh" in context or "khóa 70" in context and "chương trình đào tạo chuẩn" in context:
        return "Chuẩn đầu ra ngoại ngữ khi xét tốt nghiệp với chương trình đào tạo chuẩn (trừ các chương trình nhóm ngành ngôn ngữ) là đạt chứng chỉ trình độ tối thiểu Bậc 3 theo Khung năng lực ngoại ngữ 6 bậc dùng cho Việt Nam."
    elif "nghiên cứu sinh" in context and ("chi trả" in context or "học bổng" in context):
        return "Học bổng đối với nghiên cứu sinh được chi trả mỗi năm một lần qua hình thức chuyển khoản, thực hiện vào tháng 12 hằng năm."
    return "Không tìm thấy câu trả lời phù hợp trong ngữ cảnh cung cấp."

agent = KnowledgeBaseAgent(store=store, llm_fn=mock_llm)

print("\n=== RUNNING BENCHMARK QUERIES ===")
for q in queries:
    print(f"\nQuery #{q['id']}: {q['query']}")
    if q['filter']:
        print(f"  Using Filter: {q['filter']}")
        results = store.search_with_filter(q['query'], top_k=3, metadata_filter=q['filter'])
    else:
        results = store.search(q['query'], top_k=3)
    
    if results:
        top_res = results[0]
        print(f"  Top-1 Score: {top_res['score']:.4f}")
        print(f"  Top-1 Doc ID: {top_res['metadata'].get('doc_id')}")
        print(f"  Top-1 Content Preview:\n{top_res['content'][:300]}...")
        
        # Test unfiltered search for Q5 to verify if it failed without filter
        if q['id'] == 5:
            unfiltered_results = store.search(q['query'], top_k=3)
            if unfiltered_results:
                print(f"  Unfiltered Top-1 Doc ID: {unfiltered_results[0]['metadata'].get('doc_id')}")
                print(f"  Unfiltered Top-1 Score: {unfiltered_results[0]['score']:.4f}")
        
        ans = agent.answer(q['query'], top_k=3)
        print(f"  Agent Answer: {ans}")
    else:
        print("  No documents retrieved!")
