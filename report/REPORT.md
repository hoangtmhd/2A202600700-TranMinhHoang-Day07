# Báo Cáo Lab 7: Embedding & Vector Store
 
**Họ tên:** Trần Minh Hoàng
**Nhóm:** Nhóm thanh niên áo kẻ
**Ngày:** 05/06/2026
 
---
 
## 1. Warm-up (5 điểm)
 
### Cosine Similarity (Ex 1.1)
 
**High cosine similarity nghĩa là gì?**
> Độ tương đồng cosine (cosine similarity) cao nghĩa là hai vector embedding của hai đoạn văn bản có hướng (hướng chỉ các đặc trưng ngữ nghĩa) rất gần nhau trong không gian vector nhiều chiều, cho thấy hai đoạn văn bản có sự tương đồng lớn về mặt ngữ nghĩa hoặc ngữ cảnh.
 
**Ví dụ HIGH similarity:**
- Sentence A: "Chó là loài vật nuôi vô cùng trung thành với con người."
- Sentence B: "Loài thú bốn chân này luôn gắn bó và hết lòng bảo vệ chủ."
- Tại sao tương đồng: Hai câu diễn đạt cùng một ý nghĩa cốt lõi (chó/loài thú bốn chân là loài vật trung thành/gắn bó bảo vệ) bằng các từ ngữ khác nhau.
 
**Ví dụ LOW similarity:**
- Sentence A: "Học thuyết tương đối của Einstein đã cách mạng hóa vật lý hiện đại."
- Sentence B: "Để làm bánh bông lan, bạn cần chuẩn bị bột mì, đường và trứng."
- Tại sao khác: Hai câu thuộc hai chủ đề hoàn toàn khác nhau và không có mối liên hệ ngữ nghĩa nào (vật lý học vs. nấu ăn).
 
**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity chỉ đo góc giữa hai vector (tập trung vào hướng/ngữ nghĩa) và không bị ảnh hưởng bởi độ dài (độ lớn vector) của văn bản. Euclidean distance nhạy cảm với độ dài văn bản, nên hai văn bản có cùng nội dung nhưng độ dài khác nhau sẽ bị đánh giá là cách xa nhau.
 
### Chunking Math (Ex 1.2)
 
**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:*
> Áp dụng công thức: num_chunks = ceil((doc_length - overlap) / (chunk_size - overlap))
> Với doc_length = 10000, chunk_size = 500, overlap = 50:
> num_chunks = ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = ceil(22.11) = 23
> *Đáp án:* 23 chunks
 
**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Khi overlap tăng lên 100: num_chunks = ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = ceil(24.75) = 25 chunks (số lượng chunk tăng lên 25).
> Việc muốn overlap nhiều hơn giúp đảm bảo rằng ngữ cảnh ở các điểm ranh giới phân tách không bị mất hoặc bị cắt đôi; phần thông tin giáp ranh sẽ nằm trọn vẹn trong ít nhất một chunk, giúp quá trình truy xuất (retrieval) và sinh câu trả lời trong hệ thống RAG chính xác hơn.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** [ví dụ: Customer support FAQ, Vietnamese law, cooking recipes, ...]

**Tại sao nhóm chọn domain này?**
> *Viết 2-3 câu:*

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| | | | |
| | | | |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| | FixedSizeChunker (`fixed_size`) | | | |
| | SentenceChunker (`by_sentences`) | | | |
| | RecursiveChunker (`recursive`) | | | |

### Strategy Của Tôi

**Loại:** [FixedSizeChunker / SentenceChunker / RecursiveChunker / custom strategy]

**Mô tả cách hoạt động:**
> *Viết 3-4 câu: strategy chunk thế nào? Dựa trên dấu hiệu gì?*

**Tại sao tôi chọn strategy này cho domain nhóm?**
> *Viết 2-3 câu: domain có pattern gì mà strategy khai thác?*

**Code snippet (nếu custom):**
```python
# Paste implementation here
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| | best baseline | | | |
| | **của tôi** | | | |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | | | | |
| [Tên] | | | | |
| [Tên] | | | | |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> *Viết 2-3 câu:*

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Chúng tôi dùng regex `r'(\. |\! |\? |\.\n)'` để phân chia văn bản thành các câu và giữ lại các ký tự phân tách (separator) ở cuối mỗi câu bằng tính năng bắt nhóm (capturing group). Sau đó, ghép phần thân câu và ký tự phân tách lại với nhau để khôi phục cấu trúc hoàn chỉnh, rồi gom nhóm các câu này thành từng chunk chứa tối đa `max_sentences_per_chunk` câu trước khi làm sạch khoảng trắng bằng `strip()`.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Thuật toán hoạt động đệ quy bằng cách duyệt qua danh sách các ký tự phân tách theo độ ưu tiên `["\n\n", "\n", ". ", " ", ""]`. Nếu đoạn văn bản hiện tại lớn hơn `chunk_size`, nó sẽ được tách ra bằng ký tự phân tách hiện tại, sau đó thử ghép các mảnh nhỏ lại mà không vượt quá giới hạn kích thước; các mảnh đơn lẻ vẫn vượt giới hạn sẽ được đưa vào hàm đệ quy để tách tiếp bằng các ký tự phân tách có độ ưu tiên thấp hơn.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Mỗi tài liệu được lưu trữ song song vào danh sách trong bộ nhớ (in-memory list) và cơ sở dữ liệu ChromaDB. Để tìm kiếm, hệ thống tính toán embedding của truy vấn, sau đó tính điểm cosine similarity giữa truy vấn và tất cả các chunk trong kho dữ liệu, cuối cùng sắp xếp giảm dần và lấy ra top_k kết quả tương đồng nhất.

**`search_with_filter` + `delete_document`** — approach:
> Khi lọc theo metadata (`search_with_filter`), chúng tôi thực hiện tiền lọc (pre-filtering) các bản ghi khớp với bộ lọc key-value trước khi thực hiện tính toán độ tương đồng và xếp hạng. Hàm xóa tài liệu (`delete_document`) thực hiện lọc bỏ toàn bộ các bản ghi trong danh sách bộ nhớ có `metadata['doc_id'] == doc_id` và đồng thời gọi xóa tương ứng trên bộ sưu tập ChromaDB.

### KnowledgeBaseAgent

**`answer`** — approach:
> Hàm answer thực hiện truy xuất top_k chunk tương quan nhất từ cơ sở dữ liệu vector làm ngữ cảnh (context), sau đó định dạng và đưa chúng cùng với câu hỏi của người dùng vào một prompt mẫu RAG hoàn chỉnh để chuyển cho mô hình ngôn ngữ lớn (LLM) xử lý tạo câu trả lời.

### Test Results

```
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.0.3, pluggy-1.6.0 -- D:\Work\Study\ai-in-action\Lab7\2A202600700-TranMinhHoang-Day07\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: D:\Work\Study\ai-in-action\Lab7\2A202600700-TranMinhHoang-Day07
plugins: anyio-4.13.0
collecting ... collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED   [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED    [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED   [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================= 42 passed in 0.80s ==============================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | The weather is nice today. | It is a sunny day outside. | high | 0.1185 | Không |
| 2 | I love eating pizza. | Cats are very cute animals. | low | -0.0854 | Đúng |
| 3 | Python is a programming language. | Java is also a coding language. | high | -0.1130 | Không |
| 4 | Quantum physics is hard. | Baking a cake requires flour. | low | 0.0841 | Không |
| 5 | Artificial intelligence is growing. | Machine learning is expanding. | high | 0.0031 | Không |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Kết quả bất ngờ nhất là các câu cùng chủ đề và ngữ nghĩa rất gần nhau (như Cặp 3 và 5) lại có điểm tương đồng rất thấp hoặc âm, trong khi cặp câu hoàn toàn không liên quan (Cặp 4) lại có điểm tương đồng dương cao hơn. Điều này chỉ ra rằng `MockEmbedder` chỉ sử dụng hàm băm MD5 để tạo ngẫu nhiên các giá trị vector (random unit vectors), dẫn đến các giá trị tương đồng phân bố ngẫu nhiên quanh 0 mà không hề phản ánh ngữ nghĩa thực tế. Một mô hình embedding thực sự (như của OpenAI hay HuggingFace) cần được học từ dữ liệu ngữ nghĩa thực tế để xếp các câu có nghĩa gần nhau vào các góc hẹp trong không gian vector.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | | |
| 2 | | |
| 3 | | |
| 4 | | |
| 5 | | |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |

**Bao nhiêu queries trả về chunk relevant trong top-3?** __ / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> *Viết 2-3 câu:*

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> *Viết 2-3 câu:*

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> *Viết 2-3 câu:*

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | / 5 |
| Document selection | Nhóm | / 10 |
| Chunking strategy | Nhóm | / 15 |
| My approach | Cá nhân | / 10 |
| Similarity predictions | Cá nhân | / 5 |
| Results | Cá nhân | / 10 |
| Core implementation (tests) | Cá nhân | / 30 |
| Demo | Nhóm | / 5 |
| **Tổng** | | **/ 100** |
