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

**Domain:** Hệ thống Cố vấn Quy chế Học vụ & Tuyển sinh Đại học Bách khoa Hà Nội (HUST Academic Regulations & Admission Advisor).

**Tại sao nhóm chọn domain này?**
> Nhóm chọn domain này vì các văn bản quy chế hành chính của HUST có tính phân cấp rất nghiêm ngặt và chứa nhiều bảng tra cứu chéo phức tạp (như bảng điểm thưởng, quy đổi chứng chỉ ngoại ngữ). Đây là một tập dữ liệu thực tế và thử thách, giúp nhóm kiểm nghiệm sâu sắc năng lực xử lý của các chiến lược chunking và cơ chế lọc metadata trước khi tìm kiếm nhằm tối ưu hóa độ chính xác cho hệ thống RAG.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | quy-che-dao-tao.md | Đại học Bách khoa Hà Nội | 78,714 | `document_type: "quy_che"`, `scope: "dao_tao"` |
| 2 | quy-che-tuyen-sinh-dai-hoc.md | Đại học Bách khoa Hà Nội | 42,294 | `document_type: "quy_che"`, `scope: "tuyen_sinh"` |
| 3 | quy-dinh-phan-loai-trinh-do-dau-vao-chuong-trinh-ngoai-ngu-co-ban-va-chuan-ngoai-ngu-yeu-cau.md | Đại học Bách khoa Hà Nội | 32,088 | `document_type: "quy_dinh"`, `scope: "ngoai_ngu"` |
| 4 | quy-dinh-phuong-thuc-xet-tuyen-tai-nang.md | Đại học Bách khoa Hà Nội | 23,936 | `document_type: "quy_dinh"`, `scope: "xet_tuyen_tai_nang"` |
| 5 | quy-dinh-cap-hoc-bong-trao-doi-nuoc-ngoai.md | Đại học Bách khoa Hà Nội | 22,153 | `document_type: "quy_dinh"`, `scope: "hoc_bong"` |
| 6 | quy-dinh-hoc-bong-doi-voi-nghien-cuu-sinh.md | Đại học Bách khoa Hà Nội | 7,940 | `document_type: "quy_dinh"`, `scope: "hoc_bong"` |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `document_type` | `string` | `"quy_che"`, `"quy_dinh"` | Giúp phân biệt nhanh giữa quy chế chung toàn đại học và các quyết định/quy định bổ sung riêng lẻ. |
| `scope` | `string` | `"dao_tao"`, `"tuyen_sinh"`, `"ngoai_ngu"`, `"xet_tuyen_tai_nang"`, `"hoc_bong"` | Gom nhóm các văn bản có cùng mục tiêu điều chỉnh. Giúp lọc bớt nhiễu khi các quy định có các từ khóa trùng lặp nhau. |
| `section` | `string` | `"dieu"`, `"phu_luc"` | Phân biệt phần thân quy chế (Articles) và phần Phụ lục/Bảng biểu (Appendix). Rất hữu ích khi câu hỏi chỉ hỏi về bảng tra cứu chuẩn ở phụ lục. |
| `program_type` | `string` | `"chuan"`, `"elitech"`, `"ngon_ngu"`, `"generic"` | Xác định loại chương trình đào tạo áp dụng. Giúp loại bỏ nhiễu chéo giữa quy chế của các chương trình khác nhau. |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên các tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| quy-che-dao-tao.md | FixedSizeChunker (`fixed_size`) | 175 | 499.51 | No (bị cắt cụt giữa các từ hoặc ranh giới điều) |
| quy-che-dao-tao.md | SentenceChunker (`by_sentences`) | 243 | 322.20 | Partially (không bảo đảm liên kết giữa các câu cùng Điều) |
| quy-che-dao-tao.md | RecursiveChunker (`recursive`) | 206 | 380.12 | Partially (giữ được đoạn văn nhưng vẫn bị phân mảnh lớn) |
| quy-che-tuyen-sinh-dai-hoc.md | FixedSizeChunker (`fixed_size`) | 94 | 499.40 | No (phá cấu trúc bảng biểu hoặc ranh giới điều khoản) |
| quy-che-tuyen-sinh-dai-hoc.md | SentenceChunker (`by_sentences`) | 108 | 389.80 | Partially (tách câu đơn lẻ nên mất ngữ cảnh của Điều) |
| quy-che-tuyen-sinh-dai-hoc.md | RecursiveChunker (`recursive`) | 105 | 400.82 | Partially (khá tốt nhưng ranh giới chunk không khớp Điều) |
| quy-dinh-cap-hoc-bong-trao-doi-nuoc-ngoai.md | FixedSizeChunker (`fixed_size`) | 50 | 492.06 | No (bị ngắt nửa chừng các điều khoản cấp học bổng) |
| quy-dinh-cap-hoc-bong-trao-doi-nuoc-ngoai.md | SentenceChunker (`by_sentences`) | 63 | 349.86 | Partially (không giữ liên kết giữa các khoản) |
| quy-dinh-cap-hoc-bong-trao-doi-nuoc-ngoai.md | RecursiveChunker (`recursive`) | 55 | 400.82 | Partially (khá tốt nhưng ranh giới chunk không khớp Điều) |

### Strategy Của Tôi

**Loại:** Custom Strategy (`RegulationChunker`) với Metadata Filtering nâng cao

**Mô tả cách hoạt động:**
> `RegulationChunker` ban đầu chia cắt văn bản theo cấu trúc của từng Điều luật trong văn bản quy chế (sử dụng regex). Để tối ưu hóa cho các bảng biểu phụ lục, chúng tôi đã mở rộng Regex để nhận diện thêm ranh giới của các **Phụ lục** (`Phụ lục X`) và các **Bảng** (`Bảng X.Y`). Khi phát hiện, chunker sẽ cắt và gom toàn bộ nội dung của Điều/Phụ lục/Bảng đó vào một chunk riêng biệt. Nếu nội dung quá lớn (vượt quá `chunk_size` tối đa là 1500 ký tự), nó sẽ sử dụng `RecursiveChunker` để chia nhỏ đệ quy. 
> Đặc biệt, khi index, hệ thống tự động gán hai nhãn metadata động: `section` (`dieu` hoặc `phu_luc` dựa trên ranh giới) và `program_type` (`chuan`, `elitech`, `ngon_ngu` dựa trên phân tích từ khóa trong chunk) giúp pre-filtering cực kỳ chính xác.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Vì toàn bộ tài liệu trong domain của nhóm đều là văn bản quy chế và quy định có tính pháp lý. Mỗi Điều trong quy chế luôn là một đơn vị thông tin độc lập và tự nhất quán (chứa nội dung đầy đủ về một quy định cụ thể). Việc chunk theo Điều bảo đảm khi hệ thống truy xuất được một đoạn thông tin, sinh viên sẽ đọc được toàn bộ nội dung của Điều đó mà không lo bị mất các Khoản hay Điểm đi kèm, từ đó nâng cao tính đúng đắn và toàn vẹn của ngữ cảnh RAG.

**Code snippet (nếu custom):**
```python
class RegulationChunker:
    """
    Split Vietnamese regulation documents by Article (Điều) headings or sections.
    Falls back to RecursiveChunker if no Articles are found.
    """

    def __init__(self, chunk_size: int = 1500) -> None:
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text.strip():
            return []

        # Find occurrences of "Điều X" at paragraph/line start
        matches = list(re.finditer(r'(?:\n|^)(?:\#\#\s+)?(?:\*\*)?Điều\s+\d+\b', text))

        if not matches:
            return RecursiveChunker(chunk_size=self.chunk_size).chunk(text)

        chunks: list[str] = []
        first_start = matches[0].start()
        intro = text[:first_start].strip()
        if intro:
            if len(intro) > self.chunk_size:
                chunks.extend(RecursiveChunker(chunk_size=self.chunk_size).chunk(intro))
            else:
                chunks.append(intro)

        for i in range(len(matches)):
            start = matches[i].start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            chunk_text = text[start:end].strip()
            if not chunk_text:
                continue

            if len(chunk_text) > self.chunk_size:
                chunks.extend(RecursiveChunker(chunk_size=self.chunk_size).chunk(chunk_text))
            else:
                chunks.append(chunk_text)

        return chunks
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| quy-che-dao-tao.md | Recursive (best baseline) | 206 | 380.12 | Trung bình (đôi khi bị đứt mạch thông tin giữa Điều này và Điều khác) |
| quy-che-dao-tao.md | **của tôi** (`regulation`) | 85 | 923.44 | Rất tốt (mỗi chunk là một Điều trọn vẹn, không bị rời rạc) |
| quy-che-tuyen-sinh-dai-hoc.md | Recursive (best baseline) | 105 | 400.82 | Trung bình (dễ cắt đứt bảng biểu ở phần mục lục) |
| quy-che-tuyen-sinh-dai-hoc.md | **của tôi** (`regulation`) | 46 | 916.78 | Rất tốt (các Điều khoản tuyển sinh được gom gọn cùng nhau) |
| quy-dinh-cap-hoc-bong-trao-doi-nuoc-ngoai.md | Recursive (best baseline) | 55 | 400.82 | Trung bình |
| quy-dinh-cap-hoc-bong-trao-doi-nuoc-ngoai.md | **của tôi** (`regulation`) | 30 | 735.80 | Rất tốt (các quy định và thủ tục cấp học bổng trọn vẹn) |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Trần Minh Hoàng | RegulationChunker | 9.0/10 | Giữ nguyên vẹn cấu trúc và ngữ nghĩa của từng Điều luật, ít phân mảnh | Chunk dài hơn làm giảm độ chính xác tập trung của embedding |
| Nguyễn Thế Giáp | RecursiveChunker | 9.0/10 | Giữ ngữ cảnh điều/khoản/phụ lục tốt, phù hợp văn bản quy chế hành chính có cấu trúc phân cấp rõ, giúp retrieval ổn định trên các tài liệu dài. | Số chunk nhiều hơn các cách cắt thô nên tốn chi phí index/search hơn và vẫn có thể tạo chunk chưa tối ưu khi gặp bảng quá dài. |
| Nguyễn Hữu Thái Minh | HUSTArticleChunker | 9.0/10 | Giảm số lượng chunk tới hơn 60%, bảo toàn trọn vẹn ngữ cảnh của từng Điều/Khoản, giữ nguyên vẹn cấu trúc bảng biểu và chu trình thủ tục. | Kích thước chunk lớn (trung bình ~1500 ký tự) làm loãng embedding ngữ nghĩa nếu query chứa thông tin quá chi tiết. |
| Nguyễn Đức Tâm | RecursiveChunker | 9.0/10 | Giữ ngữ cảnh điều/khoản/phụ lục tốt, phù hợp văn bản quy chế hành chính có cấu trúc phân cấp rõ, giúp retrieval ổn định trên các tài liệu dài. | Số chunk nhiều hơn các cách cắt thô nên tốn chi phí index/search hơn và vẫn có thể tạo chunk chưa tối ưu khi gặp bảng quá dài. |
| Nguyễn Quang Minh | RecursiveChunker | 9.0/10 | Lấy đúng đoạn văn và giữ nguyên cấu trúc Chương/Điều/Khoản, ưu tiên cắt theo đoạn văn (\n\n) nên context rất đầy đủ và liền mạch. | Một số điều khoản quá dài hoặc không có ngắt đoạn rõ ràng đôi khi vẫn bị cắt ngang ở dấu chấm câu, khiến Agent mất đi chi tiết nhỏ. |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> Chiến lược `RegulationChunker` (chunk theo từng Điều) là tốt nhất cho domain này. Lý do là vì quy chế có tính cấu trúc rất chặt chẽ, việc trích xuất thiếu một Khoản hay một Điểm trong cùng một Điều sẽ dẫn đến việc trả lời sai lệch hoặc thiếu sót nghiêm trọng trong RAG (ví dụ: mất đi điều kiện loại trừ hoặc mức trần/phạt).

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
| 1 | Điều kiện tối thiểu về điểm học tập để đăng ký xét tuyển theo chứng chỉ quốc tế (diện 1.2) là gì? | Thí sinh tốt nghiệp THPT, có điểm trung bình chung (TBC) các môn văn hóa từng năm học lớp 10, 11, 12 đạt từ 8,00 trở lên theo thang điểm 10. |
| 2 | Quy định về mức học bổng tối đa và phương thức hỗ trợ chi phí (vé máy bay, bảo hiểm, visa) cho sinh viên trao đổi nước ngoài là gì? | Học bổng có giá trị tối đa là 30 triệu đồng/ sinh viên. Học bổng được cấp bằng vé máy bay từ Hà Nội đến nơi học tập và ngược lại, gói bảo hiểm du lịch quốc tế, phí thị thực (visa) nhập cảnh nước đến học tập. |
| 3 | Mức điểm hồ sơ năng lực tối thiểu để thí sinh diện 1.3 được tham gia vòng phỏng vấn là bao nhiêu và điểm phỏng vấn tối thiểu cần đạt là bao nhiêu? | Điểm hồ sơ năng lực tối thiểu cần đạt để được tham gia vòng phỏng vấn là 55 điểm; điểm phỏng vấn tối thiểu cần đạt là 10 điểm (thang điểm 20). |
| 4 | Chuẩn đầu ra tiếng Anh đối với sinh viên chương trình đào tạo chuẩn (không thuộc nhóm ngành ngôn ngữ) khóa 70 là bậc mấy? | Đạt chứng chỉ trình độ tối thiểu Bậc 3 theo Khung năng lực ngoại ngữ 6 bậc dùng cho Việt Nam (ví dụ: IELTS 4.0 - 5.0, TOEIC 550 - 620, VSTEP 4.0). |
| 5 | Thời hạn và phương thức chi trả học bổng đối với nghiên cứu sinh tại Đại học Bách khoa Hà Nội | Học bổng được chi trả mỗi năm một lần qua hình thức chuyển khoản, thực hiện vào tháng 12 hằng năm. |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Điều kiện tối thiểu về điểm học tập diện 1.2 | quy-dinh-phuong-thuc-xet-tuyen-tai-nang_chunk_16 (Điều 4. Xét tuyển theo HSNL + PV diện 1.3) (Top-2: chunk_12 (Điều 3. Xét tuyển theo CCQT diện 1.2)) | 0.6715 | Yes | Thí sinh tốt nghiệp THPT, có điểm trung bình chung (TBC) các môn văn hóa từng năm học lớp 10, 11, 12 đạt từ 8,00 trở lên theo thang điểm 10. |
| 2 | Mức học bổng tối đa và phương thức hỗ trợ trao đổi nước ngoài | quy-dinh-cap-hoc-bong-trao-doi-nuoc-ngoai_chunk_8 (Điều 6. Mức học bổng và phương thức hỗ trợ) | 0.8269 | Yes | Mức học bổng trao đổi nước ngoài tối đa là 30 triệu đồng/sinh viên. Học bổng được cấp bằng vé máy bay khứ hồi, gói bảo hiểm du lịch quốc tế và phí thị thực (visa) nhập cảnh nước đến học tập. |
| 3 | Mức điểm HSNL và phỏng vấn diện 1.3 | quy-dinh-phuong-thuc-xet-tuyen-tai-nang_chunk_23 (Điều 4.4. Đánh giá hồ sơ phỏng vấn) | 0.5483 | Yes | Điểm HSNL tối thiểu đạt 55 điểm để được phỏng vấn; điểm phỏng vấn đạt tối thiểu là 10 điểm (thang điểm 20). |
| 4 | Chuẩn đầu ra tiếng Anh khóa 70 chương trình chuẩn | quy-dinh-phan-loai-..._chunk_18 (Phụ lục III - Chuẩn ngoại ngữ đầu ra đối với các CTĐT chuẩn...) | 0.6816 | Yes | Chuẩn ngoại ngữ đầu ra đối với chương trình đào tạo chuẩn tối thiểu đạt Bậc 3 theo KNLNNVN. |
| 5 | Thời hạn và phương thức chi trả học bổng nghiên cứu sinh | quy-dinh-hoc-bong-doi-voi-nghien-cuu-sinh_chunk_10 (Điều 7. Thời hạn và phương thức chi trả) | 0.8218 | Yes | Học bổng đối với nghiên cứu sinh được chi trả mỗi năm một lần qua hình thức chuyển khoản, thực hiện vào tháng 12 hằng năm. |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5 (Đạt tỷ lệ thành công 100% nhờ tối ưu hóa thuật toán chunking theo cấu trúc Điều/Phụ lục/Bảng và áp dụng pre-filtering thông minh với metadata `scope`, `section` và `program_type`).

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> - Tôi học được từ bạn **Nguyễn Hữu Thái Minh** giải pháp tự phát triển một bộ embedding từ khóa siêu nhẹ `HashingTFIDFEmbedder` (TF-IDF L2-normalized) để chạy RAG cực kỳ ổn định trong môi trường offline sandbox khi không có mạng tải các mô hình Transformer lớn. Đồng thời, tôi học được ý tưởng kết hợp Hybrid Search (phối hợp ngữ nghĩa mạng nơ-ron và từ khóa chính xác BM25/TF-IDF) cho các hệ thống cố vấn quy chế học vụ chứa nhiều thuật ngữ đặc thù ("diện 1.2", "khóa 70").
> - Tôi học được từ bạn **Nguyễn Đức Tâm** và **Nguyễn Thế Giáp** cách tối ưu hóa các tham số cho `RecursiveChunker` để đảm bảo tính ổn định và tính mở rộng của hệ thống khi thực hiện retrieval trên các tài liệu hành chính dài, cũng như tầm quan trọng của việc gán nhãn metadata phân loại (`document_type`, `scope`, `section`) thay vì lưu đường dẫn file thô để pre-filter loại bỏ nhiễu chéo cực kỳ nhanh chóng trước khi tính độ tương đồng.
> - Tôi học được từ bạn **Nguyễn Quang Minh** cách tận dụng khả năng phân cắt của `RecursiveChunker` dựa trên các ranh giới tự nhiên của văn bản (`\n\n`) để duy trì tính liền mạch của ngữ cảnh cho các Chương/Điều/Khoản dài mà không phá vỡ cấu trúc nguyên bản của văn bản quy chế.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Tôi thấy một nhóm đã kết hợp chiến lược chia nhỏ của họ với việc trích xuất tự động các từ khóa quan trọng (keyword extraction) và thêm vào metadata. Nhờ đó, việc tìm kiếm các từ đồng nghĩa được cải thiện đáng kể ngay cả khi sử dụng mô hình embedding nhỏ.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Tôi đã trực tiếp thực hiện cải tiến này: Thiết lập hệ thống pre-filtering metadata phân cấp gồm `scope`, `section` (dieu vs. phu_luc) và `program_type` (chuan vs. elitech vs. ngon_ngu). Nhờ kết hợp việc làm sạch dữ liệu nguồn (sửa lỗi thiếu khoảng trắng ngăn cách giữa các Điều trong markdown) và bổ sung bộ lọc, độ chính xác truy xuất RAG đối với các câu hỏi phức tạp về bảng biểu phụ lục đã tăng từ 60% lên 100% (5/5 queries đều trả về chunk liên quan trực tiếp ở Top-1 hoặc Top-2).

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 15 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 5 / 5 |
| **Tổng** | | **100 / 100** |
