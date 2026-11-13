# Báo cáo cá nhân — Day 6

**Học viên:** Đào Trường Vũ · 2A202603721 · L3A 

## Dữ liệu và pipeline

- **Chủ đề, nguồn/quyền, số trang và quy tắc tính trang:** OpenStax *Principles of Data Science* (`openstax-pds`), file `Principles-of-Data-Science-WEB.pdf` tải từ assets.openstax.org ngày 2026-11-13T10:24:19Z, SHA-256 `6b472054…92398`, HTTP Last-Modified 21/04/2026. Giấy phép **CC BY-NC-SA 4.0** (ghi trên trang sách); dùng phi thương mại cho bài học, có ghi nguồn; PDF em sẽ không commit (file 37 MB, có lệnh tải lại). Lấy **Chương 2 – Collecting and Preparing Data** (trang PDF vật lý 67–103) và **Chương 8 – Ethics Throughout the Data Science Cycle** (387–413) = **64 trang**; loại phần Key Terms, Group Project, Critical Thinking, References. Quy tắc: trang có ≥ 500 ký tự sau chuẩn hóa mới tính là trang nội dung → **63 trang nội dung**, 194.764 ký tự (p.413 chỉ có 120 ký tự nên bị loại). `page` = số trang vật lý trong PDF; `printed_page` = số in trên header (= vật lý − 10, đúng ở 62/64 trang; 2 trang mở chương không có header). Chi tiết từng trang: `data/manifest.json`.
- **Chuẩn hóa, lỗi và giới hạn:** PyMuPDF 1.27.2.3 đọc block/line/span. Header/footer bị loại theo vị trí (y < 40 hoặc y > 750 trên trang 792 pt); dòng heading nhận biết bằng font Bold ≥ 10,5 pt; section lấy từ outline (TOC) của PDF và chỉ được gán khi heading trên trang khớp đúng tên TOC (khớp đủ 100%, không có cảnh báo). Chuẩn hóa Unicode NFKC, đổi nháy cong → nháy thẳng, NBSP → space, gộp khoảng trắng, nối dòng trong block, giữ ngắt đoạn `\n\n`. Lỗi phát hiện khi đọc dữ liệu và đã sửa: danh sách đánh số bị tách đôi giữa câu ("subgroups based on ⏎⏎ specified factors") → nối hai đoạn khi đoạn trước không kết thúc bằng dấu câu và đoạn sau bắt đầu bằng chữ thường. Log (`results/ingest_log.json`): 0 trang rỗng, 1 trang ít nội dung. PDF là bản số nên không cần OCR. **Giới hạn:** chữ nằm trong hình (ví dụ Figure 8.2 *GDPR Principles*) không được trích; bảng bị làm phẳng theo từng ô; đoạn văn vắt qua 2 trang bị cắt ở ranh giới trang.
- **Embedding, vector store, lệnh chạy, chi phí:** `BAAI/bge-small-en-v1.5`, revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`, chạy local qua sentence-transformers 5.5.1 (torch 2.7.1, CPU), **384 chiều**, giới hạn 512 token (chunk dài nhất 341 token → không chunk nào bị cắt bớt). Document embed nguyên văn; query thêm tiền tố BGE `"Represent this sentence for searching relevant passages: "`; vector chuẩn hóa L2. Vector store: **Chroma 1.5.9 `PersistentClient`** (thư mục `chroma_db/`, 16 MB), HNSW, metric **cosine** — là tìm kiếm *xấp xỉ*; đã so với tìm kiếm chính xác bằng numpy (cùng filter): **63/63 truy vấn ra cùng tập top-3** (`results/ann_check.json`). Lệnh: `cd starter && .venv\Scripts\python -m src.run all --reset` rồi `... labels`. Chi phí: **0 USD** (chưa tính lần tải model ~130 MB).
- **Metadata, trường lọc bổ sung, nguồn giá trị, lý do:** bắt buộc `document_id`, `source`, `page`; thêm `printed_page` (int, từ header), `chapter` (int) và `chapter_title` (từ `data/corpus_config.json` theo khoảng trang), `section_id`/`section_title` (str, từ TOC của PDF), `char_start`/`char_end` (vị trí chính xác trong text trang), `text_sha1`. **Trường lọc: `chapter`** — hai chương dùng chung nhiều từ vựng (bias, outlier, data validation, sampling, imputation), nên phạm vi chương giúp loại nhiễu khi người dùng cho biết họ đang học chương nào. Không có giá trị nào được đoán; filter chỉ cho phép các trường `chapter`, `section_id`, `document_id`.

## Cấu hình và benchmark

- **A / B / C:**

  | | Chunking | Truy xuất | Collection |
  |---|---|---|---|
  | A | Cửa sổ ký tự cố định **500, overlap 50**, trong từng trang | top-3 cosine, không lọc | `day6_strategy_A` |
  | B | Theo đoạn: heading luôn mở chunk mới, gộp đoạn tới **1000 ký tự**, đoạn quá dài tách theo câu, mảnh < 150 ký tự gộp vào chunk kề (tối đa 1200) | top-3 cosine, không lọc | `day6_strategy_B` |
  | C | **Giống hệt B (dùng lại collection B)** | thêm `where chapter == <giá trị>` **chỉ khi câu hỏi có filter** (5/21 câu) | `day6_strategy_B` |

  Khác biệt duy nhất B–C là filter metadata (cùng chunk, cùng vector). A–B thay cách chunk (kéo theo kích thước chunk khác nhau, xem phần hạn chế).
- **Benchmark (≥ 20 câu):** 21 câu trong `benchmark/questions.json`, chia trước khi chạy **16 dev / 5 test**. Câu test dùng các trang (92, 94, 397, 400, 411) mà không câu dev nào dùng. Loại câu: 7 trực tiếp, 5 diễn đạt lại, 5 điều kiện/ngoại lệ, 4 loại filter. **5 câu có filter `chapter`**: dev q08 (chương 2), q15, q16 (chương 8); test t01 (chương 8), t02 (chương 2). Filter là ngữ cảnh giả định người dùng cung cấp trước khi truy xuất ("tôi đang học chương …"), không lấy từ gold. Kiểm chứng evidence: `verify-benchmark` kiểm tra mọi `evidence_quote` (kể cả evidence thay thế) có nằm nguyên văn trong text đã chuẩn hóa của trang gold → **23/23 đạt**. Benchmark được khóa bằng SHA-256 `40349481fc09…` (`benchmark/verification.json`); `evaluate` từ chối chạy nếu file bị sửa. 
- **Tiêu chí chọn chiến lược (chốt trước khi xem test):** Evidence Hit@3 trên dev cao nhất → hòa thì xét Hit@1 dev → hòa thì chọn cấu hình ít chunk hơn (chi phí).

## Kết quả

| Strategy | Split | Hit@1 (n/N) | Evidence Hit@3 (n/N) | Chunk count | Thời gian | Chi phí |
|---|---|---|---|---|---|---|
| A | dev | 8/16 | 11/16 | 457 | đọc PDF 0,9 s + chunk 0,004 s + embed 13,3 s + upsert 0,41 s | 0 |
| A | test | 1/5 | 1/5 | 457 | như trên | 0 |
| B | dev | 11/16 | 14/16 | 301 | đọc PDF 0,9 s + chunk 0,003 s + embed 13,0 s + upsert 0,33 s | 0 |
| B | test | 4/5 | 4/5 | 301 | như trên | 0 |
| C | dev | 11/16 | 14/16 | 301 (dùng chung B) | không cần index thêm | 0 |
| C | test | 4/5 | 4/5 | 301 | không cần index thêm | 0 |

Truy vấn: 63 lượt search (21 câu × 3 cấu hình) tổng 0,12 s, chưa tính thời gian embed câu hỏi. Độ dài chunk: A trung vị 500 ký tự / 95 token, tổng 42.766 token; B trung vị 692 ký tự / 133 token, tổng 38.189 token. Số liệu thêm (`results/summary.md`): trang gold có trong top-3 — A 21/21, B 20/21; số câu nghi bị **tách evidence qua 2 chunk** — A 6/21, B 0/21.

| Strategy | Coherence trung bình /2 (n) | Metadata đủ và đúng (n/N) | Filter tuân thủ (n/N hoặc N/A) | Grounding 0/1/2 (số câu) và trung bình /2 |
|---|---|---|---|---|
| A | 0,80 (n=10; 0/1/2 = 2/8/0) | 15/15 | dev 2/3, test 2/2 (không áp filter, tuân thủ ngẫu nhiên) | 1/1/3 → 1,40 |
| B | 1,60 (n=10; 0/1/2 = 0/4/6) | 15/15 | dev 2/3, test 1/2 | 1/0/4 → 1,60 |
| C | 1,60 (= B do cùng chunk) | 15/15 | **dev 3/3, test 2/2** | 1/0/4 → 1,60 |

**Đường dẫn trace, nhãn thủ công và cách chọn mẫu:** trace đầy đủ top-3 (câu hỏi, chunk ID, text, score, metadata, nhãn hit) ở `results/traces/{A,B,C}.jsonl`. Bảng chấm ở `results/audit/`: `coherence_labeled.csv`, `grounding_labeled.csv`, `metadata_sheet.csv`; nhãn và lý do ở `manual_labels.json`; tổng hợp ở `results/manual_summary.json`. **Mẫu coherence** được chọn theo quy tắc cố định trước khi chấm: 10 trang nội dung cách đều nhau trong danh sách đã sắp xếp (p67, 74, 81, 88, 95, 101, 391, 398, 405, 412), lấy vị trí giữa text mỗi trang, chấm chunk của A và của B chứa vị trí đó. **Metadata:** 5 câu test × top-3 × 3 cấu hình = 45 bản ghi; mỗi bản ghi kiểm tra hai lần: (1) text chunk khớp đúng đoạn `page.text[char_start:char_end]` và section khớp heading của trang; (2) đối chiếu độc lập với PDF gốc — một cụm 8 từ ở giữa chunk phải xuất hiện trên đúng trang vật lý. Cả 45 đạt. Đặc trưng toàn corpus: **400/457 chunk A (87,5%) bắt đầu giữa câu**, so với 33/301 chunk B (11%); 5 chunk A vắt qua 2 section nhưng chỉ mang section ở điểm bắt đầu (ví dụ `A:openstax-pds:p0398:c003` ghi 8.1 nhưng chứa heading 8.2); B có 0 chunk như vậy.

## Phân tích và kết luận

- **So sánh A–B và B–C; ví dụ sai:**
  - **A → B (đổi cách chunk):** Evidence Hit@3 dev 11 → 14/16, test 1 → 4/5; tách evidence 6 → 0 câu; coherence 0,8 → 1,6; số chunk giảm 34% (457 → 301), thời gian embed gần như nhau. A gần như luôn lấy đúng trang (21/21) nhưng cắt mất bằng chứng — lỗi nằm ở ranh giới chunk, không phải ở embedding. Lưu ý: hai trường hợp A thua (t01, t05) là do so khớp chặt — cửa sổ chỉ cắt mất dấu chấm hoặc vài từ cuối câu. Người chấm vẫn coi đây là bằng chứng đủ (grounding = 2), nên trên test A đạt 3/5 nếu tính bằng chứng tương đương, so với 1/5 tự động.
  - **B → C (thêm filter):** Hit@1/Hit@3 không đổi, nhưng tuân thủ filter tăng từ dev 2/3 → 3/3 và test 1/2 → 2/2. Filter loại được nhiễu từ chương khác, nhưng không sửa được việc xếp hạng trong cùng một chương.
  - **Ví dụ sai 1 — q02 (dev, gold p.73, strategy A):** top-1 `A:openstax-pds:p0073:c002` bắt đầu bằng *"a random sample of these clusters picked…"*. Nửa đầu định nghĩa cluster sampling nằm ở cửa sổ trước, nên đúng trang mà không có chunk nào chứa đủ evidence. B lấy được trọn câu ở top-1.
  - **Ví dụ sai 2 — q08 (dev, gold p.87, filter `chapter=2`, strategy B/C):** B xếp `B:openstax-pds:p0407:c001` (Chương 8, ví dụ trực quan hóa outlier) ở hạng 1, vi phạm phạm vi người dùng. C loại được chunk đó, nhưng chunk chứa mục *"4. Consider the influence of outliers…"* vẫn không vào top-3, vì nhiều đoạn khác trong Chương 2 cũng nói về outlier và cạnh tranh về độ tương đồng.
  - **Ví dụ sai 3 — t03 (test, gold p.397, B/C):** câu hỏi về một mục cụ thể (data engineers) trong danh sách 8 vai trò. Danh sách nằm trong một chunk ~1000 ký tự nên tín hiệu của mục đó bị "pha loãng". Top-3 trả về các đoạn chung về bảo mật, grounding = 0 ở cả ba cấu hình.
  - **Lỗi diễn đạt lại — q04 (dev, gold p.74):** câu hỏi không dùng cụm "sampling error", nên cả B và C đều trượt.
- **Chiến lược chọn, đánh đổi, điều kiện áp dụng:** Theo tiêu chí chốt trước, B và C cùng dẫn đầu dev (Hit@3 14/16, Hit@1 11/16) và **hòa ở cả hai bước phá hòa**, vì C dùng chung index với B nên số chunk bằng nhau. Mình không lường trước trường hợp hòa này, nên tiêu chí chốt trước không phân định được B và C. Để chọn, mình dùng thêm một chỉ số dev **không nằm trong tiêu chí chốt trước**: tuân thủ filter trên dev (C 3/3 so với B 2/3) → **chọn C**. Điều kiện áp dụng: C chỉ có ích khi người dùng cung cấp phạm vi đáng tin; với câu không có filter, C chính là B. Đánh đổi: nếu người dùng chọn sai chương, filter sẽ loại mất evidence đúng. Hai chương có tham chiếu chéo nhau (p.403: *"Data validation (as introduced in Collecting and Preparing Data)"*), nên rủi ro này có thật. So với A, B/C ít chunk hơn, coherence cao hơn, metadata section luôn đúng; đổi lại chunk dài hơn, nên mục nhỏ trong danh sách dài khó được tìm thấy (t03).
- **Hạn chế của dữ liệu, nhãn, test nhỏ, đo thời gian/chi phí:**
  1. Test chỉ có 5 câu, mỗi câu tương đương 20 điểm phần trăm, nên chênh lệch B–A trên test không có ý nghĩa thống kê. Kết luận chỉ đúng với corpus này.
  2. Dev và test được tính trong cùng một lần `evaluate`, tức kết quả test hiện ra cùng lúc với dev. Không có thay đổi cấu hình hay benchmark nào sau lần chạy đó, nhưng quy trình chưa đúng chuẩn "chọn trên dev rồi mới mở test". Lần sau nên chạy dev riêng trước.
  3. Quy tắc gộp mảnh < 150 ký tự của B được thêm sau khi xem thống kê chunk trước khi chạy đánh giá.
  4. Benchmark và nhãn thủ công thì em có sử dụng Claude để hỗ trợ đồng thời xây pipeline chấm → không mù, có nguy cơ thiên lệch.
  5. Nhiễu gây ra bởi kích thước: chunk B dài hơn A (trung vị 692 so với 500 ký tự), nên một phần lợi thế của B có thể đến từ kích thước chứ không chỉ từ việc bám cấu trúc.
  6. Chunk không vượt qua trang, nên câu nối giữa hai trang bị cắt ở cả A và B (ví dụ `B:openstax-pds:p0068:c000` = "business success."). Chữ trong hình không được trích, bảng bị làm phẳng.
  7. Thời gian chỉ đo một lần trên một máy CPU; chưa tính thời gian tải model; chi phí 0 vì chạy local. Corpus và câu hỏi đều bằng tiếng Anh; chưa thử truy vấn tiếng Việt.

## Ghi chú sử dụng AI

Dùng Claude để hỗ trợ viết code pipeline, soạn nháp bộ câu hỏi và đề xuất nhãn chấm tay
