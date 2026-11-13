# Day 06 · Data Ingestion Engineering

**Lab cá nhân** · **4 giờ** · **Dữ liệu tự chọn**

Biến tài liệu thành kho tri thức có thể truy xuất cho AI agent, rồi đo xem cách chuẩn bị dữ liệu nào hiệu quả nhất.

| 📄 Dữ liệu | 🧩 Chiến lược | 🎯 Benchmark |
|---|---|---|
| Tối thiểu **30 trang nội dung** | Tối thiểu **3 cấu hình** | Tối thiểu **20 câu hỏi** |

## Bài làm — Đào Trường Vũ (2A202603721)

Corpus: OpenStax *Principles of Data Science*, Chương 2 + 8 (63 trang nội dung). Embedding local `BAAI/bge-small-en-v1.5` (384 chiều, $0). Vector store: **Chroma 1.5.9** lưu trên đĩa (`chroma_db/`), metric cosine. Báo cáo: [report/individual/2A202603721_DaoTruongVu.md](report/individual/2A202603721_DaoTruongVu.md).

| Thư mục / file | Nội dung |
|---|---|
| `starter/src/` | Pipeline: `ingestion.py`, `chunking.py`, `embeddings.py`, `store.py` (Chroma), `evaluation.py`, `run.py` |
| `data/corpus_config.json`, `data/manifest.json` | Nguồn, giấy phép, SHA-256, danh sách trang + số ký tự |
| `benchmark/questions.json`, `benchmark/verification.json` | 21 câu (16 dev / 5 test), evidence đã kiểm chứng, hash khóa benchmark |
| `artifacts/experiment_log.csv` | Cấu hình + giả thuyết A/B/C và Hit@k |
| `results/` | `pages.jsonl`, `chunks_*.jsonl`, `build_stats.json`, `traces/*.jsonl`, `summary.md`, `ann_check.json`, `reingest_check.json`, `audit/` (nhãn thủ công), `manual_summary.json` |

Tái lập (Windows PowerShell; macOS/Linux đổi đường dẫn venv):

```bash
curl -L -o data/raw/Principles-of-Data-Science-WEB.pdf https://assets.openstax.org/oscms-prodcms/media/documents/Principles-of-Data-Science-WEB.pdf
cd starter
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m src.run all --reset
.venv\Scripts\python -m src.run labels
```

`ingest` dừng nếu SHA-256 của PDF khác `data/corpus_config.json`; `evaluate` dừng nếu `benchmark/questions.json` khác bản đã khóa. Lần chạy đầu tải model (~130 MB) từ Hugging Face.

## Bắt đầu nhanh

Dùng [starter PostgreSQL + pgvector](starter/README.md) để dựng database và hoàn thiện các hàm TODO, hoặc tự chọn vector store tương đương. Starter chỉ là khung: **bạn tự hiện thực pipeline và benchmark**.

## 1. Xây pipeline

![Đọc tài liệu, chuẩn hóa, chia chunk, tạo vector, lưu và kiểm tra](assets/ingestion-pipeline.svg)

Chọn tài liệu được phép sử dụng; giữ **ID tài liệu, nguồn, trang/vị trí** và metadata phù hợp. Dùng embedding model thật qua API hoặc chạy local; lưu và truy xuất bằng pgvector, Qdrant, Chroma hoặc một store tương đương.

## 2. So sánh và đánh giá

![Ba chiến lược được đánh giá trên cùng corpus và benchmark dev/test](assets/benchmark-workflow.svg)

Dùng cùng dữ liệu, model và câu hỏi cho cả ba cấu hình. Tách **15 câu dev / 5 câu test** trước khi chạy; chọn cấu hình trên dev, rồi đánh giá test sau khi cố định. Mỗi câu hỏi cần có đoạn bằng chứng và trang nguồn; thiết kế bộ câu hỏi và đánh giá theo [BENCHMARK.md](BENCHMARK.md).

Đánh giá theo năm khía cạnh: **độ chính xác truy xuất**, **tính mạch lạc của chunk**, **tính hữu dụng của metadata**, **chất lượng grounding** và **tác động của chiến lược dữ liệu**. Giải thích lựa chọn bằng kết quả, ví dụ lỗi và các đánh đổi.

## 3. Chuẩn bị bài nộp

1. Chạy pipeline và lưu cấu hình, kết quả top-3, trace cùng nhãn đánh giá thủ công.
2. Điền [thông tin học viên](STUDENT.md), [nhật ký thí nghiệm](artifacts/experiment_log.csv) và bản sao [mẫu báo cáo cá nhân](report/individual/MSSV_HoTen.md).
3. Kiểm tra mã nguồn, cách chạy, manifest dữ liệu và checklist trong [SUBMISSION.md](SUBMISSION.md).

## Tài liệu hướng dẫn

| Bạn cần tìm | Tài liệu |
|---|---|
| Setup baseline PostgreSQL + pgvector | [starter/README.md](starter/README.md) |
| Các bước làm và cách tính chỉ số | [GUIDE.md](GUIDE.md) |
| Tiến độ trong 4 giờ | [CHECKPOINTS.md](CHECKPOINTS.md) |
| Năm tiêu chí benchmark và bộ câu hỏi | [BENCHMARK.md](BENCHMARK.md) |
| Tiêu chí chấm điểm | [RUBRIC.md](RUBRIC.md) |
| Quy tắc sử dụng dữ liệu, AI và API | [RULES.md](RULES.md) |
| Cách nộp bài | [SUBMISSION.md](SUBMISSION.md) |

[data/format-example.json](data/format-example.json) chỉ minh họa định dạng, không phải corpus đủ điều kiện. Phần xử lý và benchmark do bạn tự hiện thực; không bắt buộc xây chatbot.
