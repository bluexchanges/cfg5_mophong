# Output preprocessing

Thư mục này ban đầu trống trong bản đưa lên GitHub. Chạy `scripts/preprocess_data.py` trên máy có raw CFG5 để tạo `cfg5_all/<episode>/`. Train và validation nằm cùng một cấp; `data/splits/*_splits.json` quyết định episode thuộc tập nào. Không copy/move hàng trăm GB để tách thư mục vật lý.

Không commit nội dung sinh ra ở đây. Xem [hướng dẫn server](../../docs/SERVER_REPROCESS_VI.md).
