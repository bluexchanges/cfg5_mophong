# Đưa bản mã nguồn sạch lên GitHub

Giải nén `github_source.zip` vào một thư mục mới. Bản này đã loại raw CFG5, processed NPY, môi trường ảo, output lớn và các quy tắc làm việc riêng; không dùng thư mục `zip/` hoặc `simulation.zip` làm repository public.

Mở terminal **trong thư mục vừa giải nén**. Kiểm `README.md`, `data/processed/README.md`, `data/splits/` và `data/index/`; `data/processed/` không được có thư mục episode. Nếu bạn đã tạo repository trống trên GitHub, chạy:

```bash
git init -b main
git add .
git status --short
git commit -m "Initial RH20T ACT project source"
git remote add origin https://github.com/OWNER/REPO.git
git push -u origin main
```

Thay `OWNER/REPO` bằng repository của nhóm. Trước khi commit, kiểm `git status --short`: không được thấy `RH20T_cfg5/`, `data/processed/cfg5_all/`, `.venv*`, PDF cá nhân hoặc file checkpoint. GitHub chỉ mang code và metadata nhỏ. Dữ liệu raw được cung cấp cho server riêng; xem [hướng dẫn preprocess trên server](SERVER_REPROCESS_VI.md).

Giữ các tài liệu điều phối riêng ngoài repository public; đồng đội chỉ cần code, config và tài liệu kỹ thuật. Các quyết định triển khai liên quan đến dự án vẫn được giữ trong `docs/`.
