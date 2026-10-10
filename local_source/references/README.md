# Tài liệu tham khảo

Bạn có thể bỏ paper mới vào `papers/`, ghi chú vào `notes/`. Giữ tên file dễ nhận biết và bổ sung nguồn vào `SOURCES.md`.

| Thư mục/file | Nội dung |
|---|---|
| `papers/` | Hai PDF bạn cung cấp, được **sao chép**, không di chuyển bản gốc trong `D:\Download` |
| `papers/manifest.json` | Đường dẫn gốc, SHA-256 và số trang của mỗi PDF |
| `notes/` | Chỗ dành cho ghi chú đọc paper của bạn |
| `requests/PROJECT_INSTRUCTIONS_VI.md` | Bản yêu cầu dự án để tra cứu phạm vi từng lượt |
| `upstream/` | Những file source đã đọc, giữ nguyên nội dung; mỗi repo có `TREE.json` |
| `upstream/manifest.json` | Commit và SHA-256 của từng file source đã lưu |
| `extracted/` | Text trích từ PDF, có dấu trang; là file phụ, được bỏ qua trong Git |
| `SOURCES.md` | Danh mục nguồn và cách dùng |

Các file source ở đây là tư liệu kiểm chứng, không phải bản cài ACT hoàn chỉnh. Không import từ đây để training. Hai repo ACT ban đầu và API chỉ lưu các file liên quan; `TREE.json` mô tả cây repo nhưng không có nghĩa mọi file đã tải.

Không áp dụng MIT của code cho PDF hoặc dataset. Các bản PDF được lưu để phục vụ dự án học tập; giữ attribution khi viết báo cáo. License riêng của từng repository đi kèm snapshot.

Kiểm tra bản sao PDF bằng PowerShell tại project root:

```powershell
Get-FileHash 'references/papers/2304.13705v1.pdf'
Get-FileHash 'references/papers/RH20T_paper_compressed (1).pdf'
```

So sánh SHA-256 với `papers/manifest.json`.

## Đọc từng nhóm file để làm gì?

| File/nhánh | Logic sử dụng và điểm cần nhớ |
|---|---|
| `papers/2304.13705v1.pdf` | Paper ACT/ALOHA,18 trang, bản v1 người dùng cung cấp. Đọc kiến trúc/action chunk trong ngữ cảnh bài toán gốc, không coi robot bimanual của paper là cấu hình Panda hiện tại. |
| `papers/RH20T_paper_compressed (1).pdf` | Paper RH20T,15 trang; căn cứ dataset, cấu hình thu và thí nghiệm. CFG5 chỉ là một cấu hình/subset của dataset. |
| `extracted/2304.13705v1.txt`, `extracted/RH20T_paper_compressed (1).txt` | Text có dấu trang để tìm kiếm nhanh. Công thức/bảng/hình có thể mất bố cục; đối chiếu PDF khi cần hiểu chính xác. |
| `notes/README.md` | Hướng dẫn chỗ đặt ghi chú. Ghi rõ paper/trang, điều tác giả nói và suy luận hoặc quyết định riêng của dự án. |
| `requests/PROJECT_INSTRUCTIONS_VI.md` | Yêu cầu dự án ban đầu và phân kỳ. Đọc cùng quyết định mới nhất; ví dụ task hộp ban đầu đã được thay bằng3 chốt theo yêu cầu sau. |
| `upstream/tonyzhaozh__act/` | Snapshot ACT gốc: policy/model, `imitate_episodes.py`, `utils.py` và DETR để khảo sát architecture/training/sampling. |
| `upstream/AirExo__act-in-the-wild/` | Snapshot implementation AirExo: `wholearm.py` về dữ liệu/sampling, train/eval, policy/model để so sánh. |
| `upstream/rh20t__act_baseline/` | Nguồn lõi ACT đã chọn: `dataset/rh20t.py`, `dataset/utils.py`, `models/policy.py`, `models/detr/`, train/requirements/command. Snapshot này vẫn chỉ để khảo sát; source chạy hiện ở third_party/rh20t_act và adapter src/models/rh20t_act.py. |
| `upstream/RH20T__rh20t_api/` | Config robot và code scene/configurations/online dùng để đối chiếu schema, alignment, camera và gripper. |

Trong các snapshot, `TREE.json` là danh mục cây repository tại commit; `LICENSE` là license nguồn; README là hướng dẫn upstream; các file `detr.egg-info` nếu có là metadata snapshot, không chứng minh package đã cài. Danh sách chính xác file đã tải và hash nằm trong `upstream/manifest.json`. Giữ README/source bên trong các repo này nguyên vẹn.

Luồng khảo sát: PDF/source gốc → ghi chú/đối chiếu → [phân tích trong docs](../docs/README.md) → quyết định triển khai → adapter riêng trong src. Không tự thực thi lệnh trong paper/README upstream chỉ vì chúng tồn tại ở thư mục tham khảo.

Khi thêm tài liệu: đặt PDF vào `papers/`, bổ sung nguồn vào `SOURCES.md` và metadata/hash vào manifest nếu cần theo dõi; ghi chú riêng trong `notes/`. Không đặt dataset, checkpoint hoặc môi trường Python ở đây. Model Panda dùng riêng trong [mujoco](../mujoco/README.md), không trộn vào snapshot ACT.
