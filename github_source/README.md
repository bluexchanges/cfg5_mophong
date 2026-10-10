# RH20T CFG5 → ACT → Panda trong MuJoCo

**Cập nhật 2026-09-30:** Đã thêm trainer CFG5, validation, checkpoint/resume; test đã chạy optimizer step với ACT thu nhỏ. Chưa chạy optimizer step của full ACT trên CFG5 GPU, chưa có checkpoint đã học. Xem [hướng dẫn pretraining](docs/PRETRAINING_VI.md).

Dự án dùng robot demonstrations của RH20T CFG5 để pretrain ACT, rồi hướng tới fine-tune và đánh giá Panda gắp, đặt ba chốt Ø6 mm vào ba lỗ Ø7 mm trong MuJoCo. Mô hình nhận 8 camera RGB và trạng thái TCP/gripper 8D. Tài liệu kỹ thuật bằng tiếng Việt nằm trong [docs/](docs/README.md).

## Trạng thái hiện tại

- Đã có pipeline lọc/split/preprocess CFG5, ACT adapter, mô phỏng Panda, controller và evaluator. Split cố định theo episode/group: **531 train / 66 validation / 66 test**.
- Ở máy phát triển, 531 train và 66 validation episode từng được preprocess và kiểm cấu trúc; báo cáo lịch sử ở [docs/audit/](docs/audit/README.md). **Bản GitHub này không chứa raw CFG5, dữ liệu processed, môi trường ảo, checkpoint hay video.**
- Trainer ACT của dự án đã có và qua test fixture/ACT thu nhỏ; cần preprocess lại train+val trên server rồi smoke full model/GPU trước khi người dùng chạy pretraining dài. Chưa pretrain và chưa chạy ACT rollout. Controller demo trong MuJoCo không phải kết quả ACT.

## Cấu trúc để làm việc nhóm

| Đường dẫn | Vai trò |
|---|---|
| [configs/](configs/README.md) | Cấu hình dữ liệu 8 camera, ACT và bài toán đặt chốt. |
| [data/splits/](data/README.md) | Manifest, split đã khóa và thống kê train-only của lần preprocess trước để đối chiếu. |
| [data/index/](data/README.md) | Chỉ mục human–robot và bảng episode thuộc train/validation. |
| [data/processed/](data/processed/README.md) | Ban đầu trống; output preprocess được tạo tại đây và bị Git bỏ qua. |
| [src/](src/README.md), [scripts/](scripts/README.md), [tests/](tests/README.md) | Adapter, trainer CFG5, lệnh chạy và kiểm thử. |
| [third_party/](third_party/README.md) | Source RH20T ACT baseline tại commit cố định, giữ license. |
| [mujoco/](mujoco/README.md) | Scene/config Panda; mesh lớn tải lại bằng script. |
| [references/](references/README.md) | Nguồn, manifest và ghi chú; PDF không nằm trong bản public. |

## Chuẩn bị server

Nếu bạn chuẩn bị repository cho nhóm, làm theo [hướng dẫn đưa bản sạch lên GitHub](docs/GITHUB_PUSH_VI.md).

Git clone chỉ lấy mã nguồn. Cần cung cấp raw RH20T CFG5 **riêng**, đặt vào RH20T_cfg5/ theo layout được [src/data/common.py](src/data/common.py) hỗ trợ. Đọc [hướng dẫn preprocess trên server](docs/SERVER_REPROCESS_VI.md) trước khi chạy: fingerprint nguồn hiện có chứa mtime của file, nên máy khác có thể cần tạo manifest/split mới và đối chiếu membership với split đã khóa. Preprocess train và validation trên server; giữ test cho giai đoạn sau. Sau đó làm theo [pretraining](docs/PRETRAINING_VI.md).

Các bản lock hiện được kiểm trên **Windows CPU**. Máy chủ GPU cần cài PyTorch/CUDA phù hợp phần cứng của nó; không dùng bản +cpu để train. Trainer đã qua test fixture và ACT thu nhỏ; lệnh `scripts/train.py --smoke` là bước xác minh full model sau khi server có processed data.

Giấy phép của dataset, paper, RH20T ACT và model Panda là riêng biệt; xem [references/SOURCES.md](references/SOURCES.md) và license đi kèm source upstream.
