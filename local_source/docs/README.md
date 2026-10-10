# Bản đồ tài liệu dự án

**Cập nhật 2026-09-30:** Đọc [hướng dẫn pretraining](PRETRAINING_VI.md) và [chuẩn bị data server](SERVER_REPROCESS_VI.md). Trainer đã được thêm nhưng chưa chạy full ACT trên CFG5 GPU; các báo cáo cũ là bằng chứng lịch sử từ máy A.

Đọc theo nhu cầu dưới đây. Tài liệu lượt1/2/3 phản ánh thời điểm khảo sát; quyết định hiện hành và bản sửa placement có ưu tiên khi nội dung khác nhau. Trạng thái hiện tại: đã có dữ liệu adapter và mô phỏng/controller; đã tích hợp/kiểm ACT inference và backward nhỏ, chưa train ACT.

| File | Mô tả | Khi nào cần đọc? |
|---|---|---|
| [DECISIONS.md](DECISIONS.md) | Quyết định implementation ACT,8D action, subset CFG5,8 camera, split, task đích và phân kỳ. | Đọc đầu tiên trước khi sửa thiết kế. Các đoạn “kết quả lượt2/3” là lịch sử tại lượt đó. |
| [CFG5_FULL_FILTER_VI.md](CFG5_FULL_FILTER_VI.md) | Phân loại toàn104 task, move không copy, tiêu chí và chỉ mục human–robot; phân biệt cặp theo tên với nhãn ACT. | Đọc trước khi dùng dữ liệu đã tổ chức hoặc triển khai pretraining full. |
| [CFG5_FULL_SPLIT_AND_RUN_VI.md](CFG5_FULL_SPLIT_AND_RUN_VI.md) | Split ngẫu nhiên đúng531/66/66, lệnh tự chuyển đổi train/val, dung lượng, cơ sở10 Hz và giới hạn đồng bộ camera–robot. | Đọc trước khi tự chạy preprocessing. |
| [CFG5_VSCODE_SMOKE_VI.md](CFG5_VSCODE_SMOKE_VI.md) | Từ mở VS Code/chọn Python đến chạy1–2 smoke, xem8 ảnh, resume và bấm giờ full. | Đọc đầu tiên nếu tự thao tác trên máy. |
| [PIPELINE_GUIDE_VI.md](PIPELINE_GUIDE_VI.md) | Lệnh PowerShell và thứ tự manifest/split/preprocess/stats/check; phân biệt smoke với full. | Khi tự chuẩn bị dữ liệu hoặc kiểm môi trường. |
| [DATA_GUIDE_VI.md](DATA_GUIDE_VI.md) | Schema CFG5, trường dữ liệu, timestamp/frame/đơn vị, causal sampling và action chunks. | Trước khi sửa adapter hoặc hiểu source target. |
| [CAMERA_SETUP_VI.md](CAMERA_SETUP_VI.md) | Calibration8 camera,6 fixed+2 wrist, frame transforms và phép kiểm. | Khi thay góc nhìn, parent camera, TCP hoặc K. |
| [MUJOCO_PANDA_GUIDE_VI.md](MUJOCO_PANDA_GUIDE_VI.md) | Cài/chạy MuJoCo/Panda, API, viewer/headless, controller, evaluator và lỗi thường gặp. | Khi chạy mô phỏng hoặc viết chương trình dùng PandaEnv. |
| [PLACEMENT_LOGIC_REVIEW_VI.md](PLACEMENT_LOGIC_REVIEW_VI.md) | Các lỗi placement đã sửa, target độ nhô, support/contact, timer và kiểm chứng mới. | Khi hiểu thế nào là đặt đúng; đọc cùng configs/mujoco/README. |
| [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) | Kế hoạch so sánh target-only với CFG5 pretrain→fine-tune; split/checkpoint/metric và giới hạn. | Trước khi tổ chức training và đánh giá chính thức; không phải báo cáo đã train. |
| [PAPER_ANALYSIS.md](PAPER_ANALYSIS.md) | Phân tích hai PDF RH20T/ACT, bài toán, phương pháp và khác biệt reproduction. | Khi hiểu cơ sở nghiên cứu hoặc viết báo cáo. |
| [ACT_IMPLEMENTATION_REVIEW.md](ACT_IMPLEMENTATION_REVIEW.md) | So sánh ACT gốc, AirExo, RH20T baseline; dependencies/license và sampling/validation/padding. | Khi tích hợp model; quyết định cuối cùng tra DECISIONS. |
| [PROJECT_AUDIT.md](PROJECT_AUDIT.md) | Kiểm kê máy/dữ liệu/source ở lượt1, các giả định và giới hạn khi đó. | Tra xuất phát điểm; không dùng trạng thái môi trường cũ thay hiện tại. |
| [STAGE2_REPORT_VI.md](STAGE2_REPORT_VI.md) | Kết quả preprocessing/split/adapter/camera số học lượt2. | Tra số liệu subset và smoke; khi đó chưa render MuJoCo. |
| [STAGE3_REPORT_VI.md](STAGE3_REPORT_VI.md) | Snapshot mô phỏng lượt3 trước sửa placement, video và tests tại thời điểm đó. | Tra lịch sử; kết quả evaluator mới nằm ở PLACEMENT_LOGIC_REVIEW. |
| [audit/](audit/README.md) | JSON/log nhỏ ghi bằng chứng thực chạy, counts và hashes. | Đối chiếu lời mô tả với kết quả đo. |

Lộ trình dễ đọc: DECISIONS → DATA_GUIDE/PIPELINE → CAMERA_SETUP → MUJOCO_PANDA_GUIDE → PLACEMENT_LOGIC_REVIEW → EXPERIMENT_PROTOCOL. Paper và source gốc ở [references](../references/README.md), không đặt bản sao rải rác trong docs.

Khi cập nhật, phân biệt “đã triển khai”, “đã kiểm thử” và “dự kiến”. Report cũ giữ làm lịch sử, không sửa số liệu để trông như đã chạy code mới. Config hash, số episode/sample thực và `act_model_used` giúp xác định phạm vi kết luận.

Tài liệu thực hành: [ACT_SETUP_VI.md](ACT_SETUP_VI.md) giải thích source/model/weights và kiểm chứng tích hợp; [CFG5_HANDS_ON_VI.md](CFG5_HANDS_ON_VI.md) hướng dẫn từng lệnh chọn episode, chuyển đổi, lưu/mở NPY và trình bày.
