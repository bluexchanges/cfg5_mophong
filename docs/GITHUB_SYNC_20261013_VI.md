# Đối chiếu và bổ sung GitHub — 2026-10-13

## Phạm vi

Đối chiếu repository `https://github.com/bluexchanges/cfg5_mophong` trên nhánh `main` với workspace `F:\cfg5_mophong`. Bỏ qua `local_source_(1)/` theo yêu cầu trực tiếp của người dùng; đây là tên thư mục trên đĩa tương ứng với `local_source(1)` trong yêu cầu.

Remote đã được fetch. Trước khi bổ sung, `origin/main` và local HEAD cùng ở commit `36ac586f537fd766add3c020bd4fa3177cc3a870` (`Initial commit: CFG5 simulation and MCR ACT pipeline`). Những phần thiếu dưới đây chưa được track hoặc đang bị `.gitignore` bỏ qua, không phải xung đột với một phiên bản remote mới hơn.

## Những phần đã bổ sung

| Nhóm | Nội dung |
| --- | --- |
| Bundle MCR mới | 51 file trong `cfg5_masked_mcr_bundle/`, `cfg5_masked_mcr_bundle_v4/`, `cfg5_masked_mcr_bundle_v5/`, `cfg5_mcr_continue_bundle/`, `cfg5_mcr_three_splits_dropout_bundle/`; giữ nguyên installer, manifest, source, Slurm jobs, docs và tests. |
| Trainer server | `server_pretrain_current.py`, `server_pretrain_masked_verified.py`; giữ như snapshot để đối chiếu, không ghi đè trainer đang có trong `local_source/`. |
| Công cụ và báo cáo camera | `import_missing_cfg5_drive.sbatch`, `cfg5_robot_all_camera_report_local.html`. |
| Rule AI agent | `AGENTS.md`, `CLAUDE.md` ở root; bản gốc `local_source/AGENTS.md` và `local_source/references/requests/PROJECT_INSTRUCTIONS_VI.md`; tư liệu rule edabk trong `docs/reference_agent_rules/`. |
| Trang HTML giải thích AI | `ai-flow-site/dist/index.html`, `ai-flow-site/.openai/hosting.json`, được lưu như file thường. Gitlink cũ không có `.gitmodules` đã được thay thế; repository Sites cục bộ vẫn giữ nguyên. |
| Báo cáo/preview nhỏ | 40 file trong `local_source/outputs/`: README, JSON/CSV, HTML review và ảnh preview/mô phỏng. Numeric arrays và video không được đưa vào commit. |
| Baseline snapshot | 10 file source/docs/Slurm trong `outputs/mcr_dropout_upgrade_base/`, giữ riêng để đối chiếu nâng cấp dropout; không coi đây là run training mới. |
| Điều hướng và audit | README ở root và [audit JSON](audit/github_sync_20261013.json) với danh sách đường dẫn, dung lượng và SHA-256 của những file chuẩn bị đồng bộ. |

Không thay đổi logic model/trainer, không chạy preprocessing, training hoặc rollout. Những bài test trong bundle được công bố nguyên bản; lần đồng bộ này không tuyên bố đã chạy lại toàn bộ test runtime/GPU.

## Những phần đã kiểm tra nhưng tiếp tục không xuất bản

- `checkpoint_upload_repo/`, `github_upload_source1/`: hai clone repo edabk cũ, tổng 517 file tài liệu/source trong inventory chọn lọc. Các file CFG5 trong đó được đối chiếu theo nội dung với `local_source/`; nhiều file trùng, phần còn lại là snapshot/metadata lịch sử. Không trộn các baseline cũ vào source hiện tại. Các rule `CLAUDE.md` và `CODE-CONVENTION.md` được giữ riêng dưới `docs/reference_agent_rules/`.
- `.tmp_plot_check/`: 8 file kiểm thử đồ thị loss, dùng history mẫu 3 epoch; không phải biểu đồ validation của run thật.
- `local_source/references/extracted/`: 2 bản text trích xuất có thể tái tạo từ PDF gốc đã có trong repository.
- `.git` lồng, môi trường ảo, cache Python, raw dataset, numeric payload, video, các archive bundle và weights chưa được chọn: giữ ngoài source export. Các weights đã có trong Git LFS không bị thay đổi hoặc bổ sung trong đợt này.

Đây là những phần loại theo vai trò artifact/snapshot và chính sách dữ liệu, không phải những source/config mới của pipeline bị bỏ sót. Inventory chọn lọc không có source/báo cáo còn thiếu ngoài các nhóm loại đã nêu. Không kiểm tra nội dung `local_source_(1)/`.

## Kiểm tra trước push

- 32 file Python bổ sung được parse AST thành công, không import Torch hoặc khởi chạy job.
- 31 file JSON có sẵn trong phần bổ sung được parse thành công; audit JSON được tạo sau đó.
- 14 script Bash/Slurm được kiểm tra cú pháp bằng `bash -n`; link file trong các README mới đều trỏ đến file đã có.
- `git diff --cached --check` đạt sau chuẩn hóa line endings trong Git index; các báo cáo trên đĩa không bị ghi lại.
- Không có path thuộc thư mục người dùng loại, môi trường ảo, `.git` lồng, payload numeric, video hoặc weights mới trong danh sách staged.
- Kiểm tra các mẫu token/private-key phổ biến trên nội dung bổ sung không có kết quả khớp. Đây là kiểm tra mẫu, không phải chứng nhận bảo mật toàn repository.
- HTML được lưu dưới mode `100644`, thay cho gitlink `160000` trước đây. Không cần lấy submodule để đọc trang sau clone.

SHA-256 trong audit là nội dung file trên workspace tại lúc kiểm tra; Git có thể chuẩn hóa CRLF thành LF khi lưu text. Danh sách audit được lập trước khi thêm chính hai file báo cáo audit để tránh checksum tự tham chiếu.

## Bổ sung khi kiểm tra cuối

Sau lần push đầu (`5e46a6c`), xuất hiện thêm `BAN_GIAO_CODEX_CFG5_MCR_2026-10-13.md` và bản `.txt` ở root. Hai file có nội dung giống nhau. Lần commit/push bổ sung hai file này bị duyệt tự động từ chối vì bản bàn giao chứa địa chỉ, tài khoản và thông tin vận hành máy chủ; công bố bản đầy đủ cần xác nhận riêng. Hai file còn ở local, chưa stage/commit/push, không bị chỉnh sửa. Audit JSON ghi tên, dung lượng và hash để xác định chính xác bản đang chờ; không sao chép nội dung nhạy cảm vào audit.

Bản bàn giao mô tả pipeline joint + gripper, MCR frozen cache và các quyết định mới từ cuộc trao đổi nguồn. Source đang chỉnh sửa được ghi là `local_source_(1)/`; vẫn không kiểm tra hoặc đồng bộ thư mục đó. Các nhận định trạng thái GitHub trong bản bàn giao thuộc mốc trước lần push này; báo cáo đồng bộ hiện tại là nguồn xác nhận commit đã đẩy.
