# Quy tắc AI agent — cfg5_mophong

## Phạm vi và điểm bắt đầu

- Viết trao đổi, hướng dẫn và báo cáo bằng tiếng Việt; giữ thuật ngữ kỹ thuật khi cần.
- Đọc `README.md`, kiểm tra Git status, rồi chỉ khảo sát các thư mục liên quan đến yêu cầu.
- Code/config/tài liệu làm việc chính nằm trong `local_source/`. Khi sửa ở đó, đọc thêm `local_source/AGENTS.md`, `local_source/references/requests/PROJECT_INSTRUCTIONS_VI.md` và `local_source/docs/DECISIONS.md`.
- Các báo cáo cũ ghi trạng thái ở thời điểm lập báo cáo. Xác minh code, config và kết quả thực tế trước khi mô tả trạng thái hiện tại.
- Nếu có ở local, đọc thêm `BAN_GIAO_CODEX_CFG5_MCR_2026-10-13.md` khi tiếp tục pipeline MCR. Bản đầy đủ hiện chưa được công bố lên GitHub vì chứa thông tin vận hành máy chủ. Đây là bản bàn giao từ cuộc trao đổi nguồn, không phải bằng chứng mọi lệnh trong đó đã chạy; mục GitHub của bản bàn giao có trước đợt đồng bộ hiện tại. Nó ghi source đang chỉnh sửa ở `local_source_(1)/`, vẫn giữ ngoài phạm vi đợt này.
- `github_source/` và các thư mục `*_bundle*` là snapshot hoặc gói triển khai; không ghi đè code chính bằng một phiên bản cũ nếu chưa đối chiếu.
- `server_pretrain_current.py` và `server_pretrain_masked_verified.py` là snapshot trainer để đối chiếu, không phải entrypoint chạy trực tiếp tại root.

## Dữ liệu, training và kiểm chứng

- Không sửa, di chuyển hoặc xóa raw dataset, cache, checkpoint hay payload đang dùng nếu yêu cầu hiện tại không cho phép.
- Không tự chạy full preprocessing, training dài, rollout dài hoặc tải asset lớn. Người dùng tự chạy các thí nghiệm dài; agent chuẩn bị code, cấu hình và kiểm tra nhỏ phù hợp.
- Phân biệt future measured-state targets của CFG5 với controller command targets của dữ liệu task đích.
- Split theo episode/group trước khi tạo action chunks; fit normalization chỉ trên train. Không dùng test để chọn checkpoint.
- Không đồng nhất offline validation, scripted controller demo và learned-policy rollout. Chỉ báo chức năng đã chạy khi có bằng chứng kiểm chứng.
- Không sửa source upstream trong `third_party/` hoặc `references/upstream/` để chạy thử; dùng adapter, wrapper hoặc patch có ghi nhận.

## GitHub và rule được công bố

- Theo yêu cầu trực tiếp ngày 2026-10-13, công bố cả `AGENTS.md` và `PROJECT_INSTRUCTIONS_VI.md`. Quyết định này thay thế hướng dẫn lịch sử chỉ xuất code công khai và bỏ rule cá nhân.
- Không đọc, kiểm kê, stage hoặc đồng bộ `local_source_(1)/` (thư mục người dùng gọi là `local_source(1)`) trong đợt kiểm tra này nếu không có yêu cầu mới.
- Không đưa secrets, `.env`, môi trường ảo, `.git` lồng, raw data, video hoặc cache payload vào commit. Các weights đã được chọn trong `.gitignore` dùng Git LFS; không tự thêm weights khác.
- `checkpoint_upload_repo/` và `github_upload_source1/` là clone repo cũ, chỉ dùng đối chiếu. Rule gốc của repo edabk nằm trong `docs/reference_agent_rules/`, là tư liệu lịch sử và không áp dụng quy trình nhánh/issue của repo đó cho repo này.
- Khi đồng bộ, chọn rõ file/phạm vi, kiểm diff staged và xác nhận remote. Không reset thay đổi của người dùng hoặc force-push.
- Bản HTML của `ai-flow-site/` được GitHub lưu như file thường. Giữ nguyên repository Sites cục bộ và không xuất metadata Git của nó.

Hướng dẫn trực tiếp mới nhất của người dùng có ưu tiên cao hơn những quyết định lịch sử trong tài liệu dự án.
