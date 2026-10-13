# CLAUDE.md — edabk-physical-ai-robot

Hướng dẫn cho Claude / AI agent làm việc trong repo này.

## Đọc theo thứ tự
1. `README.md` §1 (cấu trúc) và §5 (chính sách dữ liệu).
2. `CODE-CONVENTION.md` — phần liên quan đến yêu cầu.
3. `01-Plan/plan/project-plan.md` khi yêu cầu liên quan tiến độ, chỉ tiêu, deliverable.
Chỉ đọc thêm thư mục cần cho yêu cầu; không đọc `_private/`, `_archive/` trừ khi được yêu cầu rõ.

## Quy tắc
- Quy trình GitHub (nhánh thành viên, task trong Project, commit gắn task, bàn giao): mục "Quy trình làm việc trên GitHub" bên dưới và README §6. Không commit trực tiếp lên `main`; commit sau khi `tools/check-repo.sh --staged` báo 0 FAIL.
- Thí nghiệm chạy bằng script tái lập được, không thao tác tay không ghi lại: một lệnh từ repo sạch, tham số trong file cấu hình, log ghi config, seed, commit hash (CODE-CONVENTION §13). Chưa có script thì viết và commit script trước khi chạy.
- Mỗi thí nghiệm phải đáp ứng `05-Experimental/README.md` mục "Yêu cầu bắt buộc cho mỗi thí nghiệm": tải code/dữ liệu công khai có ghim phiên bản, một lệnh chạy toàn bộ flow, kiểm tra chức năng PASS/FAIL tự động so với golden, tự ghi accuracy/performance/power ra CSV/JSON; README của mỗi `flow-*/` có đủ 6 mục "Tài liệu thực thi cho mỗi flow".
- Sau mỗi thí nghiệm: kết quả tổng hợp vào `06-Reports/experiment-logs/`, so với tiêu chí của issue và chỉ tiêu trong `01-Plan/plan/project-plan.md`, báo cáo bằng comment trong issue (README §6.5). Kết quả chưa đạt vẫn commit kèm phân tích nguyên nhân.
- Không xóa file; chuyển vào `_archive/`. Bỏ file khỏi git bằng `git rm --cached` (giữ file trên đĩa).
- Không di chuyển nội dung `_private/` vào các thư mục 01–07.
- Tài liệu hành chính đưa vào git phải lược bỏ kinh phí, số điện thoại, thông tin cá nhân (CODE-CONVENTION §12).
- Output của tool (GDS, SDF, SPEF, VCD, `runs/`) không commit; chỉ commit config và script.
- Không vendor thư viện bên thứ ba dưới dạng repo lồng.
- Mã public bên ngoài (khung SoC, IP mở, thư viện, mô hình upstream) chỉ kéo về bằng `bash 08-external/fetch.sh` theo commit ghim trong `08-external/deps.tsv`; không commit vào repo, không tạo `third_party/`, `vendor/`, `upstream/`. File sửa trên mã bên ngoài đặt trong `04-Src/` (overlay hoặc patch); script trong `05-Experimental/flow-*/` trỏ tới `08-external/<name>/`. Không sửa trực tiếp trong `08-external/<name>/`.

## Quy trình làm việc trên GitHub (README §6)

- Nhánh: mỗi thành viên một nhánh dài hạn `<ho-va-ten>-k<khoa>` (ví dụ `tran-thi-ha-k67`); không commit, push lên `main`. Trước khi commit, kiểm tra `git branch --show-current` không phải `main`.
- Task: mỗi việc là một issue nhãn `task`/`experiment` trong Project #8 "edabk-physical-ai-robot"; Status `Todo` → `In progress` → `Done`. Claude được tạo issue, chuyển *In progress*, comment báo cáo; **không** đóng issue, không chuyển *Done*, không approve, không merge.
- Commit: `<scope>: <mô tả> (task #<số>)` sau khi `bash tools/check-repo.sh --staged` báo 0 FAIL; không dùng `fixes #`/`closes #`.
- Thí nghiệm: chạy flow trên cây làm việc sạch, commit kết quả tổng hợp với scope `exp`, báo cáo vào issue (lệnh, commit hash, đường dẫn log, metric kèm đơn vị so với tiêu chí).
- Bàn giao: PR từ nhánh thành viên vào `main`, PI merge bằng merge commit (`--merge`).

```bash
git switch tran-thi-ha-k67 && git fetch origin && git merge origin/main                 # đồng bộ main
gh issue create -R edabk-hust/edabk-physical-ai-robot --title "Huấn luyện ACT trên dữ liệu cfg5, đánh giá MuJoCo" --label experiment --assignee @me --project "edabk-physical-ai-robot"
gh project item-edit 8 --owner edabk-hust --url https://github.com/edabk-hust/edabk-physical-ai-robot/issues/12 --field "Status" --value "In progress"
git commit -m "flow: thêm config cfg5_base8_seed42.yaml, seed 42 (task #12)" && git push
make -C 05-Experimental/flow-act-train run report CONFIG=configs/cfg5_base8_seed42.yaml
git commit -m "exp: cfg5_base8_seed42 — success rate 72 % (36/50 episode) (task #12)" && git push
gh issue comment 12 -R edabk-hust/edabk-physical-ai-robot --body "Đề nghị review @truong-nhom ... (lệnh, commit, log, kết quả)"
gh pr create -R edabk-hust/edabk-physical-ai-robot --base main --head tran-thi-ha-k67 --title "[handover] tran-thi-ha-k67 — kỳ 2026.1" --body-file 06-Reports/handover/2026-12_tran-thi-ha-k67.md
```


## Chất lượng đầu ra
- Số liệu có đơn vị và nguồn (đường dẫn log trong `06-Reports/experiment-logs/` hoặc tài liệu `R<nn>`).
- So sánh phải định lượng (accuracy, GOPS/W, µW, mm², technology node); bảng tổng hợp có sheet `Methodology`.
- Trích dẫn chuẩn IEEE, key trong `02-References/references.bib`.
- Action item có căn cứ, hành động, người làm, deadline.
