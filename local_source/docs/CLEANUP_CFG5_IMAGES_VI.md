# Dọn ảnh preprocessing CFG5 sau khi chuyển sang dữ liệu gắp thả

Người dùng yêu cầu ngày 2026-10-09 dọn khoảng 400 GB ảnh của lượt pretrain cũ.
Nguồn cũ là `data/processed/cfg5_all`, layout phẳng: train/val/test cùng root,
membership nằm trong split JSON; không xoá cả root hoặc các thư mục episode.
Audit lịch sử cho 531 train + 66 validation ghi 400.315 GB ảnh, khoảng 372.823 GiB.
Đây không phải số đo dung lượng server hiện tại.

Hướng triển khai đã được sửa theo yêu cầu người dùng: MCR cache, frozen ACT warmup
và full fine-tune dùng dữ liệu gắp thả mới, không warmup lại CFG5. Job cũ 14127/14128
được yêu cầu huỷ; job build dữ liệu mới 14130 được người dùng gửi. Script không tự
huỷ job, không tự đăng ký training hoặc xoá dữ liệu.

## File được phép xoá

Chỉ `images.npy` của 531 train + 66 validation trong root CFG5 cũ, theo cấu hình
server `configs/act/cfg5_all_pretrain_bs8_80epoch.json` và membership đã khóa.
Giữ metadata, state/action/timestamp/segment, những file khác, test, raw, checkpoint,
cache MCR và preprocessing dữ liệu mới. Muốn chạy lại pipeline ảnh NPY CFG5 cũ
sau khi dọn phải tạo lại ảnh từ dữ liệu gốc.

`scripts/cleanup_cfg5_images.py` mặc định chỉ audit. Kiểm đường dẫn thật, không qua
symlink, NPY header/size/contract, disjoint split, root source mới/cache context,
checkpoint E5 tồn tại. Chỉ cho phép job cache mới được chỉ định còn trong squeue;
job cũ hoặc bất kỳ job khác còn pending/running/completing làm script dừng.
Thêm `--apply` để xoá từng file, không recursive delete. Audit và journal lưu tại
`outputs/data_cleanup`. Script quét và kiểm lại trước lần xoá thực tế.

## Upload một file nhỏ từ PowerShell local

```powershell
scp "F:\cfg5_mophong\local_source\scripts\cleanup_cfg5_images.py" user08@172.16.3.8:~/cfg5-pretrain/cleanup_cfg5_images.py
```

## Chạy trên server

Thoát `tail -F` bằng Ctrl+C, rồi:

```bash
cd "$HOME/cfg5-pretrain/edabk-physical-ai-robot/04-Src/software/cfg5_act"
./.venv-data/bin/python "$HOME/cfg5-pretrain/cleanup_cfg5_images.py" --repo "$PWD" --allow-job 14130
```

Nếu có `CFG5_CLEANUP_AUDIT_OK`, root đúng CFG5 cũ và số byte cần dọn phù hợp:

```bash
./.venv-data/bin/python "$HOME/cfg5-pretrain/cleanup_cfg5_images.py" --repo "$PWD" --allow-job 14130 --apply
```

Thành công có `CFG5_OLD_IMAGES_DELETED`. Nếu job cache mới đã kết thúc thì vẫn
có thể giữ `--allow-job 14130`; whitelist không yêu cầu job đó phải đang chạy.
Nếu script báo một job khác còn hoạt động, đọc trạng thái job đó trước khi dọn;
không thêm tuỳ tiện job dùng dữ liệu cũ vào whitelist.

## Kiểm chứng

6 tests fixture nhỏ đạt: xoá đúng train/val, giữ test/numeric, chặn split chồng,
chặn file đổi trước xoá, chặn đường dẫn ngoài root, chặn job pending/completing,
và chạy audit rồi apply trên cây giả đủ 597 train/val + 66 test, giữ checkpoint
và dữ liệu mới. Không xoá dữ liệu thật local/server trong kiểm thử này.
