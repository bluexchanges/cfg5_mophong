# Preprocessing MCR đóng băng: ba tập và dropout khi train

Chốt ngày 12/10/2026: camera dropout 0,1 + dropout ACT 0,1. Dùng toàn bộ 1.317
robot episode có joint trong hai nguồn raw, giữ nguyên 4 episode không có joint
ngoài split theo quyết định đã duyệt. Không thêm lọc rating/camera và không đổi raw.

## 1. Split và normalization

Giữ split đã khóa bởi job 14287: **train 1.053, val 132, test 132**. Không random
chia lại, không trộn episode giữa các tập. Normalization đã fit riêng trên train
được dùng nguyên trạng cho val/test. Mục tiêu: 7 góc khớp radian + độ mở kẹp mét.

Giờ feature được tạo cho cả ba tập, cùng MCR/preprocess/quantization. Cache
test có marker riêng `READY_TEST_FROZEN.json`; `READY_FROZEN.json` chỉ liệt kê
train+val. Dataset/sampler train chỉ lấy `split['train']`, validation chỉ val.
Test được dành cho đánh giá checkpoint đã chọn, không dùng chọn best hoặc fit stats.

## 2. ResNet và cache

MCR ResNet50 chính thức được strict-load và kiểm SHA256. `lr_backbone=0`,
backbone luôn eval, không đổi weights hoặc thống kê BatchNorm. Mỗi ảnh RGB
được căn timestamp/mask theo numeric nhẹ hiện có; tạo/giải mã JPEG Q90 tạm
trong RAM rồi resize cạnh ngắn 256, center crop 224, normalize theo MCR.

Lưu feature INT8 (8,2048,7,7) + scale FP16 cho mỗi tick. Camera thiếu có slot
zero và mask false. Không lưu JPEG xuống đĩa trong giai đoạn frozen, không tạo
ảnh NPY lớn, không pooling 4x4, không làm random augmentation khi trích feature.

Con số 400,7 GiB cũ chỉ gồm train+val. Nếu tổng số tick toàn bộ giống snapshot
local 574.493, payload ba tập khoảng **447,1 GiB**. Job tính lại từ numeric
server, cộng 10%/overhead và 10 GiB reserve. Chỉ build nếu toàn bộ home dùng
chung và filesystem đủ chỗ theo ngân sách 1,8 TB thập phân. Không bỏ qua `phuc`.

## 3. Hai dropout không nằm trong preprocessing

**Camera dropout p=0,1:** mỗi mẫu train, từng camera đang hợp lệ có xác suất
10% bị che thêm. Mask có thật từ dữ liệu không bị sửa; camera vốn thiếu không
bao giờ trở thành hợp lệ. Nếu tất cả camera bị drop, chọn ngẫu nhiên một camera
thật để giữ. Mẫu vốn chỉ có một camera luôn giữ camera đó. Không rescale feature
còn lại; feature camera bị che được zero và token bị loại qua attention padding mask.

Ví dụ mask thật `[1,1,0,1,0,1,1,0]` có thể thành `[1,0,0,1,0,1,1,0]` khi
train: camera thứ hai bị che thêm, camera thứ ba/năm/tám vẫn thiếu như raw.
Ở lần lấy mẫu khác dropout có thể khác, còn feature trên đĩa không thay đổi.

**Dropout ACT p=0,1:** dropout sẵn trong Transformer/encoder được đặt 0,1.
ResNet vẫn đóng băng. Chỉ train ACT, visual projection và các phần policy trainable.

Val/test dùng `eval()`: cả hai dropout tắt. Camera thiếu thật vẫn được mask.
Checkpoint lưu probability trong model config/contract; resume đòi cùng contract,
và RNG được lưu để phục hồi camera dropout. Không resume run có dropout khác.

Dropout không giảm dung lượng cache. Không dùng dropout để bỏ episode/nhãn joint.

## 4. Cài bundle mới

PowerShell local:

```powershell
scp "F:\cfg5_mophong\cfg5_mcr_three_splits_dropout_bundle.zip" user08@172.16.3.8:~/cfg5-pretrain/
```

Server:

```bash
cd "$HOME/cfg5-pretrain/edabk-physical-ai-robot/04-Src/software/cfg5_act"
unzip -o "$HOME/cfg5-pretrain/cfg5_mcr_three_splits_dropout_bundle.zip" \
  -d "$HOME/cfg5-pretrain/cfg5_mcr_three_splits_dropout_bundle"
./.venv-data/bin/python "$HOME/cfg5-pretrain/cfg5_mcr_three_splits_dropout_bundle/install_bundle.py" --repo "$PWD" --dry-run
./.venv-data/bin/python "$HOME/cfg5-pretrain/cfg5_mcr_three_splits_dropout_bundle/install_bundle.py" --repo "$PWD"
```

Installer kiểm toàn bộ file trước khi cài, chỉ nhận baseline V5/preflight đã biết
hoặc file mới chưa có; bản sửa riêng trên server làm dừng và cần adaptation.
Backup code cũ trong `outputs/install_backups/mcr_dropout_<timestamp>`.
Không cài lại data/config/split/stats/checkpoint, không patch server pretrain.py.

## 5. Check và preprocessing

```bash
J=$(sbatch --parsable --time=00:30:00 scripts/slurm/cfg5_mcr_preflight.sbatch check)
echo "Job check: $J"
tail -F "$HOME/cfg5-pretrain/logs/cfg5-mcr-preflight-$J.out"
```

Report camera vẫn cùng file `outputs/data_audit/cfg5_robot_all_camera_report.html`.
JSON `outputs/mcr_checks/mcr_frozen_check_<jobid>.json` tính budget cả test.
Nếu `full_build_fits=True`, chạy:

```bash
P=$(sbatch --parsable scripts/slurm/cfg5_mcr_preflight.sbatch build)
echo "Job preprocessing ba tập: $P"
tail -F "$HOME/cfg5-pretrain/logs/cfg5-mcr-preflight-$P.out"
```

Job build tự kiểm lại ResNet/budget, pilot 3 episode train, rồi build ba tập.
Feature train/val đã có đúng contract được kiểm và dùng lại, không cần giải mã
lại chúng chỉ vì thêm test. Không tự train. Success marker `MCR_CACHE_SPLITS`
phải ghi train=1053/val=132/test=132, rồi `MCR_FROZEN_PREPROCESS_JOB_OK`.

Cache: `data/features/cfg5_robot_all_mcr_features`. Có `READY_FROZEN.json`
và `READY_TEST_FROZEN.json` sau full build thành công.

## 6. Training sau khi cache xong

Wrapper `cfg5_mcr_joint_train.sbatch` đã thêm cả `--camera-dropout 0.1` và
`--act-dropout 0.1` vào smoke và train thật. ACT mới, MCR pretrained đóng băng;
không khởi tạo từ E5/E58 TCP. Chỉ chạy training khi bạn yêu cầu giai đoạn đó.
Output mặc định mới: `outputs/cfg5_robot_all_joint_mcr_frozen_drop01`, tránh
resume nhầm một run cũ dùng dropout khác; stage full có output riêng.

Đánh giá frozen test sau khi đã chọn checkpoint bằng val: script
`scripts/evaluate_mcr_frozen.py` nhận `--split test`, kiểm marker test và
checkpoint frozen cùng feature/stats, tắt cả hai dropout và không chọn best.
Chạy trong GPU sbatch nếu dùng cuda. Offline loss không phải success rate MuJoCo.

Bundle này mở rộng **frozen** sang test. Conversion JPEG V5 còn phục vụ train/val;
test frozen tiếp tục dùng feature gốc. Nếu chuyển sang full fine-tune và đánh giá
full trên test, phải chuẩn bị JPEG test/cùng backbone mới ở giai đoạn đó.
