# Cache MCR theo giai đoạn: feature trước, JPEG sau

**Cập nhật yêu cầu mới:** [Ba tập và camera dropout + ACT dropout](CFG5_FROZEN_THREE_SPLITS_DROPOUT_VI.md).
Bundle mới tạo feature test với marker riêng; camera dropout 0,1 và ACT dropout
0,1 chỉ khi train. Nội dung train/val-only bên dưới mô tả phiên bản V5.

Cập nhật 12/10/2026. Bundle v5 nâng cấp từ v4 đã cài trên server.
Giữ 1317 episode có nhãn joint, loại đúng 4 ID đã được user duyệt khỏi split,
giữ nguyên raw. Dùng lại numeric/config/split/stats của job 14287; không chạy
lại audit/timeline toàn bộ trong job tạo cache mới.

## 1. Luồng hiện tại

```text
Raw MP4 + numeric nhẹ đã căn timestamp/mask
                   ↓
JPEG quality 90 tạo và giải mã tạm trong RAM → MCR đóng băng
                   ↓
Lưu feature INT8 + scale FP16; không ghi JPEG xuống đĩa
                   ↓
READY_FROZEN → train ACT mới, ResNet giữ nguyên
                   ↓
Lấy checkpoint frozen đã học → pilot JPEG 3 episode
                   ↓
Tạo JPEG từng episode → kiểm tra đầy đủ → dọn feature của episode đó
                   ↓
READY_FULL → full fine-tune từ JPEG, không cần features.npy
```

Giữ feature **2048×7×7** mỗi camera; chưa pooling xuống 4×4/3×3. JPEG vẫn
640×360, MCR resize cạnh ngắn 256 rồi center crop 224. Không thay đổi chính
sách timestamp 10Hz, tuổi quan sát/state 150ms, target joint+kẹp, mask camera
hay normalization. Transformer/projection/ACT vẫn train như v4.

JPEG tạm trong RAM ở giai đoạn đầu giúp feature có cùng đầu vào nén với JPEG
được tạo lại sau này. Không có file JPEG/SQLite trong cache frozen thuần.

## 2. Các artifact

Root mới: `data/features/cfg5_robot_all_mcr_features`.

| Thời điểm | File lớn giữ lại | Marker |
| --- | --- | --- |
| Feature pilot | Feature/scale của 3 episode | Chưa READY |
| Build feature xong | Feature/scale train+val | `READY_FROZEN.json` |
| JPEG pilot sau frozen | Feature/scale và JPEG của 3 episode | READY_FROZEN vẫn còn |
| Đang conversion | Episode đã chuyển giữ JPEG; episode chưa chuyển giữ feature | Thu hồi READY_FROZEN, chưa READY_FULL |
| Conversion xong | JPEG train+val; không còn feature/scale | `READY_FULL.json` |

`context.json` và feature contract giữ nguyên qua conversion để checkpoint
frozen khớp data/stats/preprocessing. Numeric, camera_valid.npy và episode.json
vẫn còn. `JPEG_CONVERSION.json`, `jpeg_progress.json`, `jpeg_storage_report.json`
và `FEATURE_RELEASE.json` từng episode ghi nguồn checkpoint và tiến độ dọn.
Test vẫn khóa riêng; không cache ảnh test trong các bước này.

Full loader đọc JPEG và không đòi features.npy. Frozen loader từ chối cache
JPEG-only. Muốn trở lại frozen sau khi dọn phải build lại feature từ raw vào
cache phù hợp; không resume frozen trực tiếp trên cache đã chuyển.

## 3. Ngân sách dưới 2TB

- Với 514877 tick hiện có, feature + scale vẫn khoảng **400,7 GiB**.
- Build feature dự phòng khoảng 10% và overhead; không cộng bộ JPEG lớn nữa.
- JPEG pilot đo dung lượng thật sau khi frozen xong; chưa có con số JPEG toàn
  bộ đáng tin cậy trước khi đo.
- Conversion dự phòng phần tăng cuối cùng và JPEG của episode lớn nhất.
  Đã có JPEG pilot thì dọn feature của các episode đó trước để giải phóng chỗ.
- Tạo/check JPEG xong mới unlink đúng hai file `features.npy`, `scales.npy`.
  Không giữ hai bộ feature+JPEG đầy đủ cùng lúc; vẫn cần chỗ tạm cho episode
  đang chuyển và phần dự phòng.
- Code tiếp tục dùng ngân sách soft **1,8TB thập phân**, giữ 10GiB dự phòng.
  Home đã đo khoảng 2411,9GiB vẫn cần dọn trước khi tạo cache mới. Đổi phương
  pháp cache không giải phóng những dữ liệu cũ đang nằm trong home.
- Không tự xóa raw, ZIP, cache cũ, checkpoint hay file của project khác.

## 4. Điều kiện chuyển và xử lý gián đoạn

JPEG conversion yêu cầu checkpoint MCR+ACT stage frozen có ít nhất một epoch
đã lưu, `smoke=false`, backbone đóng băng, cùng data config/stats/feature
contract. Không dùng E5/E58 TCP hoặc checkpoint stage full thay thế.

Trainer giữ shared lock trên cache suốt quá trình train. Build/conversion giữ
exclusive lock; conversion không chạy đồng thời với trainer đang dùng root
này. Sau khi bắt đầu dọn, frozen readiness bị thu hồi.

JPEG được ghi vào `rgb.sqlite.partial`; kiểm tra source trước/sau decode,
numeric hash, mask, đầy đủ tick/camera, SQLite integrity và SHA256. Publish
JPEG thành công mới cho phép dọn feature. Nếu JPEG lỗi, feature được giữ lại.

Nếu job gián đoạn giữa publish và metadata, có receipt để kiểm tra và khôi
phục. Nếu gián đoạn giữa hai lần unlink feature/scale, build chạy lại tiếp
tục từ những file còn lại. `.partial` do decode/budget lỗi cần đọc
`JPEG_ERROR.json`; không tự xóa để chạy lại mù. READY_FULL chỉ xuất hiện khi
mọi episode đã có JPEG hợp lệ và việc dọn feature đã hoàn tất.

## 5. Lệnh dùng trên server

Sau khi cài v5, làm trong repo:

```bash
cd "$HOME/cfg5-pretrain/edabk-physical-ai-robot/04-Src/software/cfg5_act"
mkdir -p "$HOME/cfg5-pretrain/logs"
```

### A. Feature pilot và build, trước khi train

Sau khi xử lý dung lượng home:

```bash
P=$(sbatch --parsable scripts/slurm/cfg5_mcr_features.sbatch pilot)
echo "Feature pilot: $P"
tail -F "$HOME/cfg5-pretrain/logs/cfg5-mcr-features-$P.out"
```

Pilot xong có `MCR_PILOT_OK`, `MCR_FEATURES_JOB_OK` và `COMPLETED 0:0`.

```bash
C=$(sbatch --parsable scripts/slurm/cfg5_mcr_features.sbatch build)
echo "Feature build: $C"
tail -F "$HOME/cfg5-pretrain/logs/cfg5-mcr-features-$C.out"
```

Build xong có `MCR_CACHE_READY`, READY_FROZEN và `COMPLETED 0:0`.
Các job này dùng 16 Slurm CPU, 128G RAM, 1 GPU. Không dùng script pilot v4
`cfg5_robot_all_mcr.sbatch` cho hướng mới vì script đó còn tạo cache combined.

### B. Frozen training

Sau khi build thành công, ví dụ lượt đầu 5 epoch, batch 16, workers 8:

```bash
T=$(sbatch --parsable scripts/slurm/cfg5_mcr_joint_train.sbatch frozen)
echo "Frozen train: $T"
tail -F "$HOME/cfg5-pretrain/logs/cfg5-mcr-joint-train-$T.out"
```

ACT khởi tạo mới, MCR có weights đã pretrain. Smoke chạy 1 batch train+val
trong thư mục riêng trước lượt thật. Checkpoint lưu ở
`outputs/cfg5_robot_all_joint_mcr_frozen`: best.pt, latest.pt, epoch_*.pt,
history.json. Gửi lại wrapper sẽ resume latest nếu có và runtime khớp.

### C. JPEG pilot sau khi frozen kết thúc

```bash
J=$(sbatch --parsable scripts/slurm/cfg5_mcr_jpeg.sbatch pilot)
echo "JPEG pilot: $J"
tail -F "$HOME/cfg5-pretrain/logs/cfg5-mcr-jpeg-$J.out"
```

Mặc định dùng frozen/best.pt ở mục B. Pilot không xóa feature, chỉ tạo JPEG
3 episode để đo. Có `MCR_JPEG_PILOT_OK` và `COMPLETED 0:0` mới chuyển tiếp.

### D. Conversion và dọn feature theo từng episode

```bash
J=$(sbatch --parsable scripts/slurm/cfg5_mcr_jpeg.sbatch build)
echo "JPEG conversion: $J"
tail -F "$HOME/cfg5-pretrain/logs/cfg5-mcr-jpeg-$J.out"
```

`build` của wrapper này truyền `--release-features` rõ ràng. Nó dọn feature
đã xác minh khi JPEG tương ứng đạt kiểm tra. Job JPEG dùng 8 CPU, 16G RAM,
không yêu cầu GPU; Python chạy trực tiếp trong sbatch, không dùng srun.
Kết thúc thành công có `MCR_JPEG_FULL_READY`, READY_FULL và `COMPLETED 0:0`.

### E. Full fine-tune

Sau READY_FULL, ví dụ một lượt full 5 epoch:

```bash
T=$(sbatch --parsable scripts/slurm/cfg5_mcr_joint_train.sbatch full \
  data/features/cfg5_robot_all_mcr_features \
  outputs/cfg5_robot_all_joint_mcr_full \
  5 16 8 outputs/cfg5_robot_all_joint_mcr_frozen/best.pt)
echo "Full train: $T"
```

Lượt full giữ output riêng, bắt đầu từ frozen checkpoint, dùng optimizer mới.
Backbone có gradient và learning rate riêng; feature frozen cũ không còn
được dùng. Stage full chạy từ JPEG nên không cần MP4 ở từng batch training.

Các lệnh trên là hướng dẫn, chưa được tự submit từ máy local. Mỗi bước chỉ
gửi sau khi bước trước hoàn thành đúng marker/state. Dùng `sacct -j JOB_ID -X
--format=JobID,State,ExitCode,Elapsed` để xác nhận; thay JOB_ID bằng số thật.

## 6. Code và kiểm thử

- `src/data/feature_cache.py`: feature-only, loader theo stage, readiness.
- `src/data/staged_mcr.py`: kiểm checkpoint, tạo JPEG, kiểm payload, dọn feature.
- `scripts/prepare_mcr_cache.py --features-only`: pilot/build feature trước.
- `scripts/prepare_mcr_jpeg.py`: pilot/build/check JPEG sau frozen.
- `scripts/train_mcr_act.py`: khóa cache, kiểm marker tương ứng frozen/full.
- `tests/test_staged_mcr.py`: feature-only giống combined, JPEG-only loader,
  giữ raw/numeric, JPEG lỗi không dọn feature, checkpoint sai bị chặn, resume
  sau gián đoạn, pipeline CLI và marker/lock API. Lock API được mock khi test
  trên Windows; chưa chạy kiểm tra kernel lock Linux hay GPU job thật.

Cache cũ combined v1 vẫn được loader hỗ trợ; v5 dùng root mới để không trộn
chính sách lưu và không đụng payload cũ. Bundle chỉ nâng cấp source/docs/test.
