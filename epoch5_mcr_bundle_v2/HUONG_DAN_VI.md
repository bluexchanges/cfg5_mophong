# Đánh giá ACT epoch 5, rồi dùng MCR để giảm dung lượng và thời gian train

## 1. Đã chuẩn bị gì

- Đánh giá **nguyên checkpoint `outputs/act_pretrain/best.pt`, human epoch 5** trên 10 episode validation đã split của dữ liệu pick/place. Không tạo optimizer, không backward, không cập nhật weights. Giữ normalization của checkpoint.
- Hai phép đo: inference trên ảnh/state đã ghi để tính sai số; rollout điều khiển MuJoCo bằng ảnh/state mới để tính task success và xuất video.
- MCR ResNet50 đã có trọng số học từ DROID. Đóng băng backbone và BatchNorm để tạo cache đặc trưng, sau đó chỉ train ACT; giai đoạn full fine-tune mở backbone và đọc JPEG trong cache.

Epoch 5 là checkpoint bạn chọn để khảo sát transfer. Biểu đồ vẫn cho thấy epoch 58 có validation loss nhỏ nhất toàn lượt cũ. E5/E58 là checkpoint dự án đã pretrain trên CFG5; chúng không phải full ACT policy do tác giả phát hành.

## 2. MCR lấy từ đâu và thay phần nào

Nguồn chính thức: [repository MCR](https://github.com/luccachiang/robots-pretrain-robots),
[weights](https://huggingface.co/GqJiang/robots-pretrain-robots),
[paper ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/cb9e74b26c174b60e94e14c73646a55d-Paper-Conference.pdf).
MCR học trên DROID, với Franka trong nhiều cảnh/tác vụ. Đây chưa phải backbone học trên nhiều loại cánh tay.
[CrossFormer](https://crossformer-model.github.io/) học trên 30 embodiments nhưng dùng backbone JAX khác;
[R3M](https://github.com/facebookresearch/r3m) dùng video người Ego4D. Bạn đã chọn triển khai MCR trước.

Asset cố định:

```text
HF revision: 057651ca0b6a27aba390449603f674cd0977f87c
file: mcr_resnet50.pth
bytes: 94345861
SHA256: 21560db9e795384a7b73b0626d33d6525f9d4c5de11fca6ded52089aab14f97c
```

File 94 MB đã tải và kiểm SHA ở local `outputs/backbones/mcr_resnet50.pth`.
Bundle code không chứa checkpoint; job cache sẽ tải asset này một lần trên server.
Nguồn MCR có license MIT; asset Panda đi kèm bundle giữ LICENSE và manifest Apache-2.0 của MuJoCo Menagerie.

ACT cũ dùng ResNet18, 512 channels. MCR dùng ResNet50, 2048 channels, nên có adapter và checkpoint riêng.
Warmup lấy các tensor ACT tương thích từ E5, thay backbone bằng MCR và khởi tạo mới phép chiếu `2048 -> hidden_dim`.
Không thể nạp nguyên policy E5 vào MCR rồi coi như cùng kiến trúc. Normalization số, camera order, 10 Hz và chunk 20 được kiểm trước khi chuyển.

## 3. Upload, cài và chạy đánh giá E5 trước

**PowerShell trên máy Windows:**

```powershell
scp "F:\cfg5_mophong\epoch5_mcr_bundle_v2.zip" user08@172.16.3.8:/home/user08/cfg5-pretrain/
```

**Terminal server:**

```bash
unzip -n "$HOME/cfg5-pretrain/epoch5_mcr_bundle_v2.zip" -d "$HOME/cfg5-pretrain/epoch5_mcr_bundle_v2"
cd "$HOME/cfg5-pretrain/edabk-physical-ai-robot/04-Src/software/cfg5_act"
./.venv-data/bin/python "$HOME/cfg5-pretrain/epoch5_mcr_bundle_v2/install_bundle.py" --repo "$PWD" --dry-run
./.venv-data/bin/python "$HOME/cfg5-pretrain/epoch5_mcr_bundle_v2/install_bundle.py" --repo "$PWD"
mkdir -p "$HOME/cfg5-pretrain/logs"
```

Dùng Python của `.venv-data`; Python3 hệ thống cũ từng thiếu `Path.is_relative_to`.
Installer kiểm toàn bộ bundle trước khi cài, bỏ qua file giống nhau và từ chối ghi đè file server khác.
V2 kèm module metric riêng, không yêu cầu `scripts/evaluate_pretrain_test.py` phải nằm trong repo.
Test fixture được bổ sung nếu thiếu; bản test fixture đang có trên server được giữ nguyên.
Nếu thiếu runtime, installer in `REPO=...` và tất cả dòng `MISSING: ...` để xác định đúng file.
Giữ split hiện tại và mọi checkpoint/output cũ. Nếu installer báo file khác, gửi đúng dòng lỗi để so sánh phiên bản.

Kiểm dependency nhẹ trên head node:

```bash
./.venv-data/bin/python -c 'import torch, torchvision, cv2, mujoco; print(torch.__version__, torchvision.__version__, cv2.__version__, mujoco.__version__)'
```

Nếu chưa có MuJoCo, cài vào chính venv này: `./.venv-data/bin/python -m pip install 'mujoco==3.14.0'`.
Job dùng EGL trên GPU server, không cần mở cửa sổ GUI. Bundle có Panda assets (~34 MB), không tải thêm raw/data.

Đăng ký **job đánh giá**, không train:

```bash
if J=$(sbatch --parsable scripts/slurm/probe_epoch5.sbatch); then
  echo "Job đánh giá E5: $J"
  squeue -j "$J" -o '%.18i %.24j %.2t %.10M %R'
fi
tail -f "$HOME/cfg5-pretrain/logs/cfg5-e5-probe-$J.out"
# Ctrl+C chỉ dừng xem log.
```

Kiểm kết thúc:

```bash
sacct -j "$J" -X --format=JobID,State,ExitCode,Elapsed
tail -n 40 "$HOME/cfg5-pretrain/logs/cfg5-e5-probe-$J.err"
```

Hoàn thành khi `COMPLETED 0:0` và `EPOCH5_PROBE_JOB_OK`. Output:
`outputs/epoch5_transfer_probe_JOBID/`.

- `selection.json`: 10 episode cố định từ **val**, seed 42, SHA checkpoint, optimizer_steps=0.
- `frozen_checkpoint.pt`: bản snapshot E5 để tái lập (thêm khoảng 320 MiB).
- `report.json`: sai số vị trí mm, góc độ, kẹp mm; số rollout thành công, hoàn tất và lỗi hạ tầng.
- `review.html`: tổng hợp video và link kết quả từng episode.
- Mỗi episode: `rollout.mp4`, `rollout_report.json`, trace raw/executed action, dự đoán đầu chunk NPZ.

Download kết quả về Windows, thay **JOBID bằng số thật**:

```powershell
scp -r user08@172.16.3.8:/home/user08/cfg5-pretrain/edabk-physical-ai-robot/04-Src/software/cfg5_act/outputs/epoch5_transfer_probe_JOBID "F:\cfg5_mophong\"
```

Mở `review.html`, hoặc video bằng VLC nếu browser không hỗ trợ codec MPEG-4 của OpenCV.
CLI có `--viewer` để mở MuJoCo trên máy có GUI và đủ episode raw; server mặc định xuất video.

### Ý nghĩa success rate

Mỗi rollout reset cùng seed/scene/camera của demo, kiểm measured state đầu khớp trong 0.2 mm và dùng evaluator của dự án.
Success yêu cầu cả ba chốt đúng lỗ/màu, đúng chiều sâu/độ nhô, được đỡ, đủ thẳng và robot đã nhả ổn định.
CFG5 pretrain dùng future measured pose làm proxy action; dữ liệu mô phỏng dùng commanded pose cùng tick.
TCP ảo của mô phỏng cũng chưa là factory EE transform CFG5 đã đo. Kết quả đo transfer này bao gồm các khác biệt đó.
Mỗi tick 10 Hz, policy chỉ thấy ảnh và measured state hiện tại, z=0; thực thi action đầu chunk rồi quan sát lại.
Giới hạn 600 tick = 60 giây mô phỏng. Chưa hoàn thành trước mốc này được báo `horizon_reached`.

Lệnh dự đoán được giới hạn workspace/kẹp và chuẩn hóa quaternion trước controller; trace giữ cả raw action và số lần clip.
Đây là policy kết hợp guard/controller của mô phỏng. Không dùng action demo để điều khiển rollout.
Sai số offline không chuyển thành tỉ lệ hoàn thành. 10 val episode chỉ là diagnostic để phát triển, chưa là test tổng quát.
Nếu lỗi dựng scene/EGL/version, báo riêng `rollout_errors`; không kết luận policy thất bại từ lỗi hạ tầng.

## 4. Cache mới lưu gì? Dung lượng bao nhiêu?

Tại từng tick hợp lệ, giữ 8 ảnh đồng bộ và state/action/segment của split hiện tại.
Giải mã MP4 tuần tự một lần, encode JPEG quality 90 vào SQLite; decode JPEG này để lấy đầu vào MCR.
Preprocess theo cách MCR công bố: resize cạnh ngắn 256, center crop 224, normalize ImageNet.
Lấy feature map trước global pooling, shape **8 x 2048 x 7 x 7**; giữ cấu trúc không gian cho ACT.
Lưu INT8 với scale FP16 riêng cho mỗi tick/camera/channel. Ghi max quantization error và hash từng chunk.

| Nội dung mỗi episode | Dùng khi nào |
|---|---|
| `features.npy` INT8 + `scales.npy` FP16 | Train ACT khi MCR đóng băng |
| `rgb.sqlite` chứa JPEG gốc 640x360 | Full fine-tune: đọc ảnh độc lập, không seek MP4 |
| state/action/timestamps/segment NPY nhỏ | Cả hai giai đoạn; chunk không vượt gap hoặc episode |
| `episode.json` | Hash backbone/transform/split/stats, kích thước và provenance |

**Cache lưu đặc trưng của từng ảnh, không chỉ lưu weights.** Với 407.776 tick train+val hiện tại,
feature+scale có payload khoảng **317.3 GiB**; còn thêm JPEG, numeric/metadata và overhead.
Dung lượng JPEG sẽ đo từ pilot 3 episode, chưa có số đo toàn bộ dữ liệu server.
Phải cộng raw, ~400 GB processed CFG5 cũ, checkpoint, archive tải về và cache mới vào quota home.
Không cần tạo thêm 2 TB RGB giải mã.

Đây là thay đổi representation và có đánh đổi: JPEG/INT8 làm mất một phần độ chính xác;
center crop cắt cạnh ảnh, map 7x7 có ít vị trí hơn backbone cũ. MCR pretrained không bảo đảm bài lắp chốt chính xác hơn.
Cache và đường dùng ảnh trực tiếp frozen có cùng phép quantization; full fine-tune dùng feature sống không quantize để truyền gradient.
Full fine-tune vẫn giải mã JPEG trên CPU, nhưng mỗi ảnh đọc độc lập. Không hứa tốc độ trước khi đo GPU thực.

## 5. Tạo cache CFG5 rồi warmup ACT

Chỉ làm sau khi xem kết quả E5. Các lệnh này dùng **531 train + 66 val CFG5** đã xử lý NPY trước đây; không đọc test.
Tài liệu quota: soft 1.8 TB / hard 2 TB cho cả home. Script đo `du` trên compute node và `disk_usage`,
ước tính phần thêm + dự phòng. `df` là dung lượng cả NFS, không là quota riêng user08.

```bash
BASE=configs/act/cfg5_all_pretrain_bs8_80epoch.json
CACHE=data/features/cfg5_mcr
C=$(sbatch --parsable scripts/slurm/mcr_cache.sbatch pilot "$BASE" "$CACHE")
echo "Pilot: $C"
tail -f "$HOME/cfg5-pretrain/logs/cfg5-mcr-cache-$C.out"
```

Khi pilot COMPLETED, đọc `data/features/cfg5_mcr/storage_report.json`: dung lượng home trước chạy,
pending estimate, quantization error. Có ngân sách thì build:

```bash
C=$(sbatch --parsable scripts/slurm/mcr_cache.sbatch build "$BASE" "$CACHE")
echo "Build: $C"
```

Build chỉ bắt đầu nếu đã có ít nhất 3 episode pilot hợp lệ. Chạy lại cùng lệnh sẽ bỏ qua episode hoàn chỉnh sau kiểm contract, kích thước, số JPEG và SHA numeric.
Muốn kiểm SHA toàn payload feature/JPEG, dùng `sbatch scripts/slurm/mcr_cache.sbatch check "$BASE" "$CACHE" --full-check`;
phép kiểm này đọc toàn bộ cache nên có thể lâu.
`.partial` bị giữ lại để đọc `ERROR.json`; không tự xóa hoặc gọi đó là episode hoàn chỉnh.
Nếu job bị ngắt, chuyển đúng thư mục `.partial` ra chỗ quarantine trong cùng cache sau khi đọc lỗi, rồi build lại;
episode đang dở phải tạo lại, episode hoàn chỉnh được giữ. Không resume giữa một video.

Chỉ sau `MCR_CACHE_READY` và COMPLETED mới đăng ký train:

```bash
T=$(sbatch --parsable scripts/slurm/mcr_train.sbatch frozen \
  data/features/cfg5_mcr outputs/act_pretrain/best.pt outputs/cfg5_mcr_frozen 5 16 8)
echo "Train ACT frozen 5 epoch: $T"
tail -f "$HOME/cfg5-pretrain/logs/cfg5-mcr-train-$T.out"
```

Đây là **train thật 5 epoch**, không phải job check. Job tự smoke 1 train batch + 1 val batch ở output riêng trước.
Chỉ ACT/projection học; MCR weights và BatchNorm giữ nguyên. Batch 16, 8 loader workers ban đầu,
Slurm yêu cầu 16 logical CPU/128G/1 GPU. Chỉ tăng batch/workers sau khi đo throughput và RAM/VRAM.
Model MCR trong gói cache phải giữ nguyên; đổi backbone/transform sẽ cần cache mới.

## 6. Cache dữ liệu của bạn và full fine-tune

Với split pick/place hiện tại 731 train + 91 val, dùng numeric light data đã tạo ở lượt MP4 trước.
Không decode test 91 episode trong hai giai đoạn học.

```bash
BASE=configs/act/pick_place_finetune_mp4_w16.json
CACHE=data/features/pick_place_mcr
C=$(sbatch --parsable scripts/slurm/mcr_cache.sbatch pilot "$BASE" "$CACHE")
# Chờ COMPLETED, xem storage_report.json rồi:
C=$(sbatch --parsable scripts/slurm/mcr_cache.sbatch build "$BASE" "$CACHE")
# Chờ MCR_CACHE_READY + COMPLETED rồi:
T=$(sbatch --parsable scripts/slurm/mcr_train.sbatch full \
  "$CACHE" outputs/cfg5_mcr_frozen/best.pt outputs/pick_place_mcr_full 5 8 8)
echo "Full fine-tune 5 epoch: $T"
```

Full fine-tune dùng **MCR+ACT đã warmup**, optimizer mới, ACT LR 1e-5 / backbone LR 1e-6; backbone và BN được mở.
Ảnh đọc từ JPEG cache. Feature INT8 cũ không được dùng làm đầu vào khi backbone thay đổi.
Numeric normalization phải giống nguồn; script sẽ từ chối nếu không khớp.
Nếu muốn warmup frozen ngay trên pick/place thay vì CFG5, dùng stage `frozen` và E5 cho cache pick/place;
đó là một thí nghiệm khác, đặt output riêng và ghi lại lựa chọn.

Mỗi epoch mới lưu `epoch_001.pt`, `epoch_002.pt`... (weights), `best.pt` (val thấp nhất), `latest.pt` (optimizer/RNG để resume), `history.json`.
Lưu thêm archive epoch tốn dung lượng checkpoint; ngân sách cache có dự phòng nhưng vẫn theo dõi tổng home.
Chạy lại đúng lệnh sbatch sẽ resume latest của output đó; batch/workers/stage phải giống.
Mất kết nối SSH không làm sbatch dừng. Hết walltime/dừng job thì chỉ tiếp tục từ epoch đã lưu; epoch dở phải chạy lại.

## 7. Đã kiểm chứng và còn phải kiểm trên server

Local: kiểm tải MCR đúng SHA và strict-load ResNet50; forward/backward fixture;
đóng băng gradient và BN; cached/live frozen inference khớp; chunk/gap/padding/state/action khớp;
cache hash/reuse; vòng train/val thật một batch; dựng scene thật và xuất video hai bước bằng policy giữ nguyên state.
Các test không đo chất lượng E5 và không phải rollout thành công của ACT.

Chưa có checkpoint E5 local, chưa chạy 10 rollout E5, full cache hoặc train GPU trên server.
Chưa đo JPEG toàn dataset, throughput, VRAM hoặc thời gian train toàn lượt.
Job E5/check/smoke và pilot trên server là bước kiểm tiếp theo. Dừng ngay nếu lỗi hạ tầng;
không coi thông báo tạo file code là kết quả đánh giá mô hình.
