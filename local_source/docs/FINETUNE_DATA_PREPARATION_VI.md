# Chuẩn bị demo pick_place từ hai thư mục để fine-tune ACT

## 1. Những gì đã xác minh

Ngày 2026-10-08, đã đọc hai mẫu người dùng cung cấp:

| Mẫu | Kiểu cảnh | Thời điểm | Camera | Ảnh |
|---|---|---:|---:|---|
| pick_place_000042 | Cơ bản, variation.varied=false | 497 | 8 | 640×360 RGB |
| pick_place_000487 | rich: màu, ánh sáng, nền, vật gây nhiễu | 503 | 8 | 640×360 RGB |

Đây là demo `mujoco_scripted_controller`, success=true; ACT chưa dùng để tạo demo.
Controller được biết pose vật khi tạo nhãn. Policy học chỉ nhận ảnh và state.
Không đưa object pose, stages, scene_config, success hay joint_positions vào input học.

`robot_trajectory.npz` chứa numeric arrays:

| Trường | Shape | Ý nghĩa |
|---|---|---|
| timestamps_ms | (T,) | tick 0,100,200,... ở 10 Hz |
| measured_tcp_gripper | (T,8) | TCP đo thực tế xyz, quaternion wxyz, độ mở kẹp, mét |
| commanded_tcp_gripper | (T,8) | Lệnh TCP/kẹp mà controller gửi tại tick đó |
| joint_positions_rad | (T,7) | Góc khớp để kiểm tra; không là action 8D |

Metadata ghi pre-step: RGB/state/command cùng tick; command điều khiển bước tiếp theo.
Do đó sample ở t=0 nhận 8 ảnh + measured state[0]; đáp án là command[0:20].
Ở t=100 ms, đáp án command[1:21]. Cuối episode thiếu target thì padding có mask.
Không nối chunk qua gap. Một episode 497 tick liên tục tạo 497 samples ở adapter này.
Pretrain CFG5 cũ dùng measured future pose tại t+1; mặc định cũ vẫn giữ nguyên.

`transformed/*.npy` và camera timestamps là legacy object NPY. Adapter mới đọc state/action
từ NPZ numeric, chỉ đọc timestamps bằng unpickler giới hạn NumPy và đối chiếu từng tick.
Không lấy measured TCP làm command và không nhân độ mở kẹp thêm 0.001: demo đã dùng mét.

## 2. Chia train/validation/test

Hai thư mục raw trên server:

```
~/cfg5-pretrain/downloads/finetune_data
~/cfg5-pretrain/downloads/dataset_moi
```

Tên đúng theo ảnh là `dataset_moi`, không phải `data_moi`.
Chúng là hai nguồn của một dataset task đích, không tự coi một nguồn là train hay test.
Không cần di chuyển/copy video để gộp. Manifest giữ đường dẫn của mỗi episode.

Script audit đủ metadata, trajectory, timestamp và video header của cả 8 camera.
Chỉ dùng demo success=true trong bản behavior cloning này. Demo lỗi ghi report; mặc định
dừng để người dùng kiểm tra, chưa khóa split. Tên trùng được so sánh cả SHA256 video:
bản giống hệt chỉ giữ một; tên giống nhưng nội dung khác thì dừng, không ghi đè.

Group là connected component của **cùng seed hoặc cùng numeric trajectory**.
Group luôn ở một split, giữ cả camera và mọi chunk trong cùng tập. Đây là cách bảo thủ
giảm rò rỉ giữa các biến thể scene dùng chung seed/trajectory; không bảo đảm phát hiện
được mọi bản gần giống hoặc seed bị đổi tên. Nếu generator có một ID parent riêng cần
bổ sung group đó trước khi chốt thí nghiệm.

Chia seed 42, mục tiêu 80% train / 10% val / 10% test, stratify theo variation profile
(fixed/rich/...). Ví dụ 1.000 episode độc lập, cùng profile: 800/100/100. Khi có nhiều
profile, rounding được tính trong từng profile nên tổng có thể lệch vài episode.
Không tách group để ép tỷ lệ. Profile có dưới 3 group độc lập được giữ toàn bộ ở train;
val/test lấy từ các group của profile còn lại. Cần ít nhất 2 group còn lại cho val/test.
Với dữ liệu người dùng audit, fixed chỉ có 1 group: không thể đánh giá fixed độc lập
bằng cách chia các bản lặp sang val/test. Split JSON ghi train_only_profiles và
assignment_notes để giới hạn kết luận. Khi exact grouping không khả thi, tỷ lệ thực
có thể lệch 80/10/10; giữ nguyên group quan trọng hơn ép số episode.
Hai mẫu local không đủ để chia 3 tập chính thức.

Output membership: `data/splits/pick_place_finetune_splits.json`. Sau khi khóa,
không tạo lại split/đổi seed theo kết quả val/test. Test chỉ giữ membership, chưa đọc
để tính loss/chọn checkpoint trong lượt train. Audit schema test được phép; không fit
statistics từ test.

## 3. Upload và cài trên server

PowerShell trên máy Windows:

```powershell
scp "F:\cfg5_mophong\finetune_prepare_bundle.zip" user08@172.16.3.8:/home/user08/cfg5-pretrain/
```

Terminal SSH trên server:

```bash
unzip -n "$HOME/cfg5-pretrain/finetune_prepare_bundle.zip" \
  -d "$HOME/cfg5-pretrain/finetune_prepare_bundle"
cd "$HOME/cfg5-pretrain/edabk-physical-ai-robot/04-Src/software/cfg5_act"
./.venv-data/bin/python "$HOME/cfg5-pretrain/finetune_prepare_bundle/install_bundle.py" --repo "$PWD"
mkdir -p "$HOME/cfg5-pretrain/logs"
```

Installer sao lưu file thay đổi, kiểm checksum của bundle và phiên bản file server.
Nếu code trên server khác phiên bản đã kiểm, installer dừng trước khi sửa và báo tên
file; gửi output đó để áp patch đúng phiên bản, không copy đè thủ công.
Không cài code giữa lúc có job ACT/fine-tune đang chạy. Bộ cài không dừng job.
Python hệ thống trên headnode có thể là 3.8; dùng Python 3.12 trong `.venv-data`
cho pipeline. Nếu bộ cài cũ báo thiếu `is_relative_to`, chạy lại bằng interpreter
trong `.venv-data`: lỗi này xảy ra trước khi sửa file và trước khi đăng ký job.

## 4. Audit nguồn và khóa split (job CPU)

Đăng ký:

```bash
P=$(sbatch --parsable scripts/slurm/prepare_pick_place_finetune.sbatch)
echo "Prepare job: $P"
squeue -j "$P" -o "%.18i %.20j %.2t %.10M %R"
```

Theo dõi:

```bash
tail -n 20 "$HOME/cfg5-pretrain/logs/cfg5-ft-prepare-$P.out"
tail -n 20 "$HOME/cfg5-pretrain/logs/cfg5-ft-prepare-$P.err"
sacct -j "$P" -X --format=JobID,State,ExitCode,Elapsed
```

Thành công: COMPLETED 0:0 và `FINETUNE_SPLIT_LOCKED` / `FINETUNE_PREPARE_JOB_OK`.
Output in số episode train/val/test thực, profile và GiB ảnh processed ước tính.
Report đầy đủ: `outputs/data_audit/pick_place_finetune.json`.
Nếu có rejected, đọc lý do trước. Chỉ khi đã đồng ý loại những episode lỗi, chạy lại
lệnh Python `prepare` trong sbatch thêm `--exclude-invalid`. Không thay đổi raw.

Lệnh Python được job CPU gọi:

```bash
./.venv-data/bin/python -u scripts/prepare_finetune_data.py prepare \
  --sources "$HOME/cfg5-pretrain/downloads/finetune_data" "$HOME/cfg5-pretrain/downloads/dataset_moi" \
  --init-checkpoint outputs/act_pretrain_bs8_to80/best.pt
```

## 5. Smoke trước full preprocessing

Sau prepare thành công:

```bash
S=$(sbatch --parsable scripts/slurm/smoke_pick_place_finetune.sbatch)
echo "Smoke job: $S"
```

Job giải mã 4 bước của 1 train + 1 val episode, kiểm adapter và chạy 1 train batch +
1 val batch trên GPU. Có optimizer update ở train smoke. Output riêng tại
`data/processed/pick_place_finetune_smoke` và `outputs/pick_place_finetune/smoke`.
Không dùng checkpoint smoke để khởi tạo full run.

```bash
sacct -j "$S" -X --format=JobID,State,ExitCode,Elapsed
tail -n 20 "$HOME/cfg5-pretrain/logs/cfg5-ft-smoke-$S.out"
tail -n 20 "$HOME/cfg5-pretrain/logs/cfg5-ft-smoke-$S.err"
```

Cần `FINETUNE_GPU_SMOKE_OK` và COMPLETED 0:0. Smoke thành công không chứng minh
đã preprocess full dataset. Muốn chạy lại smoke sau khi đã có smoke checkpoint, giữ
kết quả cũ bằng đổi tên thư mục `outputs/pick_place_finetune/smoke` trước; trainer
từ chối ghi đè. Không đụng output full.

## 6. Preprocessing đầy đủ train + val bằng Python

Dung lượng ảnh uint8: T×8×360×640×3 bytes. Mẫu 42: ~2,56 GiB; mẫu 487: ~2,59 GiB.
1.000 demo ~500 tick cần ~2,57 TiB nếu xử lý toàn bộ. Raw MP4 nhỏ hơn rất nhiều.
Script kiểm disk free trước, yêu cầu thêm 10% +1 GiB; quota tài khoản có thể thấp hơn
disk free, kiểm thêm `quota -s` nếu hệ thống có quota. Khi thiếu chỗ, dừng và chọn
scratch/định dạng khác trước; không giảm ảnh/ít camera mà vẫn gọi là cùng contract.

Đăng ký job CPU (8 CPU, 32 GiB RAM, 4 process decode, không xin GPU):

```bash
D=$(sbatch --parsable scripts/slurm/preprocess_pick_place_finetune.sbatch)
echo "Preprocess job: $D"
squeue -j "$D" -o "%.18i %.20j %.2t %.10M %R"
```

Lệnh Python bên trong job:

```bash
./.venv-data/bin/python -u scripts/prepare_finetune_data.py preprocess \
  --splits train val --workers 4 --resume
./.venv-data/bin/python -u scripts/prepare_finetune_data.py check
```

`--workers 4`: 4 episode được xử lý đồng thời ở 4 process; mỗi process OpenCV 1 thread.
Worker preprocessing khác `num_workers` của DataLoader khi train.
Ghi ảnh bằng memmap, không load toàn dataset vào RAM. Từng camera đọc tuần tự, BGR→RGB;
uint8 lưu trên disk, ImageNet normalization thực hiện một lần lúc lấy batch.

Output flat: `data/processed/pick_place_finetune/pick_place_xxxxxx/`, gồm:
`images.npy`, `state.npy`, `action.npy`, `timestamps_ms.npy`, `segment_ids.npy`, `episode.json`.
Train/val/test phân biệt bằng split JSON, không cần ba bản video/processed.

```bash
tail -n 20 "$HOME/cfg5-pretrain/logs/cfg5-ft-preprocess-$D.out"
tail -n 20 "$HOME/cfg5-pretrain/logs/cfg5-ft-preprocess-$D.err"
sacct -j "$D" -X --format=JobID,State,ExitCode,Elapsed
```

Cần `FINETUNE_PREPROCESS_JOB_OK`, COMPLETED 0:0. `--resume` kiểm và bỏ qua episode
hoàn chỉnh. Nếu job bị dừng giữa episode, thư mục `.partial` còn lại sẽ làm script
dừng: xem ERROR.json; giữ nó để chẩn đoán, chuyển episode partial đó ra khỏi processed
root rồi đăng ký lại sau khi xác định nguyên nhân. Không chạy hai job cùng ghi root.
Mất mạng ở laptop không dừng job sbatch.

Nếu muốn lưu processed trên scratch, thêm `--output-root /duong/dan/scratch/...` vào
preprocess và `--processed-root` vào check/train; dùng cùng path khi resume. Config
run cũng có trường processed_root để khóa vị trí chính thức. Phải đổi trước khi train.

## 7. Normalization và bắt đầu fine-tune

Run này kế thừa chính numeric mean/std của **CFG5 train** trong best.pt epoch58.
Không fit lại state/action mean/std trên test/val, không đổi thang đầu vào/đầu ra
khi giữ trọng số pretrain. Statistics origin ghi rõ inherited_from, source hash/epoch;
episodes=[] nghĩa là chưa fit trên episode task đích, không phải thiếu demo đã xử lý.
Nếu đổi sang target-train stats cần migration weights và thí nghiệm riêng.

Fine-tune nạp model weights, tạo optimizer mới và bắt đầu epoch 1 của task đích.
Không `--resume` best.pt của CFG5: data/split/target semantics đã đổi. Run mới batch8,
DataLoader workers4, GPU1, CPU8, RAM64GiB, CUDA prefetch, LR1e-5, 20 epoch ban đầu.
Validation chọn best; chưa dùng test để chỉnh hyperparameters. 20 chỉ là ngân sách
khởi đầu, chưa bảo đảm hội tụ/success rate.

Sau smoke và full preprocessing thành công, tự đăng ký train:

```bash
T=$(sbatch --parsable scripts/slurm/train_pick_place_finetune.sbatch)
echo "Fine-tune job: $T"
squeue -j "$T" -o "%.18i %.20j %.2t %.10M %R"
tail -n 20 "$HOME/cfg5-pretrain/logs/cfg5-ft-train-$T.out"
```

Script chọn `--init-checkpoint` lần đầu; nếu đã có latest.pt của fine-tune thì dùng
`--resume` của chính run mới. Lưu mỗi epoch: outputs/pick_place_finetune/latest.pt;
best.pt chỉ cập nhật khi val loss tốt hơn; history.json chứa loss/timing từng epoch.
Không thay đổi checkpoint/history của pretrain.

Xem epoch đã lưu:

```bash
python3 - <<'PY'
import json
from pathlib import Path
p=Path('outputs/pick_place_finetune/history.json')
if not p.exists():
    print('Chưa có epoch fine-tune hoàn tất')
else:
    for r in json.loads(p.read_text()):
        seconds=r['train'].get('timing',{}).get('epoch_seconds',0)+r['val'].get('timing',{}).get('epoch_seconds',0)
        print(f"E{r['epoch']+1}: step={r['global_step']}, train={r['train']['loss']:.4f}, val={r['val']['loss']:.4f}, train+val={seconds/60:.1f} phút")
PY
```

Thời gian trên chưa gồm ghi checkpoint và overhead giữa epoch.
Preprocessing test chỉ chạy sau khi khóa checkpoint/config để đánh giá cuối:

```bash
./.venv-data/bin/python -u scripts/prepare_finetune_data.py preprocess --splits test --workers 4 --resume
```

Lệnh này chỉ chuẩn bị test, không đánh giá. Evaluator CFG5 cũ hardcode66 test và
preprocessor RH20T nên không dùng nguyên cho task đích. Cần evaluator task đích/rollout
riêng ở giai đoạn test. Loss offline chưa là tỷ lệ hoàn thành tác vụ MuJoCo.

## 8. Giới hạn kiểm chứng

Đã đọc cả 8 video/timestamps của hai mẫu thật; đã decode 4 tick/mẫu vào smoke memmap,
đối chiếu action với NPZ command. Đã kiểm fixtures về offset0, padding, gap, duplicate
name, seed groups/profile split, timestamp lệch, pickle globals, write guard, fresh
optimizer và resume. Tổng 29 tests đạt, gồm hồi quy data/trainer CFG5 cũ và kiểm bộ
cài checksum/phiên bản, backup, chạy lại và path escape. Đã kiểm cú pháp bash 4 sbatch.
Đã nạp strict
toàn bộ 83.873.929 tham số từ best.pt epoch 58 vào model fine-tune đúng kiến trúc.

Chưa audit toàn bộ hai thư mục server, chưa khóa split thực, chưa chạy full preprocessing
hoặc fine-tune GPU. Số episode/split thực sẽ do job prepare báo, không suy từ hai mẫu.
