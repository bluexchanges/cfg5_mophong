# Fine-tune đọc MP4: split đã khóa, 16 worker, cache RAM

## Mục tiêu và cấu hình

Dùng nguyên manifest/split/normalization đã tạo cho 913 episode hợp lệ. Không chia
lại, không đổi seed chọn split, không đọc ảnh test để chọn checkpoint. Chỉ chuẩn
bị train/val. Khởi tạo weights best pretrain epoch 58, optimizer mới; epoch fine-tune
bắt đầu 1. Số epoch kế thừa config `pick_place_finetune.json` (bản prepare mặc định
20); terminal sẽ in số thực tế. Không phải tiếp tục pretrain từ epoch 78.

Train batch tối đa 8, 16 DataLoader worker; validation 2 worker; prefetch_factor=1.
Slurm xin 1 GPU A100 80GB, 16 logical CPU, RAM 128G. Lần kiểm 14105 cho thấy 16 logical
CPU tương ứng 8 core vật lý với SMT 2; 16 worker không có nghĩa 16 core vật lý.
OpenCV/FFmpeg/BLAS được giới hạn một luồng mỗi worker. Không dùng `srun` vì cluster
đã có lỗi DNS dgx02 ở job 14104; Python chạy trực tiếp trên compute node của batch.

## Thuật toán

1. Từ locked manifest kiểm raw fingerprint, timestamps, trajectory và header video.
2. Tạo `data/processed/pick_place_finetune_mp4/<episode>/` chỉ chứa episode.json,
   state.npy, action.npy, timestamps_ms.npy, segment_ids.npy. Không tạo images.npy.
3. Mỗi epoch tạo thứ tự episode ngẫu nhiên từ generator được lưu trong checkpoint.
   Chia episode cho các worker bằng vị trí modulo số worker. Một episode có đúng
   một worker phụ trách trong epoch đó; phân công đổi theo thứ tự mới ở epoch sau.
4. Worker giải mã tuần tự 8 video của episode một lần, giữ RGB uint8 trong RAM.
   Kiểm từng frame và kiểm không dư frame. Không random seek MP4 cho từng sample.
5. Xáo trộn chỉ số observation trong episode; ACTDataset dùng đúng ảnh/state(t),
   command(t:t+20), padding/gap mask. Không xáo trộn 20 bước bên trong target chunk.
6. Gộp thành batch trong cùng episode; batch cuối episode có thể 1..7 mẫu và được
   giữ lại. Không bỏ/nhân đôi mẫu; tổng sample mỗi epoch bằng tất cả valid samples.
   Số batch có thể lớn hơn `ceil(total_samples/8)` do giữ đuôi từng episode.
7. Giải phóng cache khi hết episode, trước khi cấp phát cache episode tiếp theo.
   Sau epoch worker còn sống nhưng cache được giải phóng. Mỗi epoch giải mã lại.
8. CUDA prefetch/pinned-memory của trainer tiếp tục chuyển batch sang GPU trong
   lúc xử lý batch trước. Chưa có benchmark full server, không bảo đảm GPU luôn 100%.

Cache cap 3.5 GiB/worker, kiểm tất cả train/val metadata trước train. Một episode
500 tick × 8 × 360 × 640 × 3 byte khoảng 2.57 GiB; 16 cache ảnh khoảng 41 GiB,
cap train tối đa 56 GiB. Val2 thêm tối đa7 GiB nếu cache cùng tồn tại; còn bộ nhớ
decoder, Python/PyTorch, batch float32, pinning và prefetch. 128G là request đã được
Slurm chấp nhận; cgroup test chưa thấy limit128G, không suy ra RAM vô hạn. Không
chuyển toàn episode sang float32; chỉ chuyển ảnh của sample khi tạo batch.

Ổ đĩa chỉ tăng numeric arrays/metadata nhỏ, checkpoint và smoke outputs. Raw MP4
phải còn nguyên ở hai thư mục download. Home user08 theo tài liệu: soft1.8TB,
hard2TB; `df`49T free là toàn filesystem, không phải quota cá nhân. `quota` trống
không chứng minh hết hạn mức. Bộ check in estimated MiB, không cần 2TB ảnh decoded.
Video fingerprint dùng size/mtime và SHA các file nhỏ, không phải SHA toàn video.

## Upload từ PowerShell trên máy local

```powershell
scp "F:\cfg5_mophong\finetune_mp4_bundle.zip" user08@172.16.3.8:/home/user08/cfg5-pretrain/
```

## Cài trên terminal server

```bash
set -e
unzip -o "$HOME/cfg5-pretrain/finetune_mp4_bundle.zip" \
  -d "$HOME/cfg5-pretrain/finetune_mp4_bundle"
cd "$HOME/cfg5-pretrain/edabk-physical-ai-robot/04-Src/software/cfg5_act"
./.venv-data/bin/python "$HOME/cfg5-pretrain/finetune_mp4_bundle/install_mp4.py" \
  --repo "$PWD" --dry-run
./.venv-data/bin/python "$HOME/cfg5-pretrain/finetune_mp4_bundle/install_mp4.py" \
  --repo "$PWD"
mkdir -p "$HOME/cfg5-pretrain/logs"
```

Bộ cài từ chối code server khác baseline hoặc có job ACT/fine-tune đang chạy/chờ.
Sao lưu hai file thay đổi trong outputs/install_backups/mp4_...; thêm module,
script, tests và guide. Không sửa upstream/model, raw, locked JSON, checkpoint cũ.
Không chạy full NPY preprocessing của bộ preprocess16 trước đó.

## Đăng ký check và tự train sau khi check thành công

```bash
if C=$(sbatch --parsable scripts/slurm/check_pick_place_mp4.sbatch); then
  C=${C%%;*}
  echo "Job check MP4: $C"
  if T=$(sbatch --parsable --dependency="afterok:$C" scripts/slurm/train_pick_place_mp4.sbatch); then
    T=${T%%;*}
    printf '%s\n' "$C" > "$HOME/cfg5-pretrain/mp4_check.jobid"
    printf '%s\n' "$T" > "$HOME/cfg5-pretrain/mp4_train.jobid"
    echo "Job train MP4: $T (chỉ chạy khi check thành công)"
    squeue -j "$C,$T" -o "%.18i %.24j %.2t %.10M %R"
  fi
fi
```

Job check chuẩn bị metadata train/val (resume bỏ qua file đã kiểm), kiểm locked
hashes, chạy GPU smoke 1trainbatch+1valbatch dùng 1worker ở output riêng theo jobid,
rồi benchmark loader16worker100batch. Smoke không chọn best chính thức; metadata
full không bị giới hạn tick. Check thành công có `MP4_CHECK_JOB_OK`. Job train chỉ
bắt đầu khi check ExitCode0. Có hai job được gửi nhưng không chạy đồng thời. Nếu
check thất bại, train bị chặn; đọc .err và hủy job phụ thuộc trước khi gửi lại.

```bash
C=$(cat "$HOME/cfg5-pretrain/mp4_check.jobid")
T=$(cat "$HOME/cfg5-pretrain/mp4_train.jobid")
tail -n 30 "$HOME/cfg5-pretrain/logs/cfg5-ft-mp4-check-$C.out"
tail -n 40 "$HOME/cfg5-pretrain/logs/cfg5-ft-mp4-check-$C.err"
sacct -j "$C,$T" -X --format=JobID,State,ExitCode,Elapsed,AllocCPUS,ReqMem
tail -f "$HOME/cfg5-pretrain/logs/cfg5-ft-mp4-train-$T.out"
```

Ctrl+C chỉ thoát tail, không hủy Slurm job. `.partial` metadata có ERROR.json phải
đọc trước khi xử lý; runner không tự xóa dữ liệu dở. Lỗi nguồn/frame trong train
làm dừng job, không bỏ mẫu lỗi hoặc tự chọn lại split.

## Theo dõi epoch, batch và tốc độ

Log có `MP4_CACHE worker=... episode=... decode_s=...`, `MP4_LOADER` tổng sample,
số batch và worker. Train log100batch in data_wait_ms, step_ms, samples_per_s;
những số này có H2D/compute/scalar synchronization, không phải kernel-only.
Benchmark loader-only chưa phản ánh throughput train hoặc toàn epoch. Nếu GPU
vẫn đợi, đo data_wait/RAM/I/O thật; 16worker trên8physicalcore chưa chắc nhanh hơn8.

```bash
./.venv-data/bin/python - <<'PY'
import json
from pathlib import Path
run=json.loads(Path('configs/act/pick_place_finetune_mp4_w16.json').read_text())
print('Batch tối đa:',run['batch_size'],'train worker:',run['num_workers'],
      'val worker:',run['val_num_workers'],'epoch mục tiêu:',run['epochs'])
p=Path(run['output_dir'])/'history.json'
if not p.exists():
    print('Chưa lưu xong epoch đầu; xem log train để biết batch đang chạy.')
else:
    rows=json.loads(p.read_text())
    for r in rows[-5:]:
        a=r['train']['timing']['epoch_seconds']; b=r['val']['timing']['epoch_seconds']
        print(f"Epoch {r['epoch']+1}: train {a/60:.1f} phút, val {b/60:.1f} phút, "
              f"tổng compute {(a+b)/60:.1f} phút; train loss={r['train']['loss']:.5f}, val={r['val']['loss']:.5f}")
    print('Tổng trên chưa gồm thời gian ghi checkpoint.')
PY
```

Checkpoint mới `outputs/pick_place_finetune_mp4_w16/best.pt` (best validation),
latest.pt (model+optimizer+RNG để resume), history.json; lưu mỗi epoch. Nếu job hết
walltime/ngắt, đăng ký lại train sbatch: tự resume latest của chính run này, chạy
lại phần epoch chưa hoàn thành. Không sửa batch/workers/config giữa resume cùng
run; nếu cần đổi phải fork run có provenance. Không resume checkpoint smoke.

## Kiểm chứng local

Fixtures video32×32, split/gap hợp lệ, so sánh toàn ảnh/state/command/padding với
NPY, cache uint8/giới hạn, nguồn đổi, numeric corruption, partition episode,
spawn2worker persistent và thứ tự epoch khi resume. Trainer cũ vẫn mặc định NPY.
Chưa train/benchmark16worker trên server trong lúc tạo bundle; cần job check ở trên.
