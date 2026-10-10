# Giảm chờ dữ liệu: batch 4, workers 2, tổng 40 epoch

## Căn cứ và thay đổi

Ngày 2026-10-04, người dùng đo job ACT batch 4 trên A100 80 GB: GPU utilization
0–93%, CPU tiến trình chính khoảng 100–119% (khoảng 1–1,2 lõi), VRAM 10.437
MiB. Đoạn cuối GPU hoạt động cao hơn đoạn đầu. Đây là dấu hiệu có thời gian
chờ phía CPU/I/O hoặc đồng bộ, chưa chứng minh nguyên nhân duy nhất là loader.

Theo yêu cầu giảm chờ, thử `num_workers=2` với batch size 4. DataLoader dùng
spawn, `prefetch_factor=2` (mỗi worker chuẩn bị trước 2 batch),
`persistent_workers=True` khi workers > 0; workers=0 vẫn không nhận các
tham số multiprocessing. Giữ pin_memory và non_blocking transfer đã có.
Job cấp 4 CPU/64 GB RAM/1 GPU, giới hạn thread OMP/MKL/OpenBLAS=1 để giảm
oversubscription. Không đổi model, LR, precision, ảnh, split, stats hoặc raw.

PyTorch mô tả tải dữ liệu song song và prefetch trong
[tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html).
Chưa benchmark full CFG5 workers2 trên server. Prefetch không bảo đảm mọi
khoảng chờ biến mất: có thể còn NFS, host synchronization, lưu checkpoint
hoặc chuyển phase. Không đặt mục tiêu GPU utilization 100% làm tiêu chí duy
nhất; ưu tiên samples/s và thời gian epoch.

## Chuyển từ job batch 4 đang chạy

Upload `act_workers2_to40.zip` vào `/home/user08/cfg5-pretrain/`. Chạy trên SSH:

```bash
unzip -n "$HOME/cfg5-pretrain/act_workers2_to40.zip" -d "$HOME/cfg5-pretrain/act_workers2_to40_bundle"
nohup python3 -u "$HOME/cfg5-pretrain/act_workers2_to40_bundle/scripts/slurm/switch_to_workers.py" >> "$HOME/cfg5-pretrain/switch_workers2.log" 2>&1 < /dev/null &
tail -n 12 "$HOME/cfg5-pretrain/switch_workers2.log"
```

Watcher chỉ đọc queue/history bằng stdlib trên head node. Nó nhận đúng job
`cfg5-act-bs4-to40` đang R của account đang đăng nhập, kiểm config nguồn lưu
mỗi epoch, chờ history của **epoch tiếp theo** (gồm train+val và latest đã
ghi), rồi scancel có filter user/name/jobID. Đợi job cũ rời queue trước khi
submit `cfg5-act-bs4-w2`. Không hủy job khác. Khi source job tự kết thúc,
watcher dùng checkpoint cuối đã lưu. Có khóa để tránh hai watcher; nếu job
workers2 đã chạy/chờ thì không gửi trùng. Nếu không có job ACT, có thể gửi
ngay từ checkpoint batch 4 đã lưu. Khi đã đạt 40 epoch thì không chạy thêm.

Code chỉ được cài trong compute job mới, sau khi job cũ kết thúc. Installer
kiểm SHA256/cú pháp, backup và ghi atomic+fsync+đọc kiểm lại để phát hiện lỗi
byte NUL hoặc copy. Fork cho phép đổi num_workers cùng epochs/batch/output;
giữ model, optimizer moments, global_step, RNG và lineage. Hash contract và
resume thường vẫn kiểm nghiêm. Parent `outputs/act_pretrain_bs4` được giữ,
child `outputs/act_pretrain_bs4_w2` nhận history/best và chạy tiếp đến **40
epoch tổng cộng**. Ít batch của epoch sau có thể chạy trong khoảng giữa
commit history và scancel; chúng chưa lưu và được chạy lại từ boundary.

## Xem trạng thái và thời gian chờ

```bash
tail -n 12 "$HOME/cfg5-pretrain/switch_workers2.log"
J=$(cat "$HOME/cfg5-pretrain/act_bs4_workers2.jobid")
squeue -j "$J" -o '%.18i %.22j %.2t %.10M %R'
tail -n 15 "$HOME/cfg5-pretrain/logs/cfg5-act-bs4-w2-$J.out"
tail -n 15 "$HOME/cfg5-pretrain/logs/cfg5-act-bs4-w2-$J.err"
bash "$HOME/cfg5-pretrain/act_workers2_to40_bundle/scripts/slurm/monitor_act_job.sh"
```

Chờ watcher báo Submitted trước khi đọc jobid. Log mỗi 100 train/val batch:

```text
train batch=100 samples=400 ... data_wait_ms=... step_ms=... samples_per_s=...
```

- `data_wait_ms`: thời gian host chờ iterator trả batch, trung bình trong cửa
  sổ 100 batch; cửa sổ đầu gồm startup workers.
- `step_ms`: thời gian host từ nhận batch qua H2D, forward/backward/optimizer,
  metric/scalar synchronization. Không phải GPU kernel time thuần.
- `samples_per_s`: throughput của cửa sổ; history thêm timing cả epoch.

Metrics hiện đọc CUDA scalars sau optimizer trên cùng stream nên step interval
bao gồm synchronization sẵn có; không thêm cuda.synchronize mỗi batch.
Data wait thấp hơn nhiều step time là dấu hiệu loader đang cung cấp kịp.
Đo nhiều cửa sổ, đặc biệt qua đổi episode, rồi so với tốc độ trước. Không suy
ra phần trăm GPU idle chính xác từ host timing này.

Spawn/persistent workers thay đổi cách tiêu thụ generator giữa các epoch;
dataset hiện không có augmentation ngẫu nhiên trong worker. Vẫn giữ tập mẫu
và seed, nhưng không tuyên bố bit-for-bit trùng run workers0 hoặc một run
không bị gián đoạn. Nếu hết thời gian 48 giờ, chạy lại watcher sau khi job
kết thúc để resume child latest. Nếu worker báo shared-memory/bus error,
kiểm `/dev/shm` từ log và gửi lỗi, không sửa config rồi resume cùng run.

Kiểm fixture đã bao gồm worker spawn thực tế đọc đủ mẫu qua 2 epoch,
checkpoint/optimizer/RNG fork+resume, chống submit trùng, chỉ cancel khi
history đạt boundary, và từ chối code bundle sai checksum trước khi ghi.
Không có full training, dataset copy hoặc asset download do agent chạy.
