# Tăng batch, thử thêm worker và tiếp tục fine-tune từ checkpoint

## Bản này làm gì

Job14111 đang batch8/worker16, VRAM19.4/80GB, RAM57.9/128GB. Log đã gửi có
data_wait0.5–175.2ms nhưng step385.4–1499.8ms: chưa đủ bằng chứng decode MP4 là
nút thắt chính. Step gồm H2D/CPU dispatch/GPU compute/scalar synchronization.

- Thử batch16 và32; batch là số observation xử lý cùng lúc. Mỗi observation có
  ảnh8camera + state8D + chunk20action8D + padding mask.
- Thử worker8/16/24; worker là process đọc/decode/lắp batch. Job vẫn xin16CPU
  logic,128G RAM,1A10080GB. Worker24 không tạo thêm CPU được cấp.
- Chuyển ảnh dạng RGBuint8CHW rồi normalize float32 trên GPU: lượng byte ảnh H2D
  giảm4lần so với float32 cùng batch. Batch8 khoảng42.19MiB thay168.75MiB;
  batch16 khoảng84.38MiB; batch32 khoảng168.75MiB (640×360×8camera).
- Lấy batch và H2D ở một thread nền, copy stream riêng; compute stream chờ event
  của batch hiện tại. Đường cũ chờ lấy batch sau trước khi giao batch hiện tại.
- Cache vẫn decode tuần tự8MP4/episode, giữ RGBuint8 trong RAM, không tạo images.npy.
  Max cache hiện2.601GiB/worker;24cache khoảng62.4GiB. Tổng RAM còn có worker,
  pinned queue, checkpoint/model và cache validation. Số này chưa phải peak RAM.

Giữ split731train/91val/91test, observation10Hz,8camera, action/normalization và
model. Không xử lý ảnh test. Đổi batch đổi số optimizersteps/epoch; LR giữ nguyên,
cần tiếp tục theo dõi validation. Đây là thí nghiệm runtime mới có output riêng.

## 1. Upload ở PowerShell máy local

```powershell
scp "F:\cfg5_mophong\finetune_mp4_fast_bundle.zip" user08@172.16.3.8:/home/user08/cfg5-pretrain/
```

## 2. Server: chờ epoch hiện tại lưu xong rồi tự chuyển

Không cài đè core source khi job hiện tại còn chạy. Installer kiểm squeue và
baseline/checksum, sao lưu code cũ; không ghi đè custom code khác phiên bản.

```bash
unzip -n "$HOME/cfg5-pretrain/finetune_mp4_fast_bundle.zip" \
  -d "$HOME/cfg5-pretrain/finetune_mp4_fast_bundle"

nohup python3 -u "$HOME/cfg5-pretrain/finetune_mp4_fast_bundle/switch_mp4_fast.py" \
  --job 14111 \
  --repo "$HOME/cfg5-pretrain/edabk-physical-ai-robot/04-Src/software/cfg5_act" \
  >> "$HOME/cfg5-pretrain/switch_mp4_fast.log" 2>&1 < /dev/null &

tail -f "$HOME/cfg5-pretrain/switch_mp4_fast.log"
```

Watcher chạy nhẹ trên head node: chỉ đọc JSON và gọi Slurm, không load model.
Nó ghi số epoch đã lưu lúc bắt đầu; chờ thêm1epoch hoàn chỉnh (train+val và
latest.pt/history đã ghi), cancel đúng job/user/name, chờ job giải phóng, cài code,
submit benchmark, rồi submit train với dependency afterok. Chỉ1job chạy cùng lúc.
Mất kết nối SSH không làm dừng watcher/job sbatch. Ctrl+C tại tail chỉ dừng xem log.

Nếu đã có checkpoint và chấp nhận bỏ phần epoch hiện tại chưa lưu, có thể thêm
`--use-saved-epoch` vào lệnh watcher. Không dùng tùy chọn này khi chưa có latest.pt.
Không chạy đồng thời hai watcher. Nếu job nguồn đã kết thúc, dùng cách thủ công dưới.

## 3. Hai job mới

### Job benchmark/check, không phải train20epoch

`scripts/slurm/benchmark_pick_place_mp4_fast.sbatch` xin CPU16/RAM128G/GPU1,3giờ.
Chạy gate CUDA (bắt buộc GPU thật), kiểm data contract, sau đó8case riêng process:

| Batch | Worker | Đường dữ liệu |
|---|---|---|
|8|16|baseline float32 + prefetch cũ|
|8|16|uint8 + thread nền|
|16|8/16/24|uint8 + thread nền|
|32|8/16/24|uint8 + thread nền|

Mỗi case8warmup và ít nhất160batch đo. Tự tăng số batch đến
`worker × ceil(max_episode_samples / batch) + 80` để đi qua đợt đổi episode
đầu tiên của loader; baseline8/16 khoảng1100batch,32/24 khoảng464batch. Vì thế
benchmark có thể mất1–3giờ tùy tốc độ thực, không phải vài giây. Cùng seed và
checkpoint nguồn. Chỉ đọc train. Có
forward/backward/optimizer tạm để đo đúng chi phí train; weights và optimizer
benchmark không lưu hoặc chuyển sang run thật. CUDA/RAM workers được giải phóng
giữa case. Batch OOM bị đánh dấu failed; case còn lại tiếp tục trong process mới.

Report đo samples/s toàn pipeline, wait_ms, step_ms, peak allocated/reserved VRAM,
startup và step với batch giữ sẵn GPU sau khi dừng worker. Đo resident vẫn gồm
CPU dispatch/optimizer/sync, không gọi là thời gian kernel thuần. Warm filesystem
và subset ngắn có thể ảnh hưởng so sánh; xác nhận lại qua log epoch đầy đủ.

Theo yêu cầu tăng cả batch lẫn worker, chọn giữa batch16/worker24 và
batch32/worker24: case nhanh nhất có peak_reserved≤80%VRAM tổng và nhanh hơn
baseline8/16. Các mức worker8/16 được đo để đối chiếu; report còn ghi `best_any`
để biết tăng worker có thực sự tốt hơn mọi mức khác không. Nếu không có cấu hình
tăng cả hai đạt yêu cầu, job FAIL và train dependency bị hủy. Report ở
`outputs/mp4_fast_benchmark/job_JOBID/report.json`.

### Job train thật

`scripts/slurm/train_pick_place_mp4_fast.sbatch`: CPU16/RAM128G/GPU1,72giờ.
Config tự tạo sau benchmark: `configs/act/pick_place_finetune_mp4_fast.json`.
Output mới: `outputs/pick_place_finetune_mp4_fast`.

Trước train, fork checkpoint từ `outputs/pick_place_finetune_mp4_w16/latest.pt`
qua config mới, giữ model, optimizer moments, RNG, epoch, history và best; ghi
runtime changes/provenance và signature mới. Không chỉnh checkpoint cũ. Train
tiếp epoch kế tiếp, tới tổng20epoch như mục tiêu cũ. Không quay lại pretrain58.
Val giữ2worker. Train đã fork thì lần sau đăng ký lại job train để --resume;
không chạy lại benchmark vào output đã tồn tại. Walltime72giờ vẫn có thể chưa đủ
20epoch; resume từ epoch đã lưu, phần epoch bị ngắt phải chạy lại.

## 4. Check tiến độ

```bash
B=$(cat "$HOME/cfg5-pretrain/mp4_fast_bench.jobid")
T=$(cat "$HOME/cfg5-pretrain/mp4_fast_train.jobid")
squeue -j "$B,$T" -o "%.18i %.25j %.2t %.10M %R"
tail -n 25 "$HOME/cfg5-pretrain/logs/cfg5-ft-mp4-bench-$B.out"
tail -n 25 "$HOME/cfg5-pretrain/logs/cfg5-ft-mp4-bench-$B.err"

# Khi benchmark đã đạt và train chạy:
tail -f "$HOME/cfg5-pretrain/logs/cfg5-ft-mp4-fast-$T.out"
bash "$HOME/phuc/tools/a100top/a100top.sh"
```

```bash
cd "$HOME/cfg5-pretrain/edabk-physical-ai-robot/04-Src/software/cfg5_act"
python3 -m json.tool "outputs/mp4_fast_benchmark/job_$B/report.json"
python3 -m json.tool configs/act/pick_place_finetune_mp4_fast.json
sacct -j "$B,$T" -X --format=JobID,State,ExitCode,Elapsed
```

Không suy ra throughput từ một snapshot GPU0%. So samples/s sau startup và qua
nhiều lần đổi episode; VRAM tăng chỉ hữu ích khi tốc độ/learning phù hợp.

## 5. Nếu job cũ đã kết thúc: cài và submit thủ công

Chỉ tiếp tục nếu latest.pt full fine-tune tồn tại; không cancel giữa epoch nếu
muốn giữ phần đang chạy. Không chạy watcher đồng thời với các lệnh này.

```bash
cd "$HOME/cfg5-pretrain/edabk-physical-ai-robot/04-Src/software/cfg5_act"
./.venv-data/bin/python "$HOME/cfg5-pretrain/finetune_mp4_fast_bundle/install_fast.py" --repo "$PWD"
if B=$(sbatch --parsable scripts/slurm/benchmark_pick_place_mp4_fast.sbatch); then
  echo "$B" > "$HOME/cfg5-pretrain/mp4_fast_bench.jobid"
  if T=$(sbatch --parsable --dependency="afterok:$B" --kill-on-invalid-dep=yes scripts/slurm/train_pick_place_mp4_fast.sbatch); then
    echo "$T" > "$HOME/cfg5-pretrain/mp4_fast_train.jobid"
  fi
fi
```

Resume sau timeout/completed checkpoint nhưng chưa đủ20:

```bash
cd "$HOME/cfg5-pretrain/edabk-physical-ai-robot/04-Src/software/cfg5_act"
if T=$(sbatch --parsable scripts/slurm/train_pick_place_mp4_fast.sbatch); then
  echo "$T" > "$HOME/cfg5-pretrain/mp4_fast_train.jobid"
fi
```

## Kiểm chứng

Local dùng PyTorchCPU, kiểm fixture nhỏ, regression/checkpoint và mock delayed
prefetch; GPU gate phải chạy server trước train thật. Không tuyên bố speedup
hoặc batch32fit80GB trước report thật. Chưa upload/đăng ký server tự động do SSH
BatchMode thiếu xác thực; các lệnh ở trên dành cho người dùng chạy. Không push
GitHub trong bước này.

Nguồn về pinned/nonblocking transfer và CUDA streams:
https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html
