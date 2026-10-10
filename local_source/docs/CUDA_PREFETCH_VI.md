# Chuyển batch kế tiếp sang GPU trong lúc tính batch hiện tại

## Căn cứ và giới hạn

Người dùng đo batch4 thật: 84,38 MiB, pinned memory cho cả bốn tensor,
CPU → GPU trung vị 4,45 ms. Log workers2 có data_wait_ms 0,2–0,8 và
step_ms khoảng244. Nút đọc dữ liệu đã nhỏ; H2D đo riêng tương đương1,8%
của một bước nếu đại diện cho training. Không hứa GPU hết khoảng nghỉ
hoặc tốc độ tăng rõ; scalar synchronization/launch/compute vẫn có thể tạo khoảng nghỉ.

Trainer bổ sung cờ boolean `cuda_prefetch`, mặc định false cho config cũ.
Khi true, một batch được chuyển trước trên stream riêng, trong lúc compute
stream chạy batch hiện tại. Cần RAM pinned; compute chờ copy đúng batch,
GPU tensors record_stream để tránh allocator tái sử dụng sớm. Giữ CPU nguồn
đến khi copy hoàn tất; giới hạn hai batch copy đang chờ, đóng stream khi
kết thúc/exception. Không sửa raw, upstream, loss, optimizer, LR, precision,
sample order hoặc số batch; smoke giới hạn iterator trước prefetch.

Nguồn: [PyTorch về pin_memory, non_blocking và stream riêng](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html).

## Kiểm đã thực hiện

CPU fixture kiểm train/validation, batch cuối, migration bảo toàn checkpoint,
chuyển job sau history đã commit và không gửi trùng. Máy phát triển chỉ có
torch CPU: ba kiểm CUDA chưa chạy. Batch script bắt buộc chạy kiểm CUDA
nhỏ trên GPU trước khi cài trainer hoặc train tiếp. Kiểm CUDA so mẫu/tensor,
trọng số/optimizer với serial, giới hạn smoke và đóng iterator sớm. Nếu kiểm
thất bại, job mới dừng; checkpoint parent vẫn có thể resume. Chưa đo overlap
thực tế bằng profiler hoặc throughput full ACT của nhánh này.

## Triển khai

Bundle: `F:\cfg5_mophong\act_cuda_prefetch_to40.zip`. Mở PowerShell Windows:

```powershell
scp F:\cfg5_mophong\act_cuda_prefetch_to40.zip user08@172.16.3.8:~/cfg5-pretrain/
```

Terminal SSH server:

```bash
unzip -n "$HOME/cfg5-pretrain/act_cuda_prefetch_to40.zip" -d "$HOME/cfg5-pretrain/act_cuda_prefetch_bundle"
nohup python3 -u "$HOME/cfg5-pretrain/act_cuda_prefetch_bundle/scripts/slurm/switch_to_cuda_prefetch.py" >> "$HOME/cfg5-pretrain/switch_cuda_prefetch.log" 2>&1 < /dev/null &
tail -n 12 "$HOME/cfg5-pretrain/switch_cuda_prefetch.log"
```

Watcher chỉ nhận đúng một job own-user tên `cfg5-act-bs4-w2` đang R.
Nó chờ history thêm một epoch (được ghi sau latest.pt với save_every_epochs=1),
dừng job đó, chờ rời queue rồi gửi `cfg5-act-prefetch`. Không hủy T27.
Có thể bỏ vài batch đã bắt đầu của epoch kế tiếp trong thời gian watcher poll;
resume từ epoch hoàn chỉnh đã lưu. Nếu job cũ kết thúc tự nhiên, dùng checkpoint
cuối đã lưu. Khóa watcher và kiểm queue ngăn gửi trùng. Không chạy hai trainer
ACT cùng lúc. Không sửa trainer của job đang R.

Job mới kiểm CUDA, cài ba source đã kiểm checksum/cú pháp với backup,
fork `outputs/act_pretrain_bs4_w2/latest.pt` sang `outputs/act_pretrain_prefetch/`,
giữ AdamW/RNG/epoch/global_step/history/best, chỉ bật cuda_prefetch và output
riêng. Giữ batch4/workers2, tổng40 epoch,4CPU/64GB/1GPU/48h. Nếu nhánh con
đã có latest.pt thì resume nhánh con. Config nằm ở
`configs/act/cfg5_all_pretrain_prefetch_40epoch.json`.

```bash
squeue -u "$USER" -o "%.18i %.25j %.2t %.10M %R"
tail -n 12 "$HOME/cfg5-pretrain/switch_cuda_prefetch.log"
```

Sau khi có job mới:

```bash
JOB=$(cat "$HOME/cfg5-pretrain/act_cuda_prefetch.jobid")
tail -n 15 "$HOME/cfg5-pretrain/logs/cfg5-act-prefetch-$JOB.out"
tail -n 20 "$HOME/cfg5-pretrain/logs/cfg5-act-prefetch-$JOB.err"
```

Startup phải có `cuda_prefetch=True`. So `samples_per_s` nhiều cửa sổ với
mức trước khoảng16,4, rồi so thời gian cả epoch. `data_wait_ms` khi bật cờ
bao gồm staging batch kế tiếp; `step_ms` gồm compute và chờ copy batch hiện
tại, có overlap giữa stream nên không dùng hai số này để tách H2D/kernel.
Muốn xác nhận overlap thực tế cần CUDA profiler trace. CPU/GPU usage dashboard
phải chọn đúng job ACT.

Nếu gate CUDA thất bại, đọc `.err`; chưa cài trainer mới. Để quay về workers2,
dùng script workers2 đã có khi không còn job ACT khác. Parent và best chưa bị
ghi đè. Không sửa config đang chạy để mong thay đổi stream của tiến trình.
