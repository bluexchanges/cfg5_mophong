# Dashboard terminal A100 cho CFG5

Người dùng cung cấp ảnh dashboard `a100top`, nhưng server báo lệnh không có.
Không tìm thấy script tương ứng trong workspace/các thư mục Windows đã khảo
sát. Vì vậy đây là bản dashboard riêng `a100top_cfg5.py`, không phải bản gốc
trong ảnh. Chỉ dùng thư viện chuẩn Python3/Linux và ANSI terminal; không cài
package, không thay code/model/data training, không tải asset.

## Upload và mở

PowerShell Windows:

```powershell
scp "F:\cfg5_mophong\local_source\scripts\slurm\a100top_cfg5.py" user08@172.16.3.8:/home/user08/cfg5-pretrain/
```

Terminal SSH:

```bash
python3 "$HOME/cfg5-pretrain/a100top_cfg5.py"
```

Script chọn một job tên `cfg5-act*` đang R thuộc user hiện tại. Nếu có nhiều
job ACT chạy, chỉ định rõ `--job JOBID`. Nó dùng một srun step bên trong job
đã có, với `--mpi=none --overlap --exact -N1 -n1 -c1 --pty`. Không sbatch hoặc
scancel. Nếu chưa có job ACT đang R, in queue và thoát. Khi job cũ bị dừng
để chuyển sang workers2, monitor cũng kết thúc; mở lại để chọn job mới.

Phím `q`: thoát chỉ monitor; `p`: hiện/ẩn bảng process; `+/-`: giảm/tăng chu
kỳ refresh (tối thiểu3s). Queue cập nhật mỗi15s, tìm PID theo scontrol mỗi30s;
GPU/CPU và log/history theo chu kỳ màn hình. Nếu terminal thấp, dùng `p` để
đủ chỗ hiển thị. Alias tùy chọn, hiệu lực trong phiên SSH hiện tại:

```bash
alias a100top='python3 "$HOME/cfg5-pretrain/a100top_cfg5.py"'
a100top
```

## Cách đọc số liệu

- Job list: R/PD, node/reason, elapsed và TIME_LEFT là giới hạn Slurm, không
  phải ETA training.
- Epoch/batch/samples đọc tail log tối đa64KiB và history nhỏ. Epoch hiển
  thị theo1-index; “đã lưu” chỉ lấy từ history sau checkpoint. Tiến độ là
  lower bound do log mỗi100 batch; train/val/lưu checkpoint là phase khác.
- ETA chỉ hiện cho phase hiện tại khi log có samples_per_s từ trainer mới.
  Không đoán thời gian tổng40 epoch hoặc thời gian còn lại từ TIME_LEFT.
- GPU chọn UUID từ PID train thuộc job, không lấy nhầm GPU0 của job khác.
  Utilization/VRAM là toàn card đó (nếu card được chia sẻ thì có cả ứng dụng
  khác), không phải VRAM PyTorch allocated riêng. Không load torch/model.
- CPU đo delta ticks / thời gian từ `/proc`, train và tiến trình con.
  100% ≈ một lõi; thanh CPU chia theo NumCPUs được cấp cho job. CPU phần
  trăm của mỗi process không phải mức dùng toàn server.
- RAM là RSS main, không phải tổng bộ nhớ riêng của job hoặc tổng worker
  RSS (có vùng shared nên không cộng thành unique RAM).
- Loss latest/best lấy từ history đã lưu, không phải loss của batch đang
  chạy. Kết quả là offline loss.

Đã kiểm6 fixture cho parser, epoch/phase, batch4/58.313 mẫu=14.579 batch,
timing, config/log path, GPU UUID theo PID, CPU delta và giới hạn chiều rộng
render. CLI help đã kiểm trên Windows. Chưa xác nhận live TTY/Slurm/server
trong môi trường agent; cần người dùng mở trên SSH. Xử lý SIGTERM/terminal
restore có sẵn để đóng step mà không để terminal ẩn con trỏ.

Nguồn giao diện Slurm:
[srun](https://slurm.schedmd.com/srun.html),
[NVIDIA monitoring](https://docs.nvidia.com/deploy/nvidia-smi/).
