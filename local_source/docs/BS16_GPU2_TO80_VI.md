# Chạy tiếp đến 80 epoch: batch 16, 8 worker, 2 GPU

## Cấu hình đã chuẩn bị

| Thành phần | Giá trị |
|---|---|
| Epoch mục tiêu | Tổng 80; tiếp từ checkpoint đã lưu ít nhất 40 |
| Batch size | **16 mẫu tổng**, khoảng 8 mẫu mỗi GPU |
| GPU | Xin 2 GPU cùng node trong **một job** |
| CPU | Xin 16 CPU qua Slurm |
| DataLoader | 8 worker, spawn, persistent, prefetch 2/worker |
| RAM CPU | Xin 128 GiB; khác với VRAM GPU |
| Precision/LR | Giữ FP32 và learning rate từ config nguồn |
| Checkpoint | Mỗi epoch; latest giữ optimizer/RNG/global_step |
| Output mới | `outputs/act_pretrain_bs16_w8_gpu2` |

User báo đã xong 40 epoch của `outputs/act_pretrain_fast`. Script kiểm checkpoint thật trong compute job, không đọc file 1 GB trên head node. Đổi batch/worker/GPU là thay phương pháp chạy: dùng fork có metadata và output mới. History và best cũ được kế thừa; best có thể thuộc epoch trước khi đổi batch. Không dùng test để chọn cấu hình.

## 1. Đưa gói lên server

Trong **PowerShell Windows**:

```powershell
scp "F:\cfg5_mophong\act_bs16_gpu2_to80_bundle.zip" user08@172.16.3.8:/home/user08/cfg5-pretrain/
```

Upload `act_bs16_gpu2_to80_bundle.zip` vào `/home/user08/cfg5-pretrain/` bằng WinSCP. Trên terminal SSH Linux chạy:

```bash
unzip -n "$HOME/cfg5-pretrain/act_bs16_gpu2_to80_bundle.zip" \
  -d "$HOME/cfg5-pretrain/act_bs16_gpu2_to80_bundle"
python3 "$HOME/cfg5-pretrain/act_bs16_gpu2_to80_bundle/scripts/slurm/submit_bs16_gpu2.py"
```

Launcher tạo `$HOME/cfg5-pretrain/act_bs16_gpu2_to80.sbatch`, đăng ký bằng sbatch và ghi `act_bs16_gpu2_to80.jobid`. Nó từ chối gửi thêm nếu thấy job `cfg5-act*` đang chạy/chờ hoặc controller cũ đang giữ lock. Không hủy job nào. Slurm có thể để PD chờ hai GPU hoặc giới hạn tài khoản; CPU/GPU chỉ được cấp khi job R. [Tài liệu sbatch](https://slurm.schedmd.com/sbatch.html).

Trong compute job: kiểm thấy đúng 2 GPU → kiểm checksum/cú pháp và backup code → test hai GPU với model nhỏ → probe **3 batch full ACT thật** trên bản sao trạng thái, không lưu cập nhật probe → fork/resume → train dài. Probe lỗi/OOM thì job dừng trước khi tạo checkpoint nhánh mới. Source checkpoint vẫn còn. Gate phần cứng chưa chạy ở máy phát triển chỉ có PyTorch CPU.

## 2. Xem queue, batch và log

```bash
squeue -u "$USER" -o "%.18i %.25j %.2t %.10M %R"
J=$(cat "$HOME/cfg5-pretrain/act_bs16_gpu2_to80.jobid")
tail -n 12 "$HOME/cfg5-pretrain/logs/cfg5-act-bs16-g2-to80-$J.out"
tail -n 20 "$HOME/cfg5-pretrain/logs/cfg5-act-bs16-g2-to80-$J.err"
```

Xem epoch đã lưu, epoch/batch trong log, loss cuối và thời gian từng epoch:

```bash
python3 "$HOME/cfg5-pretrain/act_bs16_gpu2_to80_bundle/scripts/slurm/status_to80.py"
```

Train có 58.313 sample nên batch 16 có `ceil(58313/16)=3645` batch/epoch; batch cuối 9 mẫu, chia 5+4. Val có 6.214 sample nên 389 batch; cuối 6 mẫu. Mỗi batch train gộp gradient rồi cập nhật optimizer **một lần**. Worker chỉ chuẩn bị dữ liệu.

JSON `epoch=40` nghĩa là **epoch thứ 41**. `history.json` ghi sau checkpoint; dòng JSON trong log có thể xuất hiện trước khi file save xong. Status ghi rõ “đã lưu” và tiến độ trong log. Timer train+val không bao gồm ghi checkpoint. Timing 3 batch probe gồm khởi động worker, không đủ để kết luận tốc độ ổn định; xem thêm nhiều cửa sổ `samples_per_s` trong lượt thật.

Dashboard có sẵn:

```bash
bash /home/user08/phuc/tools/a100top/a100top.sh
```

Job kết thúc/timeout: kiểm `sacct -j "$J" -X --format=JobID,State,ExitCode,Elapsed`, kiểm err, rồi gửi lại bằng launcher trên. Nhánh có latest thì resume tại epoch đã lưu. Không chạy hai launcher/job ACT cùng lúc. Nếu fork bị ngắt trước latest, kiểm output lỗi/partial trước khi xử lý; script không tự xóa nhánh dở.

## 3. Hai GPU hoạt động thế nào?

Trainer thêm adapter `ParallelACT` dùng `DataParallel`, chia tensor theo chiều batch. Model gốc và upstream giữ nguyên. L1 gộp theo **số phần tử action thật**, KL theo **số mẫu**, tránh lấy trung bình hai loss sai khi số padding/size shard khác nhau. Checkpoint giữ keys model thường, không thêm prefix `module.`; optimizer vẫn trỏ cùng parameters gốc.

Phương án này giữ kiến trúc trainer một process để tiếp quản checkpoint hiện có. GPU 0 chứa optimizer và điểm gộp, đồng thời CUDA prefetch hiện tại chuyển batch lên GPU 0 rồi scatter. Hai GPU có thể nhanh hơn nhưng có chi phí replicate/scatter/gather; không cam kết nhanh gấp đôi. PyTorch khuyến nghị **DistributedDataParallel (DDP)** cho hiệu suất multi-GPU; đó là bước triển khai tiếp nếu phép đo cần tối ưu hơn. [DataParallel 2.7](https://docs.pytorch.org/docs/2.7/generated/torch.nn.DataParallel.html).

Giữ RNG cũ ở GPU 0; GPU thêm được seed theo seed run. Thay batch/số GPU thay nhóm mẫu và stochastic path, nên đây là nhánh thí nghiệm có ghi provenance, không phải tái lập bit-for-bit đường cũ. Giữ LR hiện có trước để theo dõi val; không tự nhân LR theo số GPU.

## 4. Học đa nhân

Đọc [MULTICORE_LEARNING_VI.md](MULTICORE_LEARNING_VI.md): phân biệt core/process/thread/worker, hiểu queue dữ liệu và GPU, rồi tập đo nút thắt trước khi tăng tài nguyên.

## 5. Vẽ lại loss, chỉ đọc history

Chạy trên server (có thể chạy song song training):

```bash
python3 "$HOME/cfg5-pretrain/act_bs16_gpu2_to80_bundle/scripts/plot_training_loss.py" \
  --repo "$HOME/cfg5-pretrain/edabk-physical-ai-robot/04-Src/software/cfg5_act"
```

Script tạo `docs/audit/training_plots/loss_train_val.svg`, `loss_train.svg`, `loss_val.svg`. Không cần cài matplotlib. Mỗi điểm là loss tổng kết một epoch đã lưu; epoch đang chạy chưa xuất hiện. Chọn history nhánh BS16 nếu có, nếu chưa có chọn fast40; history fork giữ các epoch nguồn. Vạch đứt đánh dấu chỗ đổi batch. Chart riêng tự đặt trục Y, chart chung có chung trục Y; không làm mượt số liệu.

Trong PowerShell Windows, tải cả ba đồ thị về:

```powershell
New-Item -ItemType Directory -Force "F:\cfg5_mophong\loss_plots" | Out-Null
scp "user08@172.16.3.8:/home/user08/cfg5-pretrain/edabk-physical-ai-robot/04-Src/software/cfg5_act/docs/audit/training_plots/loss_*.svg" "F:/cfg5_mophong/loss_plots/"
```

Mở `.svg` bằng Edge/Chrome. Chạy lại lệnh vẽ và tải để cập nhật. Chỉ truy cập JSON nhỏ và ghi ảnh; không đụng checkpoint, optimizer hoặc training process.
