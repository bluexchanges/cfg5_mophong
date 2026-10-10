# Tiếp tục pretraining với batch size 4 đến tổng 40 epoch

Theo yêu cầu ngày 2026-10-04, dùng batch size 4 trên A100 80 GB. Giữ LR,
optimizer, split, normalization, model, ảnh và num_workers theo checkpoint cũ.
Không tăng workers hoặc bật mixed precision trong lần đổi này. Chưa đo tốc độ
hoặc VRAM của full ACT batch 4 trên server; người dùng chạy job để kiểm thực tế.

## Gửi job

Upload bundle `act_batch4_to40.zip` vào `/home/user08/cfg5-pretrain/`.
Trong terminal SSH:

```bash
unzip -n "$HOME/cfg5-pretrain/act_batch4_to40.zip" -d "$HOME/cfg5-pretrain/act_batch4_to40_bundle"
bash "$HOME/cfg5-pretrain/act_batch4_to40_bundle/scripts/slurm/launch_bs4_to40.sh"
```

Launcher kiểm không có job `cfg5-act*` đang chạy/chờ trước khi cài ba file Python
(backup code cũ) và gửi `cfg5-act-bs4-to40`. Chuẩn bị checkpoint/config diễn ra
trên compute node trong Slurm job. Đọc run config trực tiếp từ checkpoint,
không đoán file config cũ. Fork chỉ cho đổi batch_size, epochs và output_dir;
contract model/data/split/stats/camera vẫn phải trùng. Giữ optimizer AdamW,
RNG, epoch và global_step; copy best model và lịch sử trước checkpoint. Nguồn
`outputs/act_pretrain` không bị ghi lại. Đích `outputs/act_pretrain_bs4` lưu
checkpoint và lịch sử mới; `fork.json` ghi nguồn và những trường thay đổi.

Batch 4 có `ceil(58313/4)=14579` train batch và `ceil(6214/4)=1554` val batch.
Batch train cuối có một mẫu, không bỏ mẫu. `train batch=100 samples=400` là
100 lần cập nhật, đã đọc 400 mẫu. Giảm số lần cập nhật mỗi epoch thay đổi quá
trình học, không đảm bảo tốc độ gấp 4 hay val giảm. Train và val đều batch 4;
validation vẫn dùng cùng tập và metric weighted, nhưng lấy latent ngẫu nhiên
trong ACT với nhóm batch mới có thể khác lần batch 1.

## Theo dõi và chạy lại nếu hết 48 giờ

```bash
J=$(cat "$HOME/cfg5-pretrain/act_bs4_to40.jobid")
squeue -j "$J" -o '%.18i %.22j %.2t %.10M %R'
tail -n 15 "$HOME/cfg5-pretrain/logs/cfg5-act-bs4-to40-$J.out"
tail -n 15 "$HOME/cfg5-pretrain/logs/cfg5-act-bs4-to40-$J.err"
sacct -j "$J" -X --format=JobID,State,ExitCode,Elapsed
```

Sau khi job đã kết thúc, có thể chạy lại cùng launcher. Nó resume checkpoint
batch 4 đã lưu, không tạo fork lần nữa. Trainer lưu latest mỗi epoch theo
config cũ (đã dùng save_every_epochs=1); best chỉ đổi khi val tốt hơn. Nếu bị
TIMEOUT hoặc gián đoạn giữa epoch, epoch chưa lưu sẽ chạy lại. Mục tiêu **40
epoch tổng cộng**, không phải thêm 40. Nếu checkpoint đã đạt 40 thì script từ
chối chạy thêm. Không chạy hai job ghi cùng output.

Nếu báo `CUDA out of memory`, batch 4 chưa vừa với cấu hình thực tế; không đổi
trực tiếp config rồi resume cùng lượt. Giữ checkpoint để tạo lượt batch nhỏ
hơn bằng cùng cơ chế fork sau khi kiểm log.

Đã kiểm fixture CPU: giữ model, AdamW moments và optimizer step, RNG,
generator, epoch/global_step, best/history; resume batch 4 vẫn giữ provenance;
resume thường từ checkpoint batch 1 sang 4 bị từ chối. Một optimizer step với
ACT thu nhỏ cũng nằm trong suite. Chưa chạy full ACT batch 4 hoặc job Slurm.
