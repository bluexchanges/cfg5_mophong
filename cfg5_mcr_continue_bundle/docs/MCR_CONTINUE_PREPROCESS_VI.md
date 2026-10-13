# Chạy tiếp preprocessing feature MCR

Ngày 13/10/2026: theo yêu cầu người dùng, job này bỏ ngân sách home 1,8 TB do script đặt ra. Không đổi quota của hệ thống; vẫn kiểm tra dung lượng trống ổ đĩa và dự phòng 10 GiB, gồm kiểm tra trước mỗi episode mới. Lỗi quota thật từ hệ thống vẫn khiến job dừng.

## Phạm vi

- 1.317 episode đã có nhãn joint ghi gốc; split giữ nguyên: 1.053 train / 132 val / 132 test.
- MCR ResNet50 đóng băng, giữ feature không gian 2.048 × 7 × 7; INT8 và scale FP16.
- Chỉ lưu feature cùng dữ liệu số/mask, chưa lưu JPEG. Payload cả ba tập khoảng 447 GiB; ước lượng cần chỗ có dự phòng khoảng 493 GiB.
- Episode cache hoàn chỉnh được kiểm tra và dùng lại. Thư mục `.partial` giữ để kiểm lỗi, không tự xóa.
- Không train trong job này. Camera dropout 0,1 và ACT dropout 0,1 áp dụng lúc train, không áp dụng khi trích feature hay đánh giá.
- Không thay raw, checkpoint, split hoặc normalization; không chạy lại light preprocessing.

## Cài và chạy trên server

```bash
cd "$HOME/cfg5-pretrain/edabk-physical-ai-robot/04-Src/software/cfg5_act"
unzip -o "$HOME/cfg5-pretrain/cfg5_mcr_continue_bundle.zip" -d "$HOME/cfg5-pretrain/cfg5_mcr_continue_bundle"
./.venv-data/bin/python "$HOME/cfg5-pretrain/cfg5_mcr_continue_bundle/install_bundle.py" --repo "$PWD" --dry-run
./.venv-data/bin/python "$HOME/cfg5-pretrain/cfg5_mcr_continue_bundle/install_bundle.py" --repo "$PWD"
C=$(sbatch --parsable scripts/slurm/cfg5_mcr_continue.sbatch)
echo "Job preprocessing: $C"
tail -F "$HOME/cfg5-pretrain/logs/cfg5-mcr-continue-$C.out"
```

Cài đặt kiểm tra SHA của code hiện có và sao lưu code thay đổi. Chỉ gửi job sau khi cả hai lệnh cài thành công. Tránh gửi job cache thứ hai khi một job cache khác đang chạy.

## Xem kết quả

`MCR_CACHE i/1317` là tiến độ build, kể cả episode được dùng lại. Pilot chỉ xử lý ba episode. Hoàn thành khi log có `MCR_FROZEN_PREPROCESS_JOB_OK` và `sacct` báo `COMPLETED 0:0`.

```bash
sacct -j "$C" -X --format=JobID,State,ExitCode,Elapsed
tail -n 40 "$HOME/cfg5-pretrain/logs/cfg5-mcr-continue-$C.err"
```
