# Sửa lỗi chia tập: profile fixed chỉ có một nhóm

Bản vá dành cho job prepare thất bại trước khi ghi split. Không thay đổi dữ liệu raw hoặc checkpoint.

Các episode cùng seed hoặc cùng trajectory được giữ trong cùng nhóm. Profile chỉ có một hoặc hai nhóm độc lập được giữ trọn trong train. Val/test được lấy từ những nhóm độc lập còn lại; không có đánh giá held-out độc lập cho profile giữ riêng ở train. Tỷ lệ 80/10/10 là mục tiêu; nếu kích thước nhóm không cho phép, script ghi tỷ lệ thực tế và lý do trong split JSON.

## Upload trên PowerShell

```powershell
scp "F:\cfg5_mophong\finetune_split_fix.zip" user08@172.16.3.8:/home/user08/cfg5-pretrain/
```

## Cài trên server

```bash
unzip -n "$HOME/cfg5-pretrain/finetune_split_fix.zip" -d "$HOME/cfg5-pretrain/finetune_split_fix"
cd "$HOME/cfg5-pretrain/edabk-physical-ai-robot/04-Src/software/cfg5_act"
./.venv-data/bin/python "$HOME/cfg5-pretrain/finetune_split_fix/apply_split_fix.py" --repo "$PWD"
```

Installer kiểm checksum phiên bản cũ/mới, tạo backup rồi ghi file atomically. Cài lại cùng bản không thay đổi file. Nếu có job ACT/fine-tune còn trong hàng chờ hoặc đang chạy, hoặc đã có split/config khóa, installer dừng để giữ nguyên trạng thái.

## Chạy lại job chia tập đã thêm exclude-invalid

```bash
if P=$(sbatch --parsable "$HOME/cfg5-pretrain/prepare_pick_place_exclude_invalid.sbatch"); then
  echo "Job chia tập: $P"
  squeue -j "$P" -o "%.18i %.20j %.2t %.10M %R"
fi
```

```bash
sacct -j "$P" -X --format=JobID,State,ExitCode,Elapsed
tail -n 30 "$HOME/cfg5-pretrain/logs/cfg5-ft-prepare-$P.out"
tail -n 30 "$HOME/cfg5-pretrain/logs/cfg5-ft-prepare-$P.err"
```

Chờ COMPLETED, ExitCode 0:0 và FINETUNE_SPLIT_LOCKED. Xem counts, profiles, group_summary và assignment_notes trong data/splits/pick_place_finetune_splits.json để biết cách chia thực tế. Sau đó mới chạy smoke/preprocessing theo hướng dẫn bundle chính.
