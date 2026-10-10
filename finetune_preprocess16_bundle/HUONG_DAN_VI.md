# Preprocessing pick_place: 16 CPU, 128 GB RAM

## Cấu hình và giới hạn

- Xin Slurm 16 CPU, 128G RAM, một node; 16 worker process xử lý các episode khác nhau.
- Mỗi worker giới hạn OpenCV, FFmpeg và BLAS ở một luồng. Xem biến OPENCV_FFMPEG_THREADS trong tài liệu chính thức: https://docs.opencv.org/4.9.0/d6/dea/tutorial_env_reference.html .
- Giữ nguyên 10 Hz, 8 camera, RGB uint8 640×360, state/action, normalization và split đã khóa.
- Không xin GPU: giải mã hiện tại dùng OpenCV/FFmpeg CPU. VRAM 8 GB không tăng tốc pipeline này. RAM 128 GB là trần cấp phát, không phải mục tiêu phải sử dụng hết.
- 16 worker tăng song song nhưng không bảo đảm nhanh hơn 8 worker nếu ổ đĩa/NFS nghẽn. Log write_MiB_per_s là tốc độ byte ảnh logic của episode hoàn tất, không đo trực tiếp thiết bị vật lý.
- Bộ xử lý lưu từng episode bằng memmap, không nạp toàn bộ dataset vào RAM. Episode hoàn tất được kiểm contract và dùng lại. Episode dở giữ .partial và ERROR.json; không tự xóa hoặc coi là hoàn tất.
- Mất SSH không dừng job sbatch. Nếu job bị hủy giữa episode, cần kiểm tra các .partial trước khi chạy lại. Không cam kết tiếp tục từ frame dở.

## 1. Kiểm tra ổ chứa home trước

Chạy trên terminal SSH server:

```bash
df -h "$HOME" "$HOME/cfg5-pretrain"
quota -s
```

Cột Avail là dung lượng trống của filesystem, không phải quota riêng. Với số liệu audit hiện tại, ảnh train+val khoảng 2100 GiB; thêm 10%+1 GiB cần khoảng 2311 GiB, tức 2,26 TiB trống. Nếu quota hữu hạn, quota còn lại cũng phải đủ. Raw và dữ liệu cũ đã chiếm ổ được phản ánh trong Avail.

Nếu quota không cài hoặc không hiện giới hạn, chưa thể kết luận không có quota project/group; xem quy định storage của cluster. Khi chưa biết quota, gửi kết quả df/quota để kiểm tra. RAM 128G không thay thế dung lượng ổ.

## 2. Upload trên PowerShell Windows

```powershell
scp "F:\cfg5_mophong\finetune_preprocess16_bundle.zip" user08@172.16.3.8:/home/user08/cfg5-pretrain/
```

## 3. Cài script mới trên server

```bash
unzip -n "$HOME/cfg5-pretrain/finetune_preprocess16_bundle.zip" -d "$HOME/cfg5-pretrain/finetune_preprocess16_bundle"
cd "$HOME/cfg5-pretrain/edabk-physical-ai-robot/04-Src/software/cfg5_act"
./.venv-data/bin/python "$HOME/cfg5-pretrain/finetune_preprocess16_bundle/install_preprocess16.py" --repo "$PWD"
mkdir -p "$HOME/cfg5-pretrain/logs"
```

Đợi PREPROCESS16_BUNDLE_INSTALLED. Bộ cài chỉ thêm script/docs/test mới; không sửa lõi, split, raw, checkpoint hoặc config đã khóa. Kiểm SHA phiên bản prerequisite; nếu server khác phiên bản, dừng và gửi lỗi, không chép đè. Bộ cài từ chối khi có job fine-tune/preprocess đang chờ/chạy.

## 4. Job kiểm tra trước khi giải mã toàn bộ

```bash
if C=$(sbatch --parsable scripts/slurm/check_pick_place_preprocess16.sbatch); then
  echo "Job check: $C"
  squeue -j "$C" -o "%.18i %.22j %.2t %.10M %R"
fi
```

```bash
sacct -j "$C" -X --format=JobID,State,ExitCode,Elapsed
tail -n 35 "$HOME/cfg5-pretrain/logs/cfg5-ft-check16-$C.out"
tail -n 25 "$HOME/cfg5-pretrain/logs/cfg5-ft-check16-$C.err"
```

Preflight kiểm config/manifest/split/normalization/checkpoint SHA; raw fingerprint, metadata, trajectory, camera timestamps, video header và frame đầu. Nếu đã processed, kiểm arrays/contract và raw fingerprint. Chỉ kiểm train+val; test chưa giải mã hoặc dùng chọn mô hình. Nó tính dung lượng pending và trừ những episode đã hoàn tất, rồi so với disk free. Không đọc/giải mã toàn bộ frame video ở preflight; bước full decode kiểm từng frame và lỗi ghi trong ERROR.json.

Đạt: COMPLETED 0:0, FINETUNE_PREFLIGHT_OK và FINETUNE_CHECK16_JOB_OK. Report: outputs/data_audit/pick_place_finetune_preflight16.json. Job check không tạo ảnh full hay học mô hình. Chỉ chạy bước 5 khi disk free và quota đều đủ.

## 5. Chạy full preprocessing train+val

```bash
if D=$(sbatch --parsable scripts/slurm/preprocess_pick_place16.sbatch); then
  echo "Job preprocessing: $D"
  squeue -j "$D" -o "%.18i %.22j %.2t %.10M %R"
fi
```

Job kiểm lại trước mỗi lượt; khóa tránh hai job mới ghi chung. Không đăng ký đồng thời script preprocessing cũ. Mỗi worker giải mã tuần tự 8 camera của một episode, đổi BGR→RGB và ghi images.npy. Frame được ghi đúng tick gốc ở 10 Hz, không nhân bản action chunks trên ổ.

## 6. Theo dõi

```bash
tail -f "$HOME/cfg5-pretrain/logs/cfg5-ft-pre16-$D.out"
```

Ctrl+C chỉ thoát tail, job Slurm vẫn chạy. Log mẫu:

```text
PREPROCESS 16/822 episode=... episode_s=... ticks_per_s=... images_per_s=... write_MiB_per_s=... elapsed_min=... eta_min=...
```

episode_s là thời gian worker hoàn tất episode, gồm đọc/ghi/kiểm. images_per_s tính tổng tám view; ETA là ước tính theo số thời điểm đã hoàn tất, lượt đầu có thể dao động vì worker hoàn tất theo nhóm. Số episode trong log lấy từ split thực, không hardcode 822.

```bash
squeue -j "$D" -o "%.18i %.22j %.2t %.10M %R"
sstat -j "${D}.batch" --format=JobID,AveCPU,MaxRSS,MaxDiskRead,MaxDiskWrite
sacct -j "$D" -X --format=JobID,State,ExitCode,Elapsed
tail -n 25 "$HOME/cfg5-pretrain/logs/cfg5-ft-pre16-$D.err"
df -h "$HOME" "$HOME/cfg5-pretrain"
```

sstat có thể không hỗ trợ một số chỉ số theo cấu hình accounting của cluster; AveCPU là thời gian CPU, không phải phần trăm CPU tức thời. Thành công: FINETUNE_PREPROCESS16_OK, FINETUNE_PREPROCESS16_JOB_OK và COMPLETED 0:0. Report cuối có preprocess_seconds và total_seconds tại outputs/data_audit/pick_place_finetune_preprocess16_report.json. Ảnh ở data/processed/pick_place_finetune; chưa đăng ký fine-tune tự động.

## 7. Chạy lại

Chạy lại cùng sbatch; mọi episode hoàn tất và kiểm được sẽ được bỏ qua. Nếu có .partial, đọc ERROR.json và nguyên nhân trước; giữ lại/quarantine thư mục dở sau khi chắc chắn job cũ đã dừng. Không tự xóa raw, split hoặc processed hoàn tất.

## Kiểm thử local

Đã dùng video fixture 32×32, 4 tick, 8 camera để kiểm spawn 2 process, raw thay đổi, partial refusal, processed array corruption và verified resume. Chưa benchmark 16 nhân/128 GB hay full dataset trên cluster. Thời gian thực phụ thuộc disk/NFS và codec.
