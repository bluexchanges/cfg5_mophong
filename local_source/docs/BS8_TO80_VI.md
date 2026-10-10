# Giảm về 1 GPU, batch 8, worker cũ; tiếp đến tổng 80 epoch

Job13988 xin2GPU đang PD với QOSMaxGRESPerJob. User yêu cầu giảm batch8 và quay lại worker trước. Gói này xin1GPU/8CPU/64GiB, giữ `num_workers` từ metadata checkpoint fast40 (tuner trước chọn2 hoặc4), LR/precision/CUDAprefetch/contract giữ nguyên. Không chạy adapter multi-GPU.

Batch/worker không phải số GPU Slurm xin. Đổi `--gres=gpu:1` mới là thay đổi trực tiếp liên quan tới lỗi giới hạn GRES vừa thấy. Vẫn có thể phải chờ tài nguyên hoặc giới hạn khác.

## Upload từ PowerShell Windows

```powershell
scp "F:\cfg5_mophong\act_bs8_to80_bundle.zip" user08@172.16.3.8:/home/user08/cfg5-pretrain/
```

## Giải nén và gửi từ terminal SSH server

```bash
unzip -n "$HOME/cfg5-pretrain/act_bs8_to80_bundle.zip" \
  -d "$HOME/cfg5-pretrain/act_bs8_to80_bundle"
python3 "$HOME/cfg5-pretrain/act_bs8_to80_bundle/scripts/slurm/submit_bs8_to80.py" \
  --replace-pending 13988
```

Launcher kiểm checksum trước khi thao tác, khóa các controller, chỉ hủy job13988 của user hiện tại khi đúng tên `cfg5-act-bs16-g2-to80` và cònPD. scancel kèm filterPENDING để tránh race job vừa chuyểnR. Nếu job đã R/khác tên/cóACT khác thì từ chối gửi thêm. Không hủy T27 hoặc job không phải ACT. Nếu13988 đã rời queue, kiểm không cóACT khác rồi gửi BS8.

Trong compute job: kiểm1GPU, testCUDAprefetch nhỏ, backup/cài code có checksum, probe3batch fullACT trên bản sao trạng thái không lưu, rồi fork từ `outputs/act_pretrain_fast/latest.pt` sang `outputs/act_pretrain_bs8_to80`. Nếu nhánh BS8 đã có latest thì resume nhánh đó. Nguồn phải lưu ít nhất40epoch; không ghi đè output dở. Giữ model/optimizer/RNG/epoch/global_step/best/history. Lưu checkpoint mỗi epoch; mục tiêu là80 tổng, không phải80 epoch thêm.

## Theo dõi

```bash
python3 "$HOME/cfg5-pretrain/act_bs8_to80_bundle/scripts/slurm/status_to80.py" --run bs8
J=$(cat "$HOME/cfg5-pretrain/act_bs8_to80.jobid")
tail -n 12 "$HOME/cfg5-pretrain/logs/cfg5-act-bs8-to80-$J.out"
tail -n 20 "$HOME/cfg5-pretrain/logs/cfg5-act-bs8-to80-$J.err"
```

58.313 mẫu train →7.290batch/epoch ở batch8; batch cuối1mẫu. Val6.214mẫu →777batch; cuối6mẫu. JSONepoch40 tương đương epoch thứ41. Thời gian status là train+val, không gồm savecheckpoint.

## Vẽ lại loss

```bash
python3 "$HOME/cfg5-pretrain/act_bs8_to80_bundle/scripts/plot_training_loss.py" \
  --repo "$HOME/cfg5-pretrain/edabk-physical-ai-robot/04-Src/software/cfg5_act"
```

Ưu tiên historyBS8, fallbackfast40 nếu chưa có nhánh mới. BaSVG tại `docs/audit/training_plots/`. Chỉ đọc JSON, không ảnh hưởng training. Để chắc chắn vẽ đúng lượt, có thể truyền `--history` trỏ file cụ thể.

## Giới hạn kiểm chứng

FixtureCPU kiểm worker cũ/signature, scope hủy pending/race, fork/resume batch8 và optimizer. Không fulltrain tại máy phát triển; jobserver thực hiện CUDAgate/probe. Giảm worker không bảo đảm nhanh hơn; đây là cấu hình theo yêu cầu user để trở về allocation đã từng dùng.
