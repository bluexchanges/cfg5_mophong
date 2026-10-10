# Tăng batch và CPU, đo tốc độ trước khi tiếp tục tới 40 epoch

## Cấu hình và phép đo

Theo yêu cầu người dùng ngày2026-10-04, chuẩn bị job `cfg5-act-fast40`:
8 CPU (trước4),64GB RAM,1GPU,48h, partitiondefq. CấpCPU không đảm bảo
CPU sử dụng100%; worker chỉ cần cung cấp batch kịpGPU. Giữ CUDA prefetch,
pin_memory,spawn,persistent workers,prefetch2/worker. Không đổi precision,
LR,loss,model,input resolution,split,normalization hoặc target40 epoch.

Benchmark các cặp `(batch,worker)`: `(4,2)` baseline,`(8,2)`,`(8,4)`,
`(16,2)`,`(16,4)`,`(32,4)`. Mỗi cặp là subprocess riêng với full ACT,
trọng số và optimizer từ cùng checkpoint;3 batch warmup rồi20 batch đo.
Checkpoint/RNG nguồn được đọc từCPU; training probe chỉ cập nhật model/AdamW
trong RAM, không lưu các trọng số probe. Mỗi probe timeout300s. CUDA OOM
bỏ cặp đó; lỗi khác dừng job để người dùng kiểm tra. Không chạy probe trên
head node. Không tải toàn dataset vàoRAM.

Chọn theo `samples_per_s`, loại peak reserved VRAM >85%GPU total. Trong
3% của tốc độ cao nhất, ưu tiên batch nhỏ hơn rồi ítworker hơn vì probe ngắn
không đủ tin cậy để kết luận khác biệt nhỏ. Batch4 có thể được giữ nếu tăng
batch không tăng tốc/không đạt headroom. Ngưỡng85% là lựa chọn của dự án,
không phải yêu cầu của PyTorch. Probe train ngắn không đảm bảo cả run không
OOM hoặc thời gianepoch chắc chắn giảm: cần quan sát logfull train/validation.

Nguồn về tuningworker theoCPU/GPU/nơi lưu dữ liệu:
[PyTorch Performance Tuning Guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#enable-asynchronous-data-loading-and-augmentation).

## Chuyển từ lượt đang chạy

Watcher mới tiếp quản hai watcher đã biết, ownUID và đúng script argument:
`act_workers2_to40_bundle/.../switch_to_workers.py` và
`act_cuda_prefetch_bundle/.../switch_to_cuda_prefetch.py`. Nó dừng controller,
đợi child Slurm đang thực hiện xong trước khi terminate, rồi giữ hai lock cũ.
Không matchtrainer/grep hoặc người dùng khác. Chỉ nhận một ACTjob có tên
`cfg5-act-bs4-w2` hoặc`cfg5-act-prefetch`; tên khác/nhiềuACTjob sẽ dừng và báo.
Nếu có fast40 rồi thì không gửi trùng. T27 không thuộc phạm vi.

Job R được chờ history commit thêm một epoch; save_every_epochs phải1.
History được ghi sau latest.pt. Sau đó scancel scopeuser/name/id, đợi rời
queue mới submitfast40. Có thể mất vài batchepoch sau trong khoảng poll5s,
nhưng resume từepoch hoàn chỉnh. Pending jobprefetch/worker2 được hủy theo
filterPD để thay bằngfast40; nếu trạng thái thay đổi, jobmới không được gửi
cho đến khi jobcũ rờiqueue. Không chạy đồng thờihai trainerACT.

Sau khiSlurm cấpGPU, jobmới kiểmCUDA fixture của bundle trước khi cài code,
cài sourcechecksum/cú pháp cóbackup, rồi benchmark. Chọn checkpoint từ
history hoàn chỉnh mới nhất củaworker2/prefetch/fast. Computejob kiểm lại
checkpointmodel/data contract. Fork nhánh riêng
`outputs/act_pretrain_fast/`, bảo toàn model/AdamW/RNG/epoch/global_step và
best/history. Bậtcuda_prefetch và đổi batch/worker theoprobe; giữ LR.
Nếu childfastđã cócheckpoint thìresumechild, khôngbenchmark lại. Nếu child
cũ hơnnguồn, dừng báo lỗi để tránh quay lùi. Config:
`configs/act/cfg5_all_pretrain_fast40.json`.

Batch lớn làm giảm sốoptimizer updates mỗi epoch và đổi cách học.
Theo dõi val_loss vàbest.pt; không tự nhânLR. Với58.313 samples,
batch8 có7.290 trainbatch, batch16 có3.645, batch32 có1.823.

## Lệnh chạy

PowerShell Windows:

```powershell
scp F:\cfg5_mophong\act_fast40_bundle.zip user08@172.16.3.8:~/cfg5-pretrain/
```

TerminalSSH:

```bash
unzip -n "$HOME/cfg5-pretrain/act_fast40_bundle.zip" -d "$HOME/cfg5-pretrain/act_fast40_bundle"
nohup python3 -u "$HOME/cfg5-pretrain/act_fast40_bundle/scripts/slurm/switch_to_fast.py" --takeover >> "$HOME/cfg5-pretrain/switch_fast40.log" 2>&1 < /dev/null &
tail -n 15 "$HOME/cfg5-pretrain/switch_fast40.log"
squeue -u "$USER" -o "%.18i %.25j %.2t %.10M %R"
```

`--takeover` yêu cầu chuyển quyền từ watcherworkers2/prefetch. Chạy mộtlần;
lock ngăn watcherfast40 trùng. Khôngchạy lại watcherprefetch cũ trong lúc này.

Sau khi có jobmới:

```bash
JOB=$(cat "$HOME/cfg5-pretrain/act_fast40.jobid")
tail -n 20 "$HOME/cfg5-pretrain/logs/cfg5-act-fast40-$JOB.out"
tail -n 20 "$HOME/cfg5-pretrain/logs/cfg5-act-fast40-$JOB.err"
```

Log có`BENCHMARK`, `RESULT`, rồi`SELECTED batch_size=...; num_workers=...`.
Kết quảJSON trong`$HOME/cfg5-pretrain/logs/fast-tuning-$JOB/` gồm
trialconfig,trials.json,selected.json. Chỉ làmetadata nhỏ. Cấu hìnhđã chọn
được ghi cùngcheckpoint. Đối chiếu `samples_per_s` với mức trước16,4 mẫu/s
và thời gianepoch; GPUutilization100% không đảm bảo throughput tốt nhất.

## Kiểm chứng

Máyphát triểnCPU kiểm lựa chọn theoFPS/VRAM/tie, các probe chạy riêng,
timeout và không sửacheckpoint, định danhwatcher chính xác, hủyđúng job
sauhistory đã lưu, lọcPD, chọnsource mới nhất vàkhông gửitrùng. CPU fixtures
củatrainer/fork/worker/prefetch tiếp tục được kiểm. BaCUDA fixtures và các
probe fullACT chưa được chạy trênmáyphát triển; GPUjob chạy chúng trênA100.
Không tuyênbố đã tăngtốc hoặcsẽ duytrìGPU100% trước khi cólogserver.
