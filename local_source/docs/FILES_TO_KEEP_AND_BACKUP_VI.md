# Những file cần giữ và sao lưu từ server về local

Kiểm kê local ngày 2026-10-10. Trạng thái server dựa trên log người dùng cung cấp;
chưa có SSH không tương tác để quét filesystem server trực tiếp. Không xoá/copy
checkpoint hoặc raw trong lượt kiểm kê này.

## 1. Các bản sao local đã kiểm tra

| Artifact | Đường dẫn local | Kết quả |
|---|---|---|
| ACT pretrain E5 | `F:\cfg5_mophong\epoch5_transfer_probe_14120\frozen_checkpoint.pt` | SHA khớp checkpoint E5 đã chạy probe |
| ACT pretrain best E58 | `F:\cfg5_mophong\checkpoints\before_test_20261007_142338\best.pt` | SHA khớp E58 đã test/push LFS |
| MCR ResNet50 | `F:\cfg5_mophong\local_source\outputs\backbones\mcr_resnet50.pth` | SHA khớp checkpoint MCR chính thức đã pin |
| Kết quả probe E5 | `F:\cfg5_mophong\epoch5_transfer_probe_14120\` | Có report, selection, trace, 10 rollout video và scene |
| History pretrain backup | `F:\cfg5_mophong\checkpoints\before_test_20261007_142338\history.json` | File hiện có; chưa đối chiếu bản history mới nhất server |

SHA256 đã đo:

```text
E5  edca36c9c80c453bb38ed56f68b77a2e3fcd4b175f8296119670131c5c7ffc71
E58 7567df2b01291fbd8f096cce5693bde542fbfc9709b9d5b2d7f18d722863ac9e
MCR 21560db9e795384a7b73b0626d33d6525f9d4c5de11fca6ded52089aab14f97c
```

**Bản local chưa dùng được:**
`F:\cfg5_mophong\checkpoints\before_test_20261007_142338\latest.pt`
có 315359232 byte nhưng `torch.load(..., weights_only=True)` báo không tìm được
central directory của archive. Có thể file bị cắt/truyền chưa đầy đủ; chưa kết luận
nguyên nhân hoặc epoch. Giữ file hiện có để đối chiếu, tải lại bản server dưới tên
khác, so SHA và đọc metadata trước khi dùng resume.

## 2. Artifact nhỏ/quan trọng cần tải từ server

Mọi đường dẫn dưới đây tính từ repo server:
`/home/user08/cfg5-pretrain/edabk-physical-ai-robot/04-Src/software/cfg5_act`.

### Checkpoint và lịch sử

- `outputs/act_pretrain/best.pt`: E5, giữ bản original và provenance; local có bản tương ứng trong probe.
- `outputs/act_pretrain_bs8_to80/best.pt`: E58; local đã có bản đúng SHA.
- `outputs/act_pretrain_bs8_to80/latest.pt`: theo metadata user đã gửi là E78; tải lại để có bản resume tốt.
- `outputs/act_pretrain_bs8_to80/history.json`: lịch sử đầy đủ/latest của run; history không thể tái tạo weights từng epoch.
- `outputs/backbones/mcr_resnet50.pth`: local đã có bản chính thức, không cần tải trùng khi SHA khớp.
- Các run MCR mới sau khi bắt đầu train: lưu `best.pt`, `latest.pt`, `history.json`
  và `epoch_*.pt`. Frozen warmup dự kiến dùng `outputs/pick_place_mcr_frozen/`;
  tên output cuối cùng phải theo lệnh sbatch thực tế. Chưa có bằng chứng run này đã được gửi.

`best.pt`/archive mỗi epoch dùng inference hoặc khởi tạo giai đoạn sau;
`latest.pt` mới có optimizer/RNG/runtime để resume chính run đó. Không thay latest
bằng best rồi coi là resume tương đương.

### Config, split, normalization và manifest

- Tải **toàn bộ `configs/`** của server vào snapshot riêng: chứa cấu hình model,
  dữ liệu, train, preprocessing, camera, task và các JSON do job chuẩn bị sinh ra.
- Tải **toàn bộ `data/splits/`**: đặc biệt các file `pick_place_finetune_*` và
  `cfg5_all_*` (splits, manifest, normalization). Bản CFG5 local tồn tại từ trước
  nhưng không mặc định giống server, phải so hash trước khi thay.
- `configs/act/pick_place_finetune_mp4_w16.json`: input hiện tại của cache mới.
- `configs/act/pick_place_finetune_model.json`, `configs/act/pick_place_finetune.json`,
  `configs/data/pick_place_finetune8.json`: cấu hình sinh cho bộ dữ liệu mới.
- `configs/act/cfg5_all_pretrain_bs8_80epoch.json`: cấu hình run pretrain thật trên server.
- `outputs/data_audit/pick_place_finetune.json`: audit có lý do loại 111 demo và accepted membership.
- Mọi audit/split-fix report có liên quan nếu tồn tại: lưu để giải thích dữ liệu bị loại và group/profile assignment.

Các cấu hình/split mới sinh trên server chưa thấy trong cây `local_source/configs`
và `local_source/data/splits` lúc kiểm kê. Đây là nhóm cần kéo về ưu tiên cao.

### Kết quả đánh giá và log

- `outputs/act_pretrain_bs8_to80/test_final/test_report.json`: kết quả test E58 trên CFG5.
- Các artifact metric/prediction/report đi kèm test nếu có; duplicate best.pt có
  thể không cần sao lưu thêm nếu SHA khớp bản best chính.
- `outputs/epoch5_transfer_probe_14120/`: local đã có; giữ report/selection,
  per-episode offline/rollout reports, traces, scene configs/XML/manifest và video.
- Log server tại `/home/user08/cfg5-pretrain/logs/`: `.out`/`.err` của prepare,
  pretrain/test, probe, pilot14129, build14130 và các run train tiếp theo.
- `outputs/data_cleanup/`: audit/journal dọn ảnh nếu đã chạy; không coi là đã có
  khi người dùng chưa gửi kết quả chạy script.

### Metadata cache mới

Trong `data/features/pick_place_mcr/`, lưu local:

- `context.json`: source, toàn bộ config/split/stats và feature contract.
- `storage_report.json`: số đo và ước lượng dung lượng; báo cáo pilot không phải báo cáo full.
- `READY.json`: chỉ xuất hiện sau khi build full hoàn tất; chưa có bằng chứng build14130 đã xong.
- `episode.json` của từng episode hoàn chỉnh: SHA numeric/JPEG/chunks, shape,
  chất lượng JPEG và sai số quantization để đối chiếu/rebuild cache.
- `ERROR.json` của episode `.partial` nếu có lỗi: lưu chẩn đoán, không tự xoá partial khi job đang chạy.

Các file nhỏ này giúp tái lập nhưng **không thay được payload feature/JPEG**.

## 3. Dữ liệu lớn cần giữ trên server hoặc có bản sao độc lập

| Dữ liệu | Mục đích | Cần tải toàn bộ về local ngay? |
|---|---|---|
| `~/cfg5-pretrain/downloads/finetune_data` và `dataset_moi` | Raw mới: MP4, trajectory, timestamps, scene/metadata | Nên có ít nhất một bản sao raw đầy đủ, đã kiểm. Hai episode local không phải backup toàn bộ dataset. Drive chỉ là backup khi còn đủ nội dung và truy cập được. |
| `data/processed/pick_place_finetune_mp4/` | Numeric đã căn, segment và reference raw; root thật theo config server | Nên tải metadata/numeric nhỏ về local; không chứa bộ ảnh NPY 2 TB |
| `data/features/pick_place_mcr/` | Feature INT8/scales, JPEG SQLite, numeric, metadata cho warmup/full | Giữ server khi đang build/train; không bắt buộc tải khoảng400+GiB về local, có thể tái tạo từ raw + config + MCR |
| Raw CFG5 | Tái tạo preprocessing và kiểm chứng pretrain/test cũ | Giữ ít nhất một bản đầy đủ; không thuộc lệnh dọn ảnh preprocessing |
| Ảnh preprocessing test CFG5 nếu đã tạo | Tái chạy test với pipeline NPY cũ | Giữ server trong phạm vi dọn hiện tại; có thể tái tạo từ raw khi cần |
| Numeric/episode metadata preprocessing CFG5 | Căn thời gian, labels và provenance cũ | Giữ; dung lượng nhỏ hơn ảnh rất nhiều |

Chỉ `images.npy` của 531train+66val CFG5 cũ được user cho phép dọn theo script
cleanup. Không dọn cache mới, test, raw, configs, checkpoint hoặc cả thư mục outputs.

## 4. Source, mô phỏng và môi trường

- Giữ `src/`, `scripts/` (bao gồm Slurm), `configs/`, `tests/`, `docs/`.
- Giữ `third_party/rh20t_act/`, upstream commit và license để nạp đúng ACT.
- Giữ MuJoCo Panda XML/assets, camera rig/task config, scene builder/controller/evaluator;
  XML scene có đường dẫn mesh tuyệt đối từ server cần regenerate/đổi meshdir khi chạy local.
- Giữ requirements/lockfiles, bundle manifest, nguồn/checksum MCR, Git commit và
  local patch của server nếu có. Code local đã có nhưng server có thể đã sửa độc lập.
- Xuất `pip freeze` của đúng `.venv-data` server để biết dependency thực tế, nhất là MuJoCo.
  Không cần sao chép thư mục `.venv-data` Linux sang Windows; tái tạo theo dependency phù hợp OS.
- Không đưa token rclone, private SSH key, credential hoặc raw/checkpoint/cache lớn vào Git.

## 5. Chỗ lưu snapshot local

Khuyến nghị lưu bản server vào:
`F:\cfg5_mophong\local_source\server_snapshots\20261010\`
với các nhóm `configs`, `data/splits`, `reports`, `logs`, `cache_metadata`,
`checkpoints` và `server_code`. Đây là vị trí đề xuất, chưa có lệnh download thực tế.
Giữ riêng để tránh ghi đè source local hoặc membership cũ khi chưa so checksum.

Ưu tiên tải: config/split/stats/manifest + latest E78 tốt + report test/log +
cache metadata/numeric mới. E5/E58/MCR đã có SHA đúng ở local; không cần tải trùng.
