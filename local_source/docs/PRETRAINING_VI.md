# Pretraining ACT trên toàn CFG5

Đây là pipeline **offline imitation learning** trên robot demonstrations CFG5. Mỗi sample gồm 8 ảnh RGB, state TCP/gripper 8D ở thời điểm `t`, và tối đa 20 target tương lai. Target cánh tay là *future measured-state proxy*, không phải lệnh motor đã xác minh. MuJoCo không chạy trong mỗi bước pretraining. Dữ liệu task đặt chốt và fine-tuning là giai đoạn sau.

## Thành phần và trạng thái kiểm chứng

| Thành phần | File | Trạng thái |
|---|---|---|
| Dữ liệu | `src/data/act_dataset.py` | Đã kiểm sample train/val trên lần preprocess cũ; server phải preprocess lại. |
| Model | `src/models/rh20t_act.py`, `third_party/rh20t_act/` | Inference full trên một sample và backward model nhỏ đã kiểm; source upstream giữ nguyên. |
| Trainer | `src/training/pretrain.py`, `scripts/train.py` | Có train loop, validation weighted theo số target thật, checkpoint và resume tại ranh giới epoch. Đã kiểm bằng fixture và **một optimizer step với ACT kiến trúc thu nhỏ**; chưa chạy full ACT optimizer step trên CFG5. |
| Validation độc lập | `scripts/validate.py` | Đọc `best.pt` hoặc `latest.pt`, kiểm contract và tính toàn bộ validation mặc định. Đã kiểm logic bằng fixture; chưa có checkpoint ACT thật. |

Config model toàn CFG5 là `configs/act/rh20t_all8.json`, trỏ tới `configs/data/cfg5_all8.json`. Config job là `configs/act/cfg5_all_pretrain.json`. Nó khởi tạo ACT và ResNet18 ngẫu nhiên (`backbone_weights_path: null`); không tự tải pretrained weights. `epochs: 10`, `batch_size: 1` là điểm bắt đầu để đo trên server, chưa được benchmark. Kiểm VRAM/thời gian trên máy thật rồi mới quyết định thời lượng thí nghiệm. Muốn kéo dài sau một checkpoint epoch, tăng `epochs` và `--resume`; các trường khác phải giữ nguyên.

## Trước khi chạy trên server

Làm theo [SERVER_REPROCESS_VI](SERVER_REPROCESS_VI.md): chuẩn bị raw riêng, tạo manifest/split server mới, **so đúng membership 531/66/66 với split đã khóa**, preprocess đủ train và val, fit normalization từ train rồi kiểm loader. Bản GitHub không chứa processed NPY. Manifest/split server có thể có hash khác do `mtime` nguồn; không dùng lại normalization cũ. Giữ test khóa.

Cài PyTorch có CUDA phù hợp GPU/driver server, rồi kiểm `python -c "import torch; print(torch.cuda.is_available())"` trả `True`. Lock ACT cũ dùng `+cpu` của Windows, không dùng nguyên lock đó để cài GPU. Môi trường mô phỏng MuJoCo và model Panda không phải điều kiện của offline pretraining; chúng phục vụ đánh giá rollout về sau.

## Chạy thử và chạy chính thức

Các lệnh dưới đây chạy từ **root repository trên server**, sau khi đã có `data/processed/cfg5_all/`, split và normalization server. Thay tên file nếu bạn đặt khác. `--smoke` dùng **1 train batch + 1 val batch, 1 epoch**, ghi riêng trong `outputs/act_pretrain/smoke/`. Nó vẫn dựng full ACT; cần GPU có đủ VRAM. Không tự chạy training dài khi chỉ muốn kiểm.

```bash
python scripts/train.py --device cuda --smoke \
  --splits data/splits/cfg5_server_splits.json \
  --stats data/splits/cfg5_server_normalization.json \
  --processed-root data/processed/cfg5_all
```

Kiểm `outputs/act_pretrain/smoke/history.json`, `latest.pt`, `best.pt`: train/val phải có `samples`, `batches`, `episodes`, `valid_action_steps`; loss hữu hạn. Đây chỉ là kiểm plumbing, không chứng minh model đã học thao tác. Sau đó chạy full, **không** có `--smoke`:

```bash
python scripts/train.py --device cuda \
  --splits data/splits/cfg5_server_splits.json \
  --stats data/splits/cfg5_server_normalization.json \
  --processed-root data/processed/cfg5_all
```

Trainer đối chiếu membership server với `cfg5_all_splits.json` đã khóa trước khi đọc sample. Nó dùng toàn bộ sample train mỗi epoch và toàn bộ sample val sau epoch; không giới hạn validation ở một số batch trong run chính. Các sample train được xáo theo episode để đọc memmap liên tục hơn. `best.pt` được chọn theo validation `L1_valid + kl_weight × KL`; L1 tổng hợp bằng tổng sai số chia **tổng số phần tử action thật**, KL theo sample. Từng batch có padding khác nhau không được cộng loss theo trung bình batch đơn giản. Validation đặt fixed RNG seed và trả trạng thái RNG train sau khi xong, để so sánh epoch ổn định. `history.json` ghi số lượng thực sự đã dùng.

`latest.pt` có model, optimizer, epoch, global step, RNG Python/NumPy/Torch/CUDA, DataLoader generator state và hash của model config/data config/manifest/split/stats/camera. Cả `latest.pt` và `best.pt` lưu config, normalization và phiên bản Python/Torch/NumPy để phục vụ nạp model và truy vết. Ghi file tạm rồi thay thế để tránh checkpoint nửa chừng. `best.pt` chứa model và contract để đánh giá/fine-tune, không chứa optimizer. Resume **chỉ tại ranh giới epoch đã lưu**, không resume giữa batch; nếu gián đoạn giữa epoch, chạy lại epoch đó.

```bash
python scripts/train.py --device cuda \
  --splits data/splits/cfg5_server_splits.json \
  --stats data/splits/cfg5_server_normalization.json \
  --processed-root data/processed/cfg5_all \
  --resume outputs/act_pretrain/latest.pt
```

Các đường dẫn và cấu hình training phải giống run cũ; chỉ `epochs` được tăng. Nếu khác split/stats/model/camera, resume sẽ dừng. Khi dùng nhiều DataLoader workers, thứ tự mẫu ở ranh giới epoch được giữ qua generator state; tính tái lập bit-for-bit còn tùy runtime GPU và kernel. Bản đầu dùng `num_workers: 0` để đơn giản hóa kiểm chứng. Cấu hình full mặc định yêu cầu CUDA nhằm tránh vô tình chạy hàng chục nghìn batch trên CPU.

Validation độc lập cho checkpoint đã có:

```bash
python scripts/validate.py --device cuda \
  --checkpoint outputs/act_pretrain/best.pt \
  --splits data/splits/cfg5_server_splits.json \
  --stats data/splits/cfg5_server_normalization.json \
  --processed-root data/processed/cfg5_all \
  --output outputs/act_pretrain/validation.json
```

`validation.json` ghi `all_val_samples` và `evaluated.samples` để thấy có đánh giá toàn bộ hay không. Đây là **offline loss**, chưa phải tỷ lệ đặt chốt thành công. Không dùng test để chọn model. Tới khi khóa checkpoint/config mới preprocess và mở test.

## Giới hạn hiện tại

- Chưa có processed CFG5 trên server và chưa kiểm full ACT optimizer step bằng GPU. CPU test kiểm điều phối/checkpoint bằng fixture và optimizer step thật với ACT kiến trúc thu nhỏ, ảnh 32×32.
- Chưa benchmark VRAM/batch size/thời gian; `batch_size: 1` vẫn có thể thiếu VRAM với 8 ảnh 640×360 và model 83,9 triệu tham số. Nếu đổi batch size sau khi đã train, tạo run mới và ghi rõ khác biệt.
- Chưa có learned ACT rollout, temporal aggregation, demonstrations task đích, fine-tuning hay đánh giá test. Chúng không phải kết quả của pretraining CFG5.
