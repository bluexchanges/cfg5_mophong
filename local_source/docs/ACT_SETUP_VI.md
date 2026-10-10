# RH20T ACT đã tải và được tích hợp đến đâu?

**Cập nhật 2026-09-30:** Tài liệu dưới đây ghi trạng thái tích hợp model lượt4a. Trainer/validation/checkpoint/resume đã được bổ sung sau đó; xem [PRETRAINING_VI.md](PRETRAINING_VI.md). Chưa có full ACT optimizer step trên CFG5 hoặc checkpoint đã học.

Ngày2026-09-28: đã tải đủ26 file source,103.547 byte, từ [RH20T ACT baseline tại commit cố định](https://github.com/rh20t/act_baseline/tree/107ce1287a46be3a1a7ea954f8bbdec3bb103288). Model đã khởi tạo và chạy inference; chưa có pretrained ACT checkpoint, chưa chạy optimizer/training, chưa rollout ACT trong MuJoCo.

“Tải model” cần tách ba việc: tải **source kiến trúc**, cài thư viện để **khởi tạo model**, và tải/nạp **weights đã học**. Lượt này hoàn tất hai việc đầu; model/backbone đang khởi tạo ngẫu nhiên. Source không tự chứa kỹ năng đặt vật.

## Source nào dùng, thay đổi gì?

Source đầy đủ giữ nguyên ở `third_party/rh20t_act/`, gồm MIT license root và Apache-2.0 license DETR. Manifest bên cạnh ghi commit, Git blob SHA, SHA-256 từng file. Snapshot ở `references/upstream/` vẫn là tư liệu khảo sát, không sửa.

Adapter chạy ở `src/models/rh20t_act.py`, config `configs/act/rh20t_base8.json`. Không sửa file upstream; import trực tiếp DETRVAE, Transformer, CVAE encoder, FrozenBatchNorm và positional encoding từ bản đã tải.

| Hạng mục | Cách tích hợp |
|---|---|
| Input | Một timestep,8 ảnh RGB đã normalize + state8D. Upstream train script mẫu chỉ chọn một camera `top`; dataset của dự án cung cấp8 view cố định thứ tự. |
| Representation | Giữ số chiều8 nhưng dùng contract riêng aligned-base xyz+wxyz+width(m); không coi cùng số chiều là tự tương thích hệ tọa độ. |
| Chunk |20 queries → output(B,20,8), targets bắt đầu t+1 do ACTDataset tạo. |
| Backbone | ResNet18 shared giữa camera, FrozenBatchNorm; `weights=None` mặc định, không tải weights ngầm. `backbone_weights_path` có thể nhận local torchvision ResNet18 state_dict khi được chuẩn bị sau. |
| Nhận diện camera | Thêm learned camera embedding vào positional embedding trước khi gọi upstream Transformer, để các vị trí ảnh giống nhau ở8 view có danh tính khác nhau. Đây là thay đổi của dự án. |
| CVAE/Transformer | Giữ upstream lõi; hidden512, feedforward3200, encoder4, decoder7, heads8, latent32. Config đầy đủ hiện83.873.929 parameters. |
| Loss | L1 chỉ trên action thật, denominator=8×số target thật toàn batch; padding không vào sum hoặc denominator. KL theo sample, weight10. |
| Padding encoder | Mask phần đệm và đặt action padding về0 trước encode, tránh NaN/giá trị rác ở padding ảnh hưởng model. |
| PyTorch API | Dùng `forward`, không override `__call__`; không đọc argparse của chương trình gọi, không ép `.cuda()`. CPU/device do caller chọn. |
| Normalization | Adapter dữ liệu đã normalize ảnh/state; model wrapper không normalize lần hai. Output còn normalized, phải denormalize và kiểm quaternion/workspace trước điều khiển. |

Đây là bản thích nghi cho dự án, không phải reproduction nguyên xi mọi setting của bài báo. Khi so sánh từ-scratch với pretrain→fine-tune, cần cùng kiến trúc/camera embedding và chính sách khởi tạo backbone ở cả hai nhánh. Chưa lựa backbone weights hay GPU training trong lượt này.

## Cài/chạy lại

Môi trường hiện có Windows CPython3.12.14, torch2.7.1+cpu, torchvision0.22.1+cpu, einops0.8.1. Cặp Torch/torchvision CPU được lấy theo [hướng dẫn phiên bản chính thức](https://pytorch.org/get-started/previous-versions/#v271). CPU build để kiểm tích hợp; `cuda_available=False` của build này không chứng minh máy không có GPU.

Tại project root, nếu tạo lại môi trường:

```powershell
& '.\.venv-data\Scripts\python.exe' -X utf8 scripts/download_act_baseline.py
& '.\.venv-data\Scripts\python.exe' -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
& '.\.venv-data\Scripts\python.exe' -m pip install -r requirements-act-lock.txt
& '.\.venv-data\Scripts\python.exe' -m pip check
& '.\.venv-data\Scripts\python.exe' -X utf8 scripts/check_act.py
& '.\.venv-data\Scripts\python.exe' -X utf8 -m unittest discover -s tests -v
```

Downloader kiểm file đã có, từ chối ghi đè source khác hash. Không chạy `train.py` hoặc `pip install -r requirements.txt` trong upstream: entrypoint đó chưa dùng adapter/split/normalization của dự án và dependency gốc cũ không phải môi trường đã kiểm.

Ví dụ API từ chương trình Python chạy tại root:

```python
import torch
from src.models.rh20t_act import load_policy
policy = load_policy(device='cpu').eval()
# state: float32(B,8), images: float32(B,8,3,360,640), đã normalize.
# with torch.inference_mode():
#     action_chunk = policy(state, images)
```

Chưa có inference scheduler/temporal aggregation và chưa gửi prediction ngẫu nhiên vào robot. Module `panda_observation.denormalize_action` + controller là phần chuyển action vật lý đã có; rollout checkpoint được làm sau training.

## Kiểm chứng và giới hạn

`docs/audit/act_integration_check.json` ghi: full model inference trên **một sample thật của train smoke**, input(1,8,3,360,640), output(1,20,8), hữu hạn. Backward dùng kiến trúc giảm64 hidden/1 encoder/1 decoder, ảnh64×96 để kiểm graph/gradient nhanh. **Không báo backward nhỏ này là đã train full model.** Optimizer steps=0.

35 tests đạt sau tích hợp, gồm batch2, padding không ảnh hưởng loss/posterior, denominator, camera identity/gradient, save/load state_dict tạm cho ra cùng prediction, contract và26 source hash. Tests không lưu checkpoint đã học. `pip check` không báo dependency lỗi.

Lệnh `check_act.py` chưa đo chất lượng dự đoán; loss của random weights chỉ dùng kiểm số học. Đường đi được xác nhận, kỹ năng chưa được học. Trainer, validation đúng denominator/count và checkpoint/resume đã bổ sung sau báo cáo này; xem [hướng dẫn pretraining](PRETRAINING_VI.md). Pretraining dài do người dùng chạy sau khi server có processed data.
