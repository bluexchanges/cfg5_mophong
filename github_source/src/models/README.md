# Lớp tích hợp ACT

`rh20t_act.py` dùng trực tiếp lõi DETRVAE/CVAE/Transformer của RH20T ACT baseline ở `third_party/rh20t_act`, giữ source upstream nguyên vẹn. `__init__.py` đánh dấu package, không import Torch ngầm cho các công cụ chỉ xử lý dữ liệu.

| Thành phần trong rh20t_act.py | Làm gì? |
|---|---|
| `verify_upstream` | Kiểm commit và SHA-25626 file nguồn trước khi dùng. |
| `CameraTransformer` | Kế thừa upstream Transformer, thêm positional embedding nhận diện8 camera vào các vùng feature nối theo chiều rộng. |
| `masked_l1` | Tổng sai số tuyệt đối trên targets thật / số phần tử thật; bỏ padding khỏi tử và mẫu. Không chấp nhận sample toàn padding. |
| `RH20TACTPolicy` | Dựng ResNet18/shared backbone, CVAE/Transformer; kiểm shape, nhận input đã normalize, trả prediction hoặc losses. Không có download/checkpoint/training ngầm. |
| `configure_optimizer` | Tạo AdamW với LR riêng backbone; chỉ tạo optimizer, chưa gọi step. |
| `load_policy` | Đọc config model/data và chuyển tới device chỉ định. Đây là khởi tạo random policy, không phải nạp weights ACT đã học. |

Input state(B,8), ảnh(B,8,3,H,W); output(B,20,8) normalized. Cả8 camera dùng một TCP action trong aligned base frame. `forward(state, images, actions, is_pad)` dùng CVAE posterior và trả L1/KL/total loss, valid target count, prediction. Không truyền actions thì prior latent bằng0 theo upstream. Dùng `.eval()` và `torch.inference_mode()` khi infer.

Ảnh/state do ACTDataset hoặc `panda_observation.to_act_observation` normalize đúng một lần. `is_pad=True` nghĩa target đệm. Model nhận tensor float; không nhận ảnhuint8 hay action vật lý chưa normalize như thể đã chuẩn hóa. Trước khi điều khiển, denormalize output rồi kiểm quaternion/workspace/width bằng controller.

Train/validate loop và checkpoint/resume của dự án nằm ở [src/training](../training/pretrain.py); scheduler action chunk và rollout learned policy chưa có. Model adapter từng qua full inference trên train smoke, backward nhỏ và state_dict roundtrip tạm; **chưa có weights được pretrain**. Trainer mới đã kiểm bằng fixture nhỏ, chưa chạy full ACT optimizer step trên CFG5. Xem [ACT_SETUP_VI](../../docs/ACT_SETUP_VI.md), [pretraining](../../docs/PRETRAINING_VI.md) và [config](../../configs/act/README.md).
