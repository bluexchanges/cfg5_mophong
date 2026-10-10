# Config kiến trúc ACT

`rh20t_base8.json` là model config của subset 5 task lịch sử. Pretraining toàn CFG5 dùng `rh20t_all8.json` trỏ `configs/data/cfg5_all8.json` và job config `cfg5_all_pretrain.json`. `data_config` cung cấp camera order, kích thước ảnh, normalization và chunk_size. Input/output8D phải cùng representation; không đổi số chiều mà bỏ qua frame/units. Xem [hướng dẫn pretraining](../../docs/PRETRAINING_VI.md).

| Trường | Ý nghĩa |
|---|---|
| `upstream_commit` | Commit RH20T baseline được tải và kiểm hash. |
| `state_dim`, `action_dim`, `chunk_size` |8,8,20; targets là tương lai t+1...t+20, padding ở cuối segment. |
| `backbone`, `backbone_weights_path` | ResNet18; null nghĩa khởi tạo random, không download. Local path nếu sau này nạp torchvision state_dict được chuẩn bị rõ nguồn. Không phải ACT checkpoint path. |
| `hidden_dim`, `dim_feedforward`, `enc_layers`, `dec_layers`, `nheads` |512,3200,4,7,8; cấu hình đầy đủ đã chạy inference. Chưa benchmark training batch size/VRAM. |
| `dropout`, `pre_norm` | Transformer dropout0.1 và kiểu normalization theo upstream. |
| `camera_identity_embedding` |True: thêm learned identity của từng camera để phân biệt view. Đây là adaptation của dự án. |
| `kl_weight` | Tổng loss=L1(valid-only)+10×KL; không chia L1 cho padding. |
| `lr`, `lr_backbone`, `weight_decay` | Dùng khi tạo optimizer; có giá trị không đồng nghĩa đã train. |
| `input_normalization` | Cam kết input đã ImageNet/state train-stat normalization, tránh normalize trùng. |

Schema1 là schema model, độc lập schema2 của task MuJoCo. Sau khi chọn config dữ liệu mới, chỉnh `data_config` tương ứng; mọi artifact phải đúng contract. Giữ cùng kiến trúc/khởi tạo khi so sánh from-scratch với pretraining. Hướng dẫn và kiểm chứng: [ACT_SETUP_VI](../../docs/ACT_SETUP_VI.md).
