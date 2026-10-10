# So sánh implementation ACT và validation sample density

Khảo sát ngày 2026-09-27 bằng source theo commit cố định. Đây là phân tích tĩnh, cộng probe hàm sampling bằng dữ liệu giả; chưa khởi tạo hoặc train ACT.

**Cập nhật lượt2:** đã chọn lõi RH20T ACT, adapter ngoài upstream dùng base-frame TCP8D và8 camera. Xem [DECISIONS.md](DECISIONS.md) và [báo cáo dữ liệu](STAGE2_REPORT_VI.md); không dùng trạng thái chờ chốt trong báo cáo lịch sử này làm chỉ dẫn hiện tại.

## 1. Ba implementation đã kiểm tra

| Tiêu chí | ACT gốc | AirExo ACT-in-the-wild | RH20T ACT baseline |
|---|---|---|---|
| Repository | `tonyzhaozh/act` | `AirExo/act-in-the-wild` | `rh20t/act_baseline` |
| Commit ngắn | `742c753` | `a11e8e6` | `107ce12` |
| Mục đích | ACT/ALOHA hai tay, sim và real | AirExo whole-arm, pretrain/fine-tune | Ví dụ pretraining ACT với RH20T |
| Model | `detr/models/detr_vae.py` | Cùng họ DETR-VAE | `models/detr/models/detr_vae.py` |
| State/action | Builder hard-code 14 | `state_dim` cấu hình, task ví dụ 8 hoặc 12 | Script state_dim=8, builder nhận tham số |
| Dữ liệu | HDF5 từng episode | `scene*/meta.json`, record NPY theo timestamp | RH20T đã giải mã JPG + transformed NPY + cleaned index JSON |
| Action semantics | Joint targets ALOHA | Robot states tương lai; chọn các khớp tùy task | TCP camera frame 7D + gripper width 1D |
| Training | `imitate_episodes.py` gồm cả rollout | `train.py`, `eval.py` tách riêng | `train.py`, chưa có Panda evaluator |
| Simulation | MuJoCo qua dm_control, robot/task ALOHA | `eval.py` dành cho phần cứng AirExo | Chưa có environment MuJoCo |
| License root | MIT | MIT | MIT |
| Phù hợp CFG5 nguyên bản | Cần adapter toàn bộ | Cần adapter toàn bộ | Gần schema nhất, nhưng loader chưa dùng trực tiếp được |

Nguồn khóa phiên bản và bản snapshot nằm ở [SOURCES](../references/SOURCES.md). `rh20t/rh20t_api` được dùng riêng để xác minh cấu hình Franka, joint order và schema, không phải implementation ACT.

AirExo là code cho paper AirExo; không thể đồng nhất nó với thí nghiệm RH20T chỉ vì có tác giả chung. Repository chính thức RH20T baseline là nguồn liên quan trực tiếp hơn, được tìm thấy khi xác minh mã nguồn thí nghiệm.

## 2. Dependencies và khả năng chạy trên máy hiện tại

**ACT gốc:** README dùng Python 3.8.10, torch/torchvision, MuJoCo 2.3.7, dm_control 1.0.14, h5py, OpenCV, einops và các tiện ích khác. Các đường đi chính gọi `.cuda()`. Môi trường ALOHA có sẵn không thể thay tên robot thành Panda rồi dùng ngay.

**AirExo:** README liệt kê NumPy, torch/torchvision, OpenCV, Pillow, easydict, tqdm, transforms3d, IPython; source training còn import matplotlib/einops. Real evaluation cần hardware và `easyrobot`. `detr/main.py` chọn CPU/CUDA, nhưng loader đặt **100 workers** và có định dạng riêng. Không dùng giá trị đó làm mặc định trên Windows. ImageNet normalization nằm trong `wholearm.py`, không nằm ở policy như ACT gốc.

**RH20T baseline:** `requirements.txt` pin torch 1.13.1, viết `torchvision=0.14.1` với một dấu `=`; cần sửa cú pháp trong bộ requirements riêng. File này cũng không liệt kê trực tiếp mọi dependency import (ví dụ NumPy/Pillow). `models/detr/main.py` vẫn gọi `.cuda()`, còn script đặt 20 workers. Python 3.12.7 hiện có không phải môi trường upstream đã được kiểm chứng. Chưa cài hoặc thử dependency graph này.

Khuyến nghị triển khai sau: môi trường ảo riêng, chọn torch/torchvision tương thích với Python/GPU thực tế, `num_workers=0` cho smoke test Windows, quản lý device ở wrapper. Import model builder trực tiếp với config object để tránh việc helper upstream đọc `sys.argv` của chương trình gọi. Không cần sửa snapshot tham khảo.

Các backbone đều bật pretrained ResNet18. Khi chưa có cache, khởi tạo có thể phát sinh tải weights. Cần config rõ `pretrained_backbone` và thông báo trước khi tải. Chưa có weights nào được tải ở lượt này.

## 3. Validation density: đọc code đến tận dataset

“Duyệt toàn bộ validation loader” không nhất thiết là “xét mọi thời điểm của mọi trajectory”. Số mẫu phụ thuộc cách `Dataset.__len__` và `__getitem__` được định nghĩa.

### ACT gốc

Nguồn: [utils.py](https://github.com/tonyzhaozh/act/blob/742c753c0d4a5d87076c8f69e5628c79a8cc5488/utils.py), [train_bc](https://github.com/tonyzhaozh/act/blob/742c753c0d4a5d87076c8f69e5628c79a8cc5488/imitate_episodes.py#L294).

- `EpisodicDataset.__len__` bằng số episode được giao cho split.
- Mỗi `__getitem__(episode_index)` chọn **một start timestamp ngẫu nhiên** bằng `np.random.choice(episode_len)`; observation tại đó đi kèm chuỗi target rồi padding.
- Split mặc định 80% train / 20% validation theo permutation episode. Không có offline test split độc lập.
- Mỗi epoch, **trước phần training**, `policy.eval()` và `torch.inference_mode()` duyệt hết validation loader, không `break` hay cap số batch.
- Loader validation có `shuffle=True`; không truyền `drop_last`, nên mặc định giữ batch cuối.
- Tổng số validation samples mỗi epoch bằng số validation episodes, **không phải tổng số timestep**.

Ví dụ tự đặt: 20 episode validation, mỗi episode 300 frame, mỗi epoch chỉ lấy 20 observation–chunk thay vì phủ gần 6.000 start positions. Các epoch sau chọn timestamp khác. Nhiễu này có thể khiến checkpoint tốt nhất phụ thuộc những đoạn ngẫu nhiên được lấy.

Ngoài ra, `get_norm_stats(dataset_dir, num_episodes)` đọc toàn bộ episodes **trước khi tách train/val cho loader**. Nghĩa là validation tham gia normalization, và cách stack còn không thuận tiện với episode dài ngắn khác nhau. Adapter mới phải chỉ fit thống kê học được trên train và xử lý dữ liệu theo stream/episode.

### AirExo

Nguồn: [train.py](https://github.com/AirExo/act-in-the-wild/blob/a11e8e69da317b9a64cc39a85b48bc2f397e0475/train.py#L123), [utils.py](https://github.com/AirExo/act-in-the-wild/blob/a11e8e69da317b9a64cc39a85b48bc2f397e0475/utils.py), [wholearm.py](https://github.com/AirExo/act-in-the-wild/blob/a11e8e69da317b9a64cc39a85b48bc2f397e0475/wholearm.py#L195).

- `WholeArmDataset.__load_data` tạo một sample cho **mỗi timestamp start đủ điều kiện** của scene được chọn. Chỉ dừng khi phần thời gian còn lại nhỏ hơn `1/freq`.
- `sample_timestamps(...)` được gọi với khoảng cách mặc định 0, nên start positions không bị lấy thưa theo `freq`. `freq` chủ yếu quyết định khoảng cách các history/future timestamps và điều kiện còn ít nhất một future action. Không mô tả nhầm rằng mỗi scene chỉ lấy `freq × duration` mẫu.
- `find_valid_timestamp_sequence` bắt đầu từ timestamp tương lai; phần chunk còn thiếu lặp giá trị cuối (hoặc zero tùy representation), có mask.
- Wrapper teleoperated mặc định chỉ chọn scene có `sid % 5 == 2` hoặc `sid % 10 == 0`. Trong số đó, scene chia hết 10 là validation; các scene còn lại là train. Wrapper in-the-wild chọn `sid <= 100`, cũng giữ scene chia hết 10 cho validation.
- Mỗi epoch validation chạy trước training, hết loader trong `inference_mode`; không có cap số batch. Loader `shuffle=True`, `drop_last=False` theo mặc định.

Vì vậy AirExo đánh giá dày các **sub-trajectory đã được bộ lọc chọn**, không phải mọi scene có trên đĩa. Dữ liệu, split và sampling khác ACT gốc; không thể so “750 epochs” giữa hai repo mà bỏ qua số optimizer steps và samples.

`utils.get_stats` duyệt các folder `scene*` ở root mà không áp dụng train/val filter. Nếu tính mới theo code đó, validation có thể tham gia thống kê; file stats có sẵn cũng cần truy xuất nguồn. Không dùng nguyên stats AirExo cho CFG5.

### Baseline chính thức RH20T — hướng được đề xuất

Nguồn: [train.py](https://github.com/rh20t/act_baseline/blob/107ce1287a46be3a1a7ea954f8bbdec3bb103288/train.py#L63), [dataset/rh20t.py](https://github.com/rh20t/act_baseline/blob/107ce1287a46be3a1a7ea954f8bbdec3bb103288/dataset/rh20t.py), [dataset/utils.py](https://github.com/rh20t/act_baseline/blob/107ce1287a46be3a1a7ea954f8bbdec3bb103288/dataset/utils.py).

- `load_all_tasks` dùng cleaned index; chọn task ID, `rating > 1`, camera theo `top_down_view`. Script hiện chọn top-down và liệt kê cả CFG1–7. Với CFG5, camera top-down hard-code là `cam_037522062165`.
- Scene `0010` là validation; các scene khác là train. Không tạo offline test split.
- Sau khi bỏ frame muộn hơn `finish_time`, `_get_input_output_frame_id_lists` tạo nhiều window. Với `num_input=1`, `frame_sample_step=1`, T≥K+1, có T−1 start samples cho mỗi episode–camera.
- Với episode ngắn 1≤T<K+1, code lặp frame cuối để đạt horizon rồi tạo K samples. Những sample bổ sung này không tương đương K observation thật.
- Mỗi epoch validation **trước training**, `eval()` + `inference_mode()`, duyệt toàn bộ loader, không cap batch; `shuffle=False` và giữ batch cuối.
- “Toàn bộ” ở đây là toàn bộ windows của cleaned list, sau bộ lọc rating, task và camera. Chưa có số validation samples thực tế cho CFG5 vì chưa xây cleaned index/split và chưa preprocessing.
- README nói cleaned list bỏ frame dư theo thay đổi TCP; vì thế không được tự gán 10 Hz cho chuỗi frame được giữ lại.

### Tổng hợp metric trong cả ba repo

`compute_dict_mean` lấy **trung bình các batch means**, không weight theo số sample của batch. Nếu batch cuối nhỏ, nó vẫn có trọng số ngang batch đầy. Policy loss là `masked_L1.mean() + beta * KL`, với denominator L1 bao gồm cả vị trí padding đã được đặt zero.

Validation truyền target actions vào CVAE, vì vậy latent vẫn được lấy mẫu từ posterior. `torch.inference_mode()` chỉ tắt autograd; nó không vô hiệu hóa sampling trong `reparametrize`. Do đó “dày và deterministic sample index” chưa có nghĩa metric hoàn toàn không ngẫu nhiên.

Tăng density giảm rủi ro bỏ sót các đoạn khó, nhưng windows chồng lấn tương quan cao. Nó không tạo thêm nhiều episode độc lập. Runtime tăng gần theo số samples/batches; decode ảnh và đọc NPY cũng có thể là nút thắt. Khi báo cáo success rate, đơn vị thống kê phải là rollout episode, không phải số windows.

## 4. Những điểm cần xử lý trước triển khai

| Phát hiện | Bằng chứng | Hướng xử lý đề xuất |
|---|---|---|
| Link `rh20t_cleaned_data.json` rỗng trong README; file không có trong tree đã khảo sát | README và `TREE.json` | Tự tạo manifest hợp lệ từ dữ liệu hiện có, ghi rõ khác cleaned subset upstream |
| Loader mở `cam_*/color/<timestamp>.jpg`, dữ liệu máy là MP4 | `__getitem__`, kiểm kê local | Adapter decode/cache ngoài raw dataset |
| Mask sai sau padding | Code tính chiều dài **sau** khi append padding; probe đã chạy xác nhận | Adapter tính `valid_length` trước padding và kiểm thử cuối episode |
| Default `image_size=(360,640)` đưa trực tiếp vào PIL `resize`, API nhận `(width,height)` | Static source | Config ghi rõ W/H, test tensor shape; không coi comment là bằng chứng shape |
| CUDA hard-code, helper parse CLI lại, loader 20 workers | Model factory và training script | Wrapper factory/config/device và số workers phù hợp |
| TCP camera-space và gripper bound 0,11 m | Dataset normalize/clip | Không chuyển thẳng vào actuator; xác minh frame, joint limits và gripper Panda |
| Stats/normalization và loss có semantics khác nhau giữa repo | Utils/policy/dataset | Ghi thành config, không dùng lẫn weights hoặc stats |

Đã chạy `scripts/probe_validation_sampling.py` bằng NumPy có sẵn. Hàm upstream được trích nguyên AST và kiểm tra hash, không sửa snapshot. Với timestamps `[1000,1100,1200]`, horizon=4, upstream tạo `[1000,1100,1200,1200]` nhưng mask là toàn `False`; mask đúng của window này phải kết thúc bằng `True`. Báo cáo: `docs/audit/upstream_sampling_probe.json`.

Probe cũng kiểm chứng AirExo có start positions dày ở dữ liệu 20 Hz ngay cả khi `freq=10`, trong khi future timestamps cách ít nhất 100 ms. Đây là kiểm thử sampling cô lập, không phải smoke test model.

## 5. Checkpoint và resume

Các repo lưu best checkpoint theo validation total loss. ACT gốc lưu định kỳ mỗi 100 epoch; AirExo và RH20T dùng `save_epoch`. Các file chủ yếu chứa `state_dict`; `resume_ckpt` ở AirExo/RH20T nạp weights nhưng khởi tạo optimizer mới, không phục hồi đầy đủ epoch/RNG/optimizer. Gọi đây là warm start khi dùng nguyên code.

Chưa xác minh được policy checkpoint công khai khớp chính xác thí nghiệm RH20T bảng IV, và không có checkpoint như vậy trong workspace. README hướng dẫn pretrain/fine-tune không phải bằng chứng đã cung cấp pretrained policy. Không tải checkpoint hoặc cài torch trong lượt này.

## 6. Đề xuất implementation và protocol validation

**Đề xuất:** dùng lõi model của `rh20t/act_baseline` làm upstream chính; viết CFG5 adapter, model factory, loss/metric wrapper và training loop riêng. Giữ source upstream nguyên vẹn khi có thể. Đây là đề xuất chờ xác nhận, chưa triển khai phụ thuộc.

Lý do: nguồn chính thức gắn trực tiếp với RH20T, action encoding TCP+gripper rõ ràng, model nhận dimension từ config. Những vấn đề hiện thấy tập trung vào loader/factory/protocol; có thể cô lập bên ngoài lõi Transformer. ACT gốc dùng làm đối chiếu thuật toán; AirExo là đối chiếu quan trọng cho dense sampling và CPU/device handling.

Định hướng wrapper validation, **chưa phải config có thể chạy**:

```yaml
validation:
  every_n_epochs: 1
  start_stride: 1
  max_episodes: null
  max_batches: null
  shuffle: false
  seed: 42
  report_posterior_cvae: true
  report_prior_z_zero: true
```

`null` nghĩa là toàn bộ split. Smoke test có thể cap vài batch nhưng phải log cap và số thực đã xét; không dùng kết quả bị cap làm kết quả cuối. Sample manifest cố định, nhóm camera/episode giữ nguyên split. Sau mỗi lần đánh giá lưu: checkpoint hash, config hash, epoch, episode count, observation count, batch count, valid action count và thời gian.

Đề xuất báo cáo `L1_valid = sum(mask * abs(error)) / (d_action * sum(mask))`, KL theo sample và posterior total loss; nếu giữ legacy loss để đối chiếu upstream thì báo cáo riêng. Metric chọn checkpoint phải chốt trước training; có thể dùng prior `z=0` L1 trên validation để gần chế độ rollout hơn. Không dùng test để quyết định.

Đối với representation TCP, cần thêm translation error theo mét, orientation error xử lý quaternion tương đương dấu và gripper error. Rollout report riêng grasp/lift/place/task success từ evaluator vật lý. Không diễn giải normalized L1 thành success rate.
