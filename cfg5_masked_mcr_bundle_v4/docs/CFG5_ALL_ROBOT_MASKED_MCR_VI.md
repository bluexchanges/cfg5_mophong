# MCR cho tất cả robot CFG5, có camera bị thiếu

Giải thích thuật toán, camera mask, nhãn, JPEG, lượng tử hóa và luồng train:
[Tài liệu preprocessing chi tiết](PREPROCESSING_MCR_JOINT_MASKED_CHI_TIET_VI.md).

Nguồn trên server: `RH20T_cfg5/training_data` và `RH20T_cfg5/robot_missing_data`.
Không lọc rating, không loại episode vì thiếu camera. Tất cả ID được audit trước
khi chia train/val/test. Camera có ảnh hợp lệ ở tick được đánh dấu trong
`camera_valid.npy`; camera thiếu không có JPEG và không chạy ResNet.
Đặc trưng của slot thiếu là số 0, nhưng ACT còn nhận attention mask để không xem
slot đó như một quan sát. 8 vị trí camera vẫn có serial cố định.

## Cập nhật đã chốt ngày 12/10/2026

User đã đồng ý tiếp tục với 1317 episode có nhãn, giữ nguyên raw của cả 1321.
Bundle v4 bật `--exclude-known-missing-joints` trong job joint: chỉ cho phép
loại 4 ID dưới đây khi joint ghi thật vẫn thiếu sau khi tìm trong ZIP:

- `task_0039_user_0007_scene_0005_cfg_0005`
- `task_0043_user_0007_scene_0009_cfg_0005`
- `task_0060_user_0007_scene_0008_cfg_0005`
- `task_0106_user_0007_scene_0005_cfg_0005`

Ghi `excluded_episodes` trong audit và `rejected` trong manifest, kèm lý do,
fingerprint và `raw_preserved=true`. Nếu joint được bổ sung lại, episode phải
được audit bình thường. Episode lỗi ngoài danh sách này vẫn chặn preparation.
Không bỏ thêm episode theo rating hoặc số camera. Split mục tiêu mới:
1053 train, 132 val, 132 test; ảnh cache chỉ train+val.

## Hành động đã chọn: joint

Lượt mới dùng 7 góc khớp + độ mở kẹp. MCR có trọng số đã pretrain; ACT được
khởi tạo mới vì E5/E58 dùng representation TCP khác.

- `tcp`: giữ TCP xyz + quaternion wxyz + gripper. Có 90 episode bổ sung không
  có TCP ở bất kỳ serial nào. FK Panda ban đầu khớp 177 điểm của 20 episode
  thuộc nhóm task nhỏ, nhưng kiểm chứng mở rộng có sai lệch góc khoảng 17°.
  Do đó chế độ TCP hiện dừng ở kiểm chứng, chưa được phép tạo nhãn khôi phục.
- `joint`: dùng 7 góc khớp ghi thật (radian) + độ mở kẹp (m), state và target
  8D. Không dùng quaternion/TCP/FK. Một số episode còn thiếu joint.npy;
  phải bổ sung từ archive gốc trước khi chế độ này dùng được toàn bộ dữ liệu.
  Đây là thay đổi representation: cần ACT mới
  với MCR đã pretrain, không được coi như resume checkpoint E5/E58 dùng TCP.
  Action là joint đo được ở tick kế tiếp và gripper command của tick đó,
  không tự gọi nó là lệnh motor ghi trực tiếp.

Audit local đã kiểm tra 1321 episode: 1317 có thể tạo sample; 4 còn thiếu
`transformed/joint.npy`. Job joint tìm đúng file ghi gốc của 4 episode trong
`$HOME/cfg5-pretrain/downloads/simulation.zip`, lưu riêng tại
`data/repairs/cfg5_recorded_joints/<episode>/joint.npy` kèm nguồn và SHA256.
Không sửa raw, không dùng FK hay số 0 thay nhãn. Nếu archive cũng thiếu,
preparation dừng và báo tên episode khi chưa bật lựa chọn loại 4 ID đã duyệt.

Metadata `finish_time=-1` là thiếu chú thích kết thúc: dùng timestamp cuối
của state ghi thật, không loại episode vì giá trị -1.

## Camera thiếu và dropout

Ví dụ camera 3 thiếu cả episode: slot đặc trưng camera 3 bằng 0 ở mọi tick,
`camera_valid[:, 2]=False`. ACT attention bỏ qua token của view đó. Nếu chỉ
thiếu một đoạn, mask chỉ tắt ở đoạn đó. Ảnh trước tick trong tối đa 150ms có
thể dùng lại; ảnh quá cũ bị mask. Không chạy ảnh đen qua ResNet thay quan sát.

Dropout ACT hiện là 0,1; camera mask xử lý dữ liệu vắng, dropout thường để
regularize mạng. Chưa thêm camera dropout ngẫu nhiên. Khi frozen, cache giữ
đặc trưng INT8 và scale FP16; training cập nhật projection, transformer,
latent encoder và action head của ACT. ResNet và BatchNorm được đóng băng.
JPEG là ảnh nén riêng để full fine-tune sau này, không phải đặc trưng.

Nếu bất kỳ episode nào không cung cấp được ít nhất một cặp observation/target
với state thật và một camera thật, script ghi tên/lý do vào báo cáo rồi dừng
cả preparation. Không âm thầm loại episode để báo thành công. Timestamp
không có state/ảnh thật không thể trở thành sample BC; chúng được thống kê.
Ngoại lệ duy nhất là bốn ID đã được user duyệt ở mục cập nhật bên trên.

## Các bước chạy trên server

1. Cài bundle vào repo server bằng `.venv-data/bin/python`; có dry-run và backup.
2. Submit `scripts/slurm/cfg5_robot_all_mcr.sbatch pilot joint` hoặc `pilot tcp`.
   Script audit cả hai nguồn, tạo manifest/split/numeric nhẹ, fit stats chỉ train,
   rồi cache thử 3 episode để đo dung lượng. Chưa train ACT.
3. Khi pilot đạt, submit cùng script với `build joint` hoặc `build tcp`.
   Cache lại train+val; test được khóa trong split, chưa trích xuất ảnh test.
4. Cache `READY.json` mới cho phép train. Với joint mới:

```bash
./.venv-data/bin/python -u scripts/train_mcr_act.py \
  --cache-root data/features/cfg5_robot_all_masked_mcr \
  --stage frozen --from-scratch --output outputs/cfg5_robot_all_joint_mcr \
  --epochs 5 --batch-size 16 --workers 8
```

Lệnh train phải chạy trong job GPU, không chạy trực tiếp trên head node.
Đây là pretrain ACT mới, MCR ResNet50 được giữ đóng băng. Khi full fine-tune
phải bắt đầu từ checkpoint của lượt joint này và sử dụng đúng JPEG/mask/stats.

## File và tài nguyên

- Cấu hình/splits mới: prefix `cfg5_robot_all_masked`; không sửa artifacts cũ.
- Numeric nhẹ: `data/processed/cfg5_robot_all_masked_light`; không có images.npy.
- Cache: `data/features/cfg5_robot_all_masked_mcr`: features.npy INT8,
  scales.npy FP16, camera_valid.npy và rgb.sqlite chứa JPEG có thật.
- Báo cáo: `outputs/data_audit/cfg5_robot_all_masked.json` (hoặc FK blocked).
- Job: 16 Slurm CPU, 128G RAM, 1 GPU; 8 decoder threads theo camera.
- Dùng lưới 10Hz, ảnh causal trước tick trong 150ms; không lấy mọi frame video.
- Pilot đo dung lượng JPEG; build kiểm tra tổng home với ngân sách soft 1,8TB,
  chừa 10GiB. Dung lượng free của NFS không chứng minh quota user.
- Build có thể resume episode hoàn thành đúng contract; `.partial/ERROR.json`
  cần kiểm tra trước khi thử lại, không tự xóa.

## Kiểm thử local

Tests kiểm tra: camera thiếu không ảnh hưởng prediction/gradient; ResNet chỉ
nhận camera thật; decode tuần tự chọn đúng frame trước timestamp, lặp và nhảy
frame; JPEG SQLite lưu đúng camera ID. Fixture rating 0, thiếu 7 camera,
TCP=None vẫn tạo được cache với target joint có thật. MCR cũ được kiểm tra hồi quy.
Chưa chạy full cache hoặc train trên server trong cuộc trò chuyện này.
