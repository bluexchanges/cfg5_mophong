# Preprocessing hiện tại: CFG5 robot → JPEG + đặc trưng MCR → ACT

Ngày cập nhật: 12/10/2026. Tài liệu mô tả code hiện có trong `local_source_(1)`.
Đường dẫn tương đối bên dưới được tính từ thư mục gốc `cfg5_act` trên server.

**Hướng lưu cache hiện tại là v5, feature trước/JPEG sau:**
[Quy trình theo giai đoạn và lệnh chạy](STAGED_MCR_CACHE_VI.md).
Tài liệu này giữ phần giải thích thuật toán; các mục layout/lệnh combined
của v4 bên dưới là để đối chiếu. Dùng lệnh v5 trong tài liệu liên kết cho lượt mới.

## 1. Mục tiêu và quyết định đã chốt

Chuẩn bị cả hai nguồn robot CFG5 để pretrain ACT mới:

```text
RH20T_cfg5/
├── training_data/          # 663 episode
└── robot_missing_data/     # 658 episode bổ sung
```

Tổng đầu vào: **1.321 episode robot**. Không nhập demonstration human.
Không lọc theo rating, không loại episode chỉ vì thiếu một số camera.
Mỗi episode vẫn phải có dữ liệu thật đủ để tạo ít nhất một sample có target.
Nếu episode không đáp ứng, preparation báo lỗi và dừng toàn bộ; không tự bỏ
episode đó rồi báo thành công.

Representation đã chọn là **7 góc khớp Panda, đơn vị radian, cộng độ mở kẹp,
đơn vị mét**. MCR ResNet50 dùng trọng số đã pretrain và được đóng băng trong
giai đoạn đầu; ACT khởi tạo mới. Checkpoint E5/E58 cũ dùng TCP xyz/quaternion
không được dùng như checkpoint resume của policy góc khớp này.

Lý do chọn joint: có episode không có TCP ghi thật. Kiểm chứng FK mở rộng
chưa đạt, nên luồng joint không dùng FK để tạo nhãn và không cần TCP.

## 2. Sơ đồ toàn bộ quá trình

```text
Raw: metadata + joint + gripper + timestamps camera + MP4
                          ↓
Audit tất cả episode, khôi phục file joint ghi gốc nếu raw thiếu
                          ↓
Tạo lưới 10 Hz; ghép state và ảnh theo thời gian, tạo camera mask
                          ↓
Chia train/val/test; lưu numeric nhẹ; tính normalization từ train
                          ↓
Đọc tuần tự MP4 tại các frame đã chọn → ảnh 640×360 → JPEG quality 90
                          ↓
Giải mã chính JPEG đó → resize/crop/normalize → MCR ResNet50 đóng băng
                          ↓
Lưu feature INT8 + scale FP16 + mask + JPEG + numeric
                          ↓
Frozen training: đọc feature đã lưu → train ACT
                          ↓
Full fine-tune sau này: đọc JPEG → chạy ResNet → cập nhật ResNet và ACT
```

Preprocessing không phải chỉ lọc timestamp: nó còn chuẩn bị nhãn, mask,
split, normalization, JPEG và chạy mạng trích xuất đặc trưng một lần trước
khi train. Bản thân bước này **chưa huấn luyện ACT**.

## 3. Đọc raw và kiểm tra nguồn

### 3.1. Nhận diện episode

Tên episode phải theo mẫu:

```text
task_XXXX_user_XXXX_scene_XXXX_cfg_0005
```

Script tìm trong các thư mục nguồn đã khai báo. Nếu ID episode bị trùng ở
hai nguồn hoặc đường dẫn nằm ngoài raw root, script dừng để tránh nhập nhầm.
Các báo cáo chất lượng, thư mục human và file không đúng mẫu không được
coi là episode robot.

**Ngoại lệ đã được user duyệt ngày 12/10/2026:** tiếp tục với 1.317 episode có
nhãn; giữ nguyên raw của 4 episode thiếu joint. Bản v4 chỉ loại đúng bốn ID ở
mục 3.3 nếu vẫn không có joint ghi thật. Lỗi ở episode khác vẫn dừng cả job.
Không bỏ episode theo rating hay số camera.

### 3.2. File đọc trong mỗi episode

| File | Vai trò |
| --- | --- |
| `metadata.json` | Task/user/calibration, rating để báo cáo, thời điểm kết thúc nếu có |
| `transformed/joint.npy` | Góc khớp robot ghi thật theo timestamp và serial |
| `transformed/gripper.npy` | Độ mở đo được và gripper command theo timestamp |
| `cam_<serial>/timestamps.npy` | Thời điểm của từng frame màu |
| `cam_<serial>/color.mp4` | Ảnh camera đã nén thành video |

Trong chế độ joint, `transformed/tcp_base.npy` không được dùng làm state hay
target. Dữ liệu NPY raw dạng object được đọc theo schema của dataset; các
numeric NPY đầu ra là array số thông thường, đọc với `allow_pickle=False`.

### 3.3. Bổ sung joint thiếu từ archive

Audit local hiện thấy **1.317 episode có thể tạo sample, 4 episode thiếu
file joint**:

- `task_0039_user_0007_scene_0005_cfg_0005`
- `task_0043_user_0007_scene_0009_cfg_0005`
- `task_0060_user_0007_scene_0008_cfg_0005`
- `task_0106_user_0007_scene_0005_cfg_0005`

Job joint tìm đúng member `<episode>/transformed/joint.npy` trong archive
gốc `$HOME/cfg5-pretrain/downloads/simulation.zip`. Chỉ đọc file joint cần
thiết, kiểm tra CRC của ZIP, schema và các record joint hữu hạn, rồi lưu tại:

```text
data/repairs/cfg5_recorded_joints/<episode>/
├── joint.npy
└── source.json            # archive, member, CRC, SHA256, số record
```

Raw không bị sửa. Đây là bổ sung **file ghi thật từ nguồn gốc**, không phải
ước lượng joint. Log server job 14285 xác nhận: 666 episode thiếu joint trong
raw server, đã lấy được 662 file từ ZIP, còn đúng 4 ID trên không có file trong
ZIP. User đã duyệt loại riêng 4 ID này khỏi dataset dùng train/val/test.
`--exclude-known-missing-joints` bật ngoại lệ đó; nếu không bật, vẫn dừng.
Không thay nhãn thiếu bằng 0. Audit và manifest lưu danh sách loại và lý do.
Nếu một ID có joint raw hoặc joint bổ sung trở lại, nó được kiểm tra bình
thường, không tự loại theo ID. File joint có nhưng lỗi cũng không được bỏ qua.

## 4. State và action được tạo thế nào?

### 4.1. Chọn stream robot

Ưu tiên serial `037522062165`. Nếu stream này không cung cấp ít nhất 2 record
joint/gripper hợp lệ, thử các serial còn lại theo thứ tự cấu hình. Khi đã
chọn, dùng một stream cho episode, không trộn từng phần từ nhiều stream.
Serial này là khóa đồng bộ state trong dữ liệu; việc chọn stream state không
bắt buộc camera cùng serial phải có video hợp lệ.

Tại timestamp của record robot, joint và gripper phải có dữ liệu đồng bộ.
Lấy **7 phần tử đầu** của vector joint làm góc khớp; phần vận tốc nếu có
không được dùng trong representation này.

```text
state tại tick i  = [q1, q2, q3, q4, q5, q6, q7, gripper_measured]
action.npy tại i  = [q1, q2, q3, q4, q5, q6, q7, gripper_command]
```

Góc khớp giữ đơn vị radian. Gripper được đổi từ mm sang m bằng hệ số `0.001`.
Giá trị đo và command phải trong `[0, 0.085]` m. Record có số không hữu hạn,
thiếu joint/gripper hoặc gripper ngoài khoảng này không được tạo thành nhãn
thật; lý do được ghi vào báo cáo.

**Ý nghĩa action:** góc khớp robot đã ghi ở thời điểm tương lai và command
kẹp tương ứng. Đây là target imitation từ trajectory ghi được, không phải
khẳng định dataset đã ghi trực tiếp lệnh motor cho 7 khớp.

### 4.2. Target của một sample ACT

`action.npy` lưu giá trị theo từng tick. Dataset dùng `action_target_offset=1`:

```text
Observation: state và các camera tại t
Target:      action tại t+0,1 s, t+0,2 s, …, t+2,0 s
```

Chunk tối đa 20 bước ở 10 Hz. Khi gần cuối segment, chỉ còn một phần chunk:
phần còn lại được padding 0 và đánh dấu `is_pad=True`, không tính loss như
nhãn thật. Mask padding hành động khác với mask camera.

## 5. Lưới thời gian và ghép camera

### 5.1. 10 Hz là tham số preprocessing

`control_hz=10` tạo một tick mỗi **100 ms**. Đây là lưới policy do cấu hình
hiện tại chọn, không phải kết luận camera hay robot raw luôn chạy đúng 10 Hz.
Không dùng mọi frame MP4 làm sample.

Tick đầu là bội số 100 ms đầu tiên không sớm hơn state hợp lệ đầu tiên.
Tick cuối không vượt state cuối; nếu `finish_time` hợp lệ thì còn bị giới
hạn bởi metadata. `finish_time=-1` được hiểu là thiếu annotation kết thúc,
nên dùng thời điểm cuối của state ghi thật, không loại episode.

### 5.2. Quy tắc causal: lấy quan sát gần nhất trước tick

Cho tick `t`, chọn timestamp nguồn lớn nhất thỏa `timestamp ≤ t`.
State và từng camera được tìm độc lập; không yêu cầu các camera có timestamp
trùng tuyệt đối và không lấy ảnh/state từ tương lai.

Ví dụ các ảnh camera tại `3,13 s`, `3,18 s`, `3,23 s`:

- Tick `3,20 s` dùng ảnh `3,18 s`, tuổi ảnh 20 ms.
- Không chọn ảnh `3,23 s` dù ảnh đó gần tick.

Tuổi state và tuổi ảnh tối đa hiện đều là **150 ms**. Một ảnh có thể được
dùng lại ở tick sau nếu vẫn trong ngưỡng. Không nội suy pixel hay joint trong
luồng này. Mỗi ảnh được lưu kèm vị trí camera cố định theo cấu hình.

### 5.3. Quyết định tick và segment

Một tick hợp lệ cần state/action thật đủ mới và **ít nhất một camera thật**
đủ mới. Tick không đáp ứng được thống kê là unused, không dựng ảnh/nhãn giả.

Các tick hợp lệ liên tiếp tạo một segment. Segment cần tối thiểu 2 tick để
có observation và target kế tiếp. Khi có khoảng trống, tạo segment mới;
chunk hành động không vượt khoảng trống này. Tick cuối segment không làm
observation có target kế tiếp. Vì vậy số tick lưu có thể lớn hơn số sample.

Giữ mọi episode không có nghĩa mọi timestamp đều tạo được sample. Nếu một
episode không có segment đủ dài, script dừng cả preparation và báo lý do;
ngoại lệ duy nhất là 4 ID thiếu joint đã duyệt, không phải lọc theo độ dài.

## 6. Camera thiếu và lỗi được xử lý ra sao?

8 slot camera được giữ theo thứ tự serial:

| Slot | Serial |
| --- | --- |
| 1 | `036422060215` |
| 2 | `037522062165` |
| 3 | `104122061850` |
| 4 | `104122063678` |
| 5 | `104422070042` |
| 6 | `104422070044` |
| 7 | `105422061350` |
| 8 | `135122079702` |

`camera_valid` có shape `(N,8)`, dtype bool; `frame_indices` có shape `(N,8)`.
Slot thiếu có `camera_valid=False` và `frame_indices=-1`.

| Trường hợp | Cách xử lý hiện tại |
| --- | --- |
| Không có folder camera, MP4 hoặc timestamp | Mask camera đó trong cả episode; dùng camera khác |
| Video không mở được hoặc frame đầu không giải mã được | Đánh dấu camera không dùng được, ghi lý do |
| Timestamp rỗng, không tăng nghiêm ngặt, số frame video lệch số timestamp | Mask cả camera trong episode để tránh ghép sai frame |
| Stream kết thúc sớm hoặc có khoảng thời gian không có ảnh đủ mới | Mask camera tại những tick bị ảnh hưởng |
| Camera còn ảnh trước tick trong ngưỡng 150 ms | Dùng ảnh gần nhất trước tick |
| Mọi camera đều không hợp lệ tại tick | Tick không tạo sample, được thống kê |
| Lỗi decode xuất hiện giữa video khi build cache | Dừng episode, ghi `.partial/ERROR.json`; chưa tự chuyển lỗi này thành mask |

Ví dụ camera 3 thiếu cả episode:

```text
camera_valid tại mọi tick: [1, 1, 0, 1, 1, 1, 1, 1]
feature slot camera 3:    toàn số 0
JPEG camera 3:            không lưu
ResNet cho camera 3:      không chạy
ACT attention camera 3:   bị mask
```

Code có thể tạo buffer pixel 0 để ghép tensor đủ 8 slot, nhưng chỉ chọn slot
hợp lệ trước khi chạy ResNet. Không dùng ảnh đen như observation thật.
Feature 0 chỉ là placeholder; attention mask mới ngăn ACT coi view đó là
dữ liệu. Khi full fine-tune, mask vẫn được sử dụng theo cùng cách.

Mask giúp sử dụng các view còn lại; nó không khôi phục thông tin hình ảnh
đã mất. Thiếu nhiều camera có thể làm việc dự đoán khó hơn.

## 7. Chia dữ liệu và normalization

Chia theo group **`(task, user, calib)`**, không chia ngẫu nhiên từng frame.
Một group nằm hoàn toàn trong train, val hoặc test. Thuật toán dùng seed
42, random có ràng buộc và tìm cách đạt số episode theo tỷ lệ 80/10/10.
Nếu không thể đạt số lượng khi giữ nguyên group, script dừng để báo.

Nếu đủ 1.321 episode và phép chia khả thi, số lượng mục tiêu là
**train 1.057, val 132, test 132**. Đây là mục tiêu tính từ tỷ lệ, chưa phải
kết quả split đã tạo thành công trên server. Membership của lượt mới có
thể khác lượt 663 episode cũ; không sửa split cũ.

Với lựa chọn đã duyệt dùng 1.317 episode, mục tiêu là **train 1.053, val 132,
test 132**. Phép chia group được kiểm tra local trên metadata của 1.317
episode audit; job server sẽ tạo artifacts chính thức của lượt mới.

Tính mean/std riêng cho 8 chiều state và 8 chiều action **chỉ từ train**.
Action statistics chỉ dùng những tick có thể làm target theo offset và
segment. Std có ngưỡng dưới `0.0001` để tránh chia cho 0.

```text
state_normalized  = (state  − train_state_mean)  / train_state_std
action_normalized = (action − train_action_mean) / train_action_std
```

Val/test dùng lại statistics train. Numeric nhẹ được chuẩn bị cho cả ba
split; cache ảnh/feature hiện chỉ build train và val. Test được giữ trong
split nhưng chưa cache ảnh để dùng đánh giá cuối riêng.

## 8. Từ MP4 thành JPEG và đặc trưng

### 8.1. Đọc video tuần tự

Mỗi camera có một decoder. Decoder tiến tuần tự đến frame index đã chọn,
giải mã frame cần dùng; nếu tick kế dùng cùng frame index thì dùng lại ảnh
trong bộ nhớ. Không seek ngẫu nhiên lại từ đầu cho từng sample khi train.
Codec vẫn có thể phải xử lý các frame phụ thuộc trên đường đến frame đích.

Job cache dùng 8 decoder threads và batch trích xuất 8 tick theo mặc định
sbatch hiện tại. Đây là tham số của bước tạo cache, không phải số worker hay
batch size của training ACT.

### 8.2. Ảnh JPEG

Ảnh chọn được resize về **640×360**, nén JPEG **quality 90** rồi lưu trong
SQLite. Database có bảng `rgb(tick, camera, jpeg)`; camera dùng index 0–7.
Chỉ lưu hàng của camera hợp lệ. Không xuất hàng triệu file JPEG rời.

JPEG là nén có mất mát; quality 90 không có nghĩa ảnh giữ nguyên mọi pixel.
Sau khi encode, code giải mã **chính JPEG vừa tạo** làm đầu vào ResNet. Nhờ
đó ảnh dùng tạo feature và ảnh dùng full fine-tune sau này cùng nguồn JPEG.

### 8.3. Tiền xử lý cho MCR

Với mỗi camera hợp lệ:

1. Chuyển BGR của OpenCV sang RGB.
2. Chuyển pixel uint8 thành float và chia 255.
3. Resize cạnh ngắn thành 256, bilinear có antialias.
4. Center crop thành **224×224**.
5. Normalize RGB với mean `[0.485, 0.456, 0.406]` và std `[0.229, 0.224, 0.225]`.
6. Chạy trunk MCR ResNet50, bỏ đầu phân loại, lấy spatial feature.

Đầu ra mỗi camera là **`(2048,7,7)`**: 49 vị trí không gian, mỗi vị trí có
2.048 kênh số. Đây không phải nhãn như “vật thể” hay “kẹp”; đó là biểu diễn
thị giác mạng đã học. Không dùng global pooling để rút còn một vector 2.048
chiều trong bản hiện tại; ACT vẫn nhận cấu trúc không gian.

MCR weights được kiểm tra SHA256:

```text
21560db9e795384a7b73b0626d33d6525f9d4c5de11fca6ded52089aab14f97c
```

### 8.4. Lượng tử hóa feature để giảm dung lượng

Với mỗi tick, camera và kênh, tính một scale trên 49 vị trí:

```text
scale = max(max(abs(feature trên 7×7)) / 127, 1e−6)
scale được lưu FP16
feature_int8 = clip(round(feature / scale), −127, 127)
feature_khôi_phục = float32(feature_int8) × float32(scale)
```

INT8 không giữ feature chính xác tuyệt đối. Code ghi
`quantization_max_absolute_error` để biết sai số lượng tử lớn nhất. Chỉ số
này không phải task success rate và không tự chứng minh policy tốt.

Slot camera thiếu lưu feature 0 cùng scale dương nhỏ; mask vẫn tắt slot đó.

## 9. Các file đầu ra và dung lượng

### 9.1. Numeric nhẹ trước khi tạo feature

```text
data/processed/cfg5_robot_all_masked_light/<episode>/
├── state.npy            # float32 (N,8)
├── action.npy           # float32 (N,8), giá trị theo tick; dataset dùng offset 1
├── timestamps_ms.npy    # int64 (N,)
├── segment_ids.npy      # int32 (N,)
├── frame_indices.npy    # int64 (N,8), -1 là thiếu
├── camera_valid.npy     # bool (N,8)
└── episode.json         # nguồn, chất lượng, cấu hình, split, hash
```

Không có `images.npy` RGB lớn trong luồng này.

### 9.2. Cache phục vụ train

```text
data/features/cfg5_robot_all_masked_mcr/
├── context.json
├── storage_report.json
├── READY.json                      # chỉ có khi build hoàn tất
└── <episode>/
    ├── features.npy                # INT8 (N,8,2048,7,7)
    ├── scales.npy                  # FP16 (N,8,2048,1,1)
    ├── rgb.sqlite                  # JPEG của camera thật
    ├── camera_valid.npy
    ├── state.npy
    ├── action.npy
    ├── timestamps_ms.npy
    ├── segment_ids.npy
    └── episode.json
```

Feature và scale hiện lưu **dense đủ 8 slot**, kể cả slot mask bằng 0.
Vì vậy thiếu camera tiết kiệm JPEG và tính ResNet, nhưng không tự giảm kích
thước `features.npy` theo số camera thật.

Payload feature + scale mỗi tick:

```text
INT8:  8 × 2048 × 7 × 7 × 1 byte = 802.816 byte
FP16:  8 × 2048 × 1 × 1 × 2 byte =  32.768 byte
Tổng:                              835.584 byte ≈ 0,797 MiB/tick
```

Ví dụ 100.000 tick train+val cần khoảng **77,8 GiB feature + scale**, chưa
cộng JPEG, numeric, SQLite, metadata và overhead. Tổng bộ 1.321 episode phải
đo theo số tick thực tế và pilot, không lấy ước lượng cũ của pick_place làm
dung lượng của bộ này. Cache vẫn có thể lớn; frozen feature không phải chỉ
lưu một file trọng số ResNet.

### 9.3. Config, split, stats và báo cáo

```text
configs/data/cfg5_robot_all_masked.json
configs/act/cfg5_robot_all_masked_model.json
configs/act/cfg5_robot_all_masked.json
data/splits/cfg5_robot_all_masked_manifest.json
data/splits/cfg5_robot_all_masked_splits.json
data/splits/cfg5_robot_all_masked_normalization.json
outputs/data_audit/cfg5_robot_all_masked.json
```

Báo cáo ghi episode bị chặn, rating (không lọc), camera lỗi/thiếu, số tick
hợp lệ theo camera, số tick không dùng, stream state đã chọn và nguồn thời
điểm kết thúc. Không chấm chất lượng bằng task success trước khi có rollout.

## 10. Pilot, build và kiểm tra tính nhất quán

**Pilot** audit và chuẩn bị numeric toàn bộ trước, sau đó tạo cache thử 3
episode train chọn bằng seed 42. Nó đo dung lượng và lỗi lượng tử; không
train ACT và không tự chuyển sang build khi xong.

Ước lượng trước pilot dùng upper estimate RGB nên có thể rất lớn. Sau pilot,
ước lượng JPEG dùng bytes/tick lớn nhất của episode hoàn thành ×1,5, cộng
feature/scale, phần dự phòng và 1 GiB. Đây là ước lượng, không phải cam kết.

**Build** tạo cache train+val còn thiếu. Phải có ít nhất 3 episode pilot đã
hoàn thành đúng contract. Episode hoàn thành hợp lệ được kiểm tra rồi bỏ
qua bằng `SKIP_VERIFIED`. Không train trong job build.

Kiểm tra gồm config/split/stats/backbone/preprocessing hash, numeric hash,
shape/dtype, camera mask và các hàng JPEG tương ứng. `--full-check` còn kiểm
hash payload feature/scale/JPEG và SQLite. Fingerprint raw dùng hash các
file không phải MP4; MP4 dùng tồn tại/kích thước/mtime, không hash toàn video.

Một episode được ghi vào `<episode>.partial`, hoàn tất mới rename thành
folder chính thức. Lỗi tạo `ERROR.json`; script không tự xóa `.partial`.
Lock ngăn hai process cùng build một cache root. Đổi backbone, preprocessing,
split hoặc stats cần cache phù hợp; không âm thầm trộn cache khác nhau.

Build xong mới có `READY.json` và marker `MCR_CACHE_READY`. Trainer yêu cầu
READY đúng train+val và contract. `MCR_PILOT_OK` chỉ có nghĩa pilot xong.

Ngân sách home mặc định là **1,8 TB thập phân**, chừa 10 GiB. Script đo tổng
home bằng `du`, kiểm tra thêm free filesystem, và theo dõi dung lượng thêm
trong build. Nó không đọc quota riêng từ storage controller; `df` không
chứng minh quota user. Tổng home còn gồm raw, ZIP, cache khác và checkpoints.

## 11. Sau preprocessing: train mạng nào?

### Frozen stage

Dataset đọc `features.npy` bằng memmap, khôi phục INT8 bằng scale, lấy state,
camera mask và target chunk. Không đọc MP4, không giải mã JPEG và không chạy
ResNet trong mỗi batch frozen.

ACT học visual projection `2048 → hidden_dim`, transformer, latent encoder
và action head. MCR weights không có gradient; BatchNorm giữ chế độ eval,
không cập nhật running statistics. Dropout ACT hiện **0,1**.

Camera mask xử lý view vắng; dropout thông thường regularize mạng. Bản hiện
tại chưa có camera dropout ngẫu nhiên. Không tăng dropout tự động vì dữ
liệu đã thiếu view. Augmentation ảnh ngẫu nhiên không được thêm trong cache
hiện tại: feature đã cố định theo transform resize/crop đã ghi trong contract.

### Full stage

Bắt đầu từ checkpoint MCR+ACT đã học ở frozen stage. Dataset đọc JPEG và
mask; normalize theo cùng transform, chạy ResNet rồi ACT. Cả backbone và
ACT có thể cập nhật, backbone dùng learning rate riêng.

Khi ResNet đã thay đổi, feature cache cũ không đại diện cho backbone mới.
Full stage đọc JPEG thay vì dùng feature đó. Nếu muốn freeze lại backbone
đã fine-tune, cần tạo cache mới cho đúng weights; bản hiện tại không tự làm
việc chuyển này.

Checkpoint và feature cache là hai loại file khác nhau: checkpoint chứa
tham số/trạng thái training; cache chứa dữ liệu đã chuẩn bị để train nhanh.

## 12. Lệnh chạy và đọc trạng thái trên server

Các file source mới phải được cài lên server trước. Cài bundle chưa có nghĩa
job đã chạy. Trong terminal SSH:

```bash
cd "$HOME/cfg5-pretrain/edabk-physical-ai-robot/04-Src/software/cfg5_act"
mkdir -p "$HOME/cfg5-pretrain/logs"

# Pilot: audit toàn bộ + numeric nhẹ + cache thử 3 episode; chưa train
if P=$(sbatch --parsable scripts/slurm/cfg5_robot_all_mcr.sbatch pilot joint); then
  echo "Pilot job: $P"
fi
```

Kiểm tra bằng các lệnh riêng ở prompt. Nếu đang `tail -F`, nhấn Ctrl+C để
trở về prompt trước khi nhập lệnh khác; việc này không dừng job:

```bash
squeue -j "$P" -o "%.18i %.24j %.2t %.10M %R"
sacct -j "$P" -X --format=JobID,State,ExitCode,Elapsed
tail -n 40 "$HOME/cfg5-pretrain/logs/cfg5-all-mcr-$P.out"
tail -n 40 "$HOME/cfg5-pretrain/logs/cfg5-all-mcr-$P.err"
```

Sau khi pilot `COMPLETED 0:0`, có `MCR_PILOT_OK`, không có episode bị chặn
và ngân sách dung lượng phù hợp, submit build:

```bash
if C=$(sbatch --parsable scripts/slurm/cfg5_robot_all_mcr.sbatch build joint); then
  echo "Build job: $C"
fi
tail -F "$HOME/cfg5-pretrain/logs/cfg5-all-mcr-$C.out"
```

Job yêu cầu 16 Slurm CPU, 128G RAM, 1 GPU, tối đa 72 giờ. Chạy Python trực
tiếp trong sbatch; không dùng `srun` do lỗi DNS node đã gặp. Không chạy build
cache hay train nặng trực tiếp trên head node.

Sau READY, lệnh trainer sau phải đặt trong **job GPU**:

```bash
./.venv-data/bin/python -u scripts/train_mcr_act.py \
  --cache-root data/features/cfg5_robot_all_masked_mcr \
  --stage frozen --from-scratch \
  --output outputs/cfg5_robot_all_joint_mcr \
  --epochs 5 --batch-size 16 --workers 8
```

Đây là ví dụ lượt frozen 5 epoch, không phải lệnh đã được submit. Để tiếp
tục cùng lượt dùng `--resume <latest.pt>` thay `--from-scratch`, giữ contract
và runtime đúng như lượt trước. Checkpoint chỉ giúp tiếp tục phần đã lưu;
không tự làm job vượt walltime hoặc tự submit job tiếp theo.

## 13. Code thực hiện từng phần

| File | Vai trò |
| --- | --- |
| `scripts/prepare_cfg5_masked_mcr.py` | Audit, bổ sung joint, tạo numeric/config/split/stats |
| `src/data/cfg5_masked.py` | Discover, ghép timeline, camera mask, decoder tuần tự, kiểm tra nguồn |
| `src/data/splits.py` | Chia group có seed, không tách episode/group giữa split |
| `src/data/act_dataset.py` | Normalization, index sample, target offset, segment và action padding |
| `scripts/prepare_mcr_cache.py` | Pilot/build/estimate/check, ngân sách dung lượng, READY |
| `src/data/feature_cache.py` | JPEG SQLite, INT8/scale, cache episode, loader frozen/full |
| `src/models/mcr_act.py` | MCR transform, backbone và ACT attention mask |
| `scripts/train_mcr_act.py` | Train ACT frozen hoặc full fine-tune |
| `scripts/slurm/cfg5_robot_all_mcr.sbatch` | Job GPU cho preparation + pilot/build |

## 14. Trạng thái đã kiểm chứng

- Audit local đã rà 1.321 episode; sau xử lý metadata kết thúc `-1`, 1.317
  episode có thể tạo sample; 4 episode cần bổ sung joint ghi gốc.
- Bộ kiểm thử local có 14 test: 13 đạt, 1 skip vì không có asset weights
  MCR tại vị trí test yêu cầu trong source này. Không coi test đó là đạt.
- Test đã kiểm tra mask không làm view thiếu ảnh hưởng prediction/gradient,
  ResNet chỉ nhận camera thật, decode causal, JPEG đúng slot, joint fallback
  từ ZIP và fixture rating thấp thiếu 7 camera vẫn tạo cache khi có nhãn thật.
- Server xác nhận ZIP không có joint của 4 ID trên. Ngoại lệ loại đúng 4 ID
  đã được user duyệt. Chưa build toàn bộ cache hay train bộ gộp này.

Tài liệu này mô tả cơ chế và cấu hình hiện có. Chỉ dùng log, READY, báo cáo
audit và checkpoint của job thực tế để xác nhận tiến độ/chất lượng trên server.
