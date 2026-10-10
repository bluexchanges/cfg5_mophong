# Hướng dẫn dữ liệu CFG5 → ACT

## Episode, observation và action

Một episode là một lần thực hiện task. Camera và trạng thái robot cùng mô tả episode, nhưng thời điểm lấy mẫu không hoàn toàn trùng nhau. ACT nhận observation hiện tại và dự đoán **action chunk**, một chuỗi hành động tương lai.

Adapter `src/data/act_dataset.py` trả NumPy arrays; chưa cần PyTorch. Ở lượt training, DataLoader có thể gom chúng thành tensor; wrapper phải dùng đúng trường và không chuẩn hóa ảnh lần thứ hai.

| Trường mỗi sample | Shape / dtype | Ý nghĩa |
|---|---|---|
| `images` | `(8,3,360,640)` float32 | 8 RGB view, thứ tự trong config, ImageNet normalization |
| `state` | `(8,)` float32 | TCP xyz + quaternion wxyz + measured width, đã chuẩn hóa |
| `actions` | `(20,8)` float32 | Future targets đã chuẩn hóa; padding bằng 0 |
| `is_pad` | `(20,)` bool | True: loại target khỏi loss |
| `episode`, `timestamp_ms` | string, int | Truy vết sample |

## Dữ liệu thật được dùng

Raw ở `RH20T_cfg5/`, giữ nguyên. Mỗi robot episode sử dụng:

- `metadata.json`: rating, calibration, finish_time; lọc rating ≥2.
- `cam_<serial>/color.mp4`: RGB nén, giải mã bằng OpenCV.
- `cam_<serial>/timestamps.npy`: dictionary với dãy `color`; số timestamp phải bằng số frame.
- `transformed/tcp_base.npy`: dictionary theo camera; mỗi phần tử có timestamp và TCP 7D trong aligned base frame.
- `transformed/gripper.npy`: dictionary theo camera/timestamp; phần tử đầu `gripper_info` là observation, `gripper_command` là target. Transformed width theo mm, nhân 0.001 thành mét.
- `calib/<id>/intrinsics.npy`, `extrinsics.npy`, `tcp.npy`: dùng riêng để tạo rig.

Timeline state chọn camera `037522062165`; **pose vẫn ở base frame**. Kiểm tra pose hữu hạn, norm quaternion sai lệch ≤0.05 rồi normalize về 1. Chọn dấu quaternion liên tục vì q và −q biểu diễn cùng hướng; đầu segment chọn w≥0. Khoảng width chấp nhận là 0–0.085 m; controller Panda sau này phải kiểm giới hạn thực tế của model.

NPY object RH20T cần `allow_pickle=True`, chỉ dùng với raw cục bộ do người dùng cung cấp. Processed NPY là mảng số đọc bằng `allow_pickle=False`. Joint/force không dùng trong baseline này; 14 phần tử joint không có nghĩa robot có 14 khớp. Inventory hiện không tìm thấy depth.mp4, không giả định có depth video.

Task 4 (di chuyển block), 8 (đặt block vào vị trí chỉ định), 13 (đặt lên cân), 14 (lấy khỏi cân), 17 (đặt bút vào ống) giữ được lần lượt 12, 14, 10, 12, 9 episode. Đây là pretraining liên quan, chưa phải demonstrations đặt block vào khuôn đích.

## Đồng bộ bằng timestamp

Không dùng `frame_index/FPS` để suy ra thời gian thu. Mẫu video có container FPS=25 nhưng khoảng timestamp thường khoảng 67 ms. Tạo lưới 100 ms trong khoảng giao của 8 camera/state, kết thúc không quá finish_time.

Tại thời điểm t, chọn `i(t) = max {i | timestamp[i] <= t}`. Giữ bước chỉ khi `0 <= t - timestamp[i(t)] <= 150 ms` cho mọi camera và state. Đây là causal sample-and-hold; không nội suy ảnh hoặc lấy ảnh tương lai. Hai camera có thể lệch đến 150 ms: đây là ngưỡng baseline, không phải thu đồng thời tuyệt đối.

Ví dụ timestamp 100,200,300: observation 250 lấy frame 200; observation 500 bị loại vì frame quá cũ. State lỗi được bỏ; có thể giữ record hợp lệ trước đó nhưng vẫn phải qua ngưỡng tuổi state. Timestamp nguồn có thể lặp lại giữa các tick.

Bước bị loại tạo gap, hai bên thành segment riêng. Chỉ giữ segment ≥3 bước; không nối thành trajectory liên tục.

## Future targets và padding

Lưu `state[t]=[TCP(t),measured_width(t)]`, `action[t]=[TCP(t),commanded_width(t)]`. Sample tại t lấy **action[t+1:t+1+K]**, K=20, không vượt cuối segment. Pose target là future measured TCP, không phải raw arm command.

Segment có 6 bước, sample tại bước 3 (đếm từ 0) còn targets bước 4,5: mask `[False,False,True,...]`. Bước cuối segment không thành sample vì không có future target. Do sample-and-hold, target của tick tương lai vẫn có thể dùng cùng record nguồn với tick trước.

Loss đã triển khai trong src/models/rh20t_act.py:

`L_recon = sum(abs(pred-target) * valid) / (8 * sum(valid))`, `valid = ~is_pad`.

KL của CVAE và trọng số nằm trong configs/act/rh20t_base8.json. Không chia reconstruction loss cho cả padding. Đã kiểm model/loss; chưa training. Quaternion sau denormalize phải normalize/kiểm tra trước controller.

## Split và normalization

Manifest/split được tạo **trước** preprocessing/chunking. Group `(task,user,calib)` giữ mọi scene/camera trong nhóm cùng split. Tỷ lệ mục tiêu 80/10/10, seed42 cho kết quả 47/5/5. User hoặc calibration có thể lặp ở task khác thuộc split khác: không tuyên bố đánh giá người dùng mới hoặc camera placement mới. Human không dùng; khi bổ sung phải kế thừa split robot counterpart.

| Split | Episode | Bước sau lọc | Sample có future target dự kiến |
|---|---:|---:|---:|
| Train | 47 | 4.470 | 3.883 |
| Val | 5 | 526 | 477 |
| Test | 5 | 477 | 429 |

Số sample đếm từ timeline, chưa giải mã toàn bộ video. Tổng 684 segment, median6 bước, dài nhất36; chỉ 275 sample có đủ K20. Cần ghi valid-target count khi đánh giá, không chỉ batch count.

Fit từng chiều bằng train: `z=(x-mean_train)/max(std_train,0.0001)`. State dùng các state rows; action dùng các row có thể làm future target, bỏ row đầu mỗi segment. Tính mỗi row một lần, không nhân theo số chunk chứa nó. Val/test dùng nguyên statistics train. Statistics từ smoke bị đánh dấu và cấm dùng trong full experiment.

Ảnh BGR→RGB, lưu uint8; khi lấy sample chia255, trừ ImageNet mean/chia std. Statistics ảnh cố định không fit từ val/test. Wrapper ACT hiện không normalize ảnh lần hai.

## Lưu trữ và tính toàn vẹn

Processed episode gồm images.npy `(T,8,H,W,3)` uint8, state.npy, action.npy, segment_ids.npy, timestamp/index arrays, episode.json. Đọc video tuần tự từng camera, ghi ảnh bằng memory map; không load toàn dataset vào RAM. Adapter giữ map một episode mỗi worker; RAM batch vẫn phải chọn phù hợp khi dùng PyTorch.

Toàn subset cần khoảng **30.26 GB** ảnh RGB không nén. Lượt này chỉ tạo 2 episode ×8 bước, khoảng88.5 MB ảnh. Chưa tự preprocess toàn bộ. Có thể nghiên cứu nén/shard khi mở rộng, nhưng phải giữ semantics/contract.

Output ghi vào `.partial`, rename khi hoàn tất; nếu lỗi giữ ERROR.json. Từ chối ghi đè output/partial; chọn output-root mới để chạy thử lại. CLI chặn output trong raw. Metadata ghi config hash, split hash, source fingerprint. Fingerprint hash file nhỏ; video kiểm size+mtime, không phải hash toàn nội dung video.

Manifest kiểm timestamp/schema/frame-count và decode frame đầu. Chỉ full preprocessing mới phát hiện lỗi decode ở giữa video. Xem lệnh ở [PIPELINE_GUIDE_VI.md](PIPELINE_GUIDE_VI.md). Shape đúng chưa chứng minh ACT điều khiển được robot.

Nguồn: [RH20T API đã khóa commit](https://github.com/rh20t/rh20t_api/tree/aa3124434729ed622109a29b2cbb9f3bbb1c5eeb), [trang dataset](https://rh20t.github.io/#download), PDF trong references/papers. Sampling/split/base-frame multi-view là quyết định dự án, không phải toàn bộ protocol gốc của paper.
