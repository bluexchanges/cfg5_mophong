# Tự chọn, chuyển đổi, lưu và xem dữ liệu CFG5

**Cập nhật sau khi phân loại toàn bộ:** Đọc [CFG5_FULL_SPLIT_AND_RUN_VI](CFG5_FULL_SPLIT_AND_RUN_VI.md) cho split full và lệnh tự chuyển đổi. Đã move2.515 episode vào `RH20T_cfg5/training_data/task_XXXX/` hoặc `missing_data/task_XXXX/...`;663 robot episode trên92 task đạt bộ lọc ACT. `RH20T_cfg5/human_robot_index.csv` và CSV từng task giữ liên hệ human–robot theo tên. `configs/data/cfg5_all8.json`, `data/splits/cfg5_all_manifest.json` và `cfg5_all_splits.json` là phạm vi mới; **chưa preprocess/train full**. Các số57/47/5/5 và lệnh không truyền `--config/--manifest/--splits` dưới đây chỉ minh họa subset lịch sử. Không dùng default đó để nghĩ đã xử lý toàn CFG5. Đường video raw khi mở thủ công giờ là `RH20T_cfg5/training_data/task_XXXX/<episode>/cam_<serial>/color.mp4` hoặc trong `missing_data/` nếu episode không đạt.

Thứ tự mới theo yêu cầu ngày2026-09-28: **tích hợp RH20T ACT baseline → bạn rà/chọn và preprocessing CFG5 → chuẩn bị pretraining**. Chưa thu demonstrations task đích. Tài liệu này giúp thao tác dữ liệu; không có lệnh tự bắt đầu training.

## 1. Phân biệt các việc đã làm

| Công việc | Trạng thái thực tế |
|---|---|
| Quét các robot episode thuộc task4/8/13/14/17 |99 episode;57 được nhận,42 bị loại. Không phải57 frame. |
| Lập timeline bằng timestamp/state và kiểm video cơ bản |Đã làm cho subset; kiểm video mở được/frame đầu/count, chưa decode toàn bộ. |
| Chia nhóm train/val/test |47/5/5 episode, lưu JSON. |
| Chuyển video thành arrays lưu trên đĩa |Chỉ2 episode thử, mỗi episode8 bước; không phải đã chuyển xong57 episode. |
| Code chuyển đổi |Đã có `src/data/cfg5.py` và các CLI. Chạy full bằng lệnh bên dưới khi bạn chốt dữ liệu. |
| ACT |Đã tải source, tích hợp và kiểm inference/backward; chưa có pretrained ACT checkpoint, optimizer chưa cập nhật weights. |

Episode = một lượt thu thao tác. Step = một mốc thời gian10 Hz còn hợp lệ. Sample ACT = một observation tại t + một chunk tối đa20 future targets. Một episode tạo nhiều samples, nhưng không phải mọi step đều tạo sample.

## 2. Mở project và xem dữ liệu hiện có

Mở PowerShell, mọi lệnh bên dưới chạy tại root:

```powershell
Set-Location 'D:\Download\mophong_RH20T'
& '.\.venv-data\Scripts\python.exe' --version
Start-Process '.\outputs\cfg5_review\review.html'
Start-Process '.\outputs\processed_preview_train\index.html'
```

Hai trang đã được tạo sẵn trong lượt tích hợp. Trang review lịch sử có57 accepted,42 rejected, task/rating/split, số bước/segment/sample/chunk đủ và link8 video raw của mỗi episode. Link đã được cập nhật sau move. Một số trình duyệt chặn mở video file cục bộ từ HTML; khi đó dùng `RH20T_cfg5\training_data\task_XXXX\<episode>\cam_<serial>\color.mp4` trong Explorer. Không suy thời gian thu bằng thanh phát video; preprocessing dùng timestamp thật.

Trang processed preview hiện là **train smoke đã lưu**, không phải toàn episode. Nó có ảnh8 camera, shape/type các NPY, state8D, targets thật và mask. Bạn có thể dùng PNG/bảng này trong trình bày nhưng ghi rõ “smoke8 bước”.

Muốn tạo bảng rà soát mới, dùng output chưa tồn tại selection:

```powershell
& '.\.venv-data\Scripts\python.exe' -X utf8 scripts/review_cfg5.py --output outputs/cfg5_review_v2
```

Script chỉ đọc manifest/split, không decode lại tất cả video. Không ghi đè `selection.json` đã có để tránh mất lựa chọn của bạn.

## 3. Chọn dữ liệu theo tiêu chí nào?

Tiêu chí hiện tại trong `configs/data/cfg5_base8.json`:

| Tiêu chí | Căn cứ/cách dùng |
|---|---|
| Task4: di chuyển block;8: đặt block;13/14: đặt/lấy trên cân;17: đặt bút vào ống | Liên quan chuyển vật và thao tác đặt chính xác. Đây là subset khởi đầu, chưa chứng minh là subset tối ưu cho transfer. |
| Robot CFG5, rating≥2 | Lọc theo loại episode và chất lượng metadata. Human chưa dùng trong adapter. |
| Đủ8 RGB camera cố định theo config | Duy trì input của policy. Không lấy episode thiếu camera rồi tự chèn ảnh đen. |
| Pose/width hữu hạn, quaternion hợp lệ, frame count khớp timestamp | Loại schema/số liệu hỏng. |
| Tuổi ảnh/state≤150 ms tại lưới10 Hz | Observation chỉ lấy từ quá khứ; mất đồng bộ tạo gap. |
| Segment ít nhất3 bước | Có observation và future targets; không nối hai bên gap. |

Các tiêu chí tự xem thêm: thao tác đúng task, vật còn nhìn được ở các góc cần thiết, có pha tiếp cận/gắp/di chuyển/đặt, không bị gián đoạn bất thường. Không loại mọi frame tay che vật: che khuất tự nhiên cũng là một phần thao tác. Ghi lý do loại trong ghi chú selection, không sửa metadata gốc.

Hiện57 episode có5.473 bước hợp lệ,684 segment,4.789 sample có future target; **chỉ275 sample đủ20 targets**, median segment6 bước. Một episode có video dài chưa chắc có nhiều đoạn đồng bộ dài. Đừng kết luận dataset “đủ mạnh cho pretraining” chỉ từ số57; trước training cần xem chất lượng và độ dài segment.

### Cách A — giữ nguyên57 episode

Dùng manifest và split hiện có. Không cần copy/xóa dữ liệu hay tạo lại split. Đi thẳng sang mục4 với lệnh mặc định.

### Cách B — chọn thủ công trong57 episode đã đạt

Mở `outputs/cfg5_review/selection.json` bằng Notepad hoặc editor. Trong array `episodes`, giữ các ID muốn dùng và xóa ID không muốn dùng. Giữ `source_manifest_hash`; không tự thêm ID chưa qua manifest. JSON phải đúng dấu phẩy/ngoặc. Ghi lý do trong `note`.

Sau khi lưu, chạy:

```powershell
& '.\.venv-data\Scripts\python.exe' -X utf8 scripts/select_cfg5.py --selection outputs/cfg5_review/selection.json --output data/splits/my_manifest.json
& '.\.venv-data\Scripts\python.exe' -X utf8 scripts/create_splits.py --manifest data/splits/my_manifest.json --output data/splits/my_splits.json
```

`select_cfg5.py` chỉ tạo manifest dẫn xuất mới; raw không thay đổi. Lựa chọn phải còn ít nhất3 group(task,user,calib) để tạo3 split; vài episode cùng một group không đủ. Đây là bước curation trước training; sau khi khóa test không chọn/lọc dựa trên kết quả model ở test.

Sau đó mọi lệnh phải dùng cùng `my_manifest.json`, `my_splits.json`, processed root mới và statistics mới. Không dùng smoke arrays/stats thuộc split cũ cho split mới, dù trùng ID episode.

### Cách C — đổi task/rating hoặc phạm vi lọc

Sao chép `configs/data/cfg5_base8.json` thành config mới rồi sửa `task_ids`/`min_rating` theo quyết định có ghi lại. Giữ representation/camera order nếu không muốn đổi model contract. Chạy lại `prepare_manifest.py --config <config_moi> --output <manifest_moi>`, rồi `create_splits.py --config <config_moi> --manifest <manifest_moi> --output <split_moi>`.

Tiếp tục truyền config mới cho preprocess/stats/check/preview. Model config `data_config` cũng phải trỏ đúng file khi dùng dataset đó. Đổi config làm đổi hash; không chỉnh hash thủ công để bỏ qua kiểm tra. Script không tải thêm dataset.

## 4. Dữ liệu được biến đổi như thế nào?

Ví dụ một mốc t:

1. Đọc `cam_<serial>/timestamps.npy`; chọn frame gần nhất có timestamp≤t và tuổi≤150 ms cho từng camera.
2. Đọc nhánh camera state `037522062165` trong `transformed/tcp_base.npy` và `transformed/gripper.npy`. TCP ở aligned base frame dù dictionary được chia theo camera.
3. Tạo state8D=`xyz(m),qw,qx,qy,qz,measured_width(m)`. Width transformed chuyển mm→m bằng0.001. Quaternion normalize/giữ dấu liên tục trong segment.
4. Tạo action tại từng mốc bằng pose đo được+gripper command width. Khi ACTDataset đọc sample t, lấy targets từ **t+1**, không lấy chính t làm future target.
5. Decode frame RGB cần dùng, resize640×360, lưu uint8. Khi lấy sample mới chia255 và normalize ImageNet đúng một lần.
6. Hết segment thì dừng targets. Ví dụ còn5 targets:5 mask False,15 mask True. L1 loss chia cho5×8 phần tử thật, không chia20×8.

Ở bước lưu, `action.npy` chưa phải array(T,20,8): nó có shape(T,8). Chunk20×8 được tạo khi đọc sample; cách này tránh lưu lặp các target chung.

## 5. Tự chạy chuyển đổi và lưu

Bạn chọn thời điểm chạy. Full57 episode dự kiến30.26 GB ảnh NPY thập phân (chưa cộng metadata); train+val ít hơn. Mất bao lâu phụ thuộc decode video và ổ đĩa, chưa có benchmark full để hứa thời gian.

**Giữ57 episode, chuyển train và val:**

```powershell
& '.\.venv-data\Scripts\python.exe' -X utf8 scripts/preprocess_data.py --split train --output-root data/processed/cfg5
& '.\.venv-data\Scripts\python.exe' -X utf8 scripts/preprocess_data.py --split val --output-root data/processed/cfg5
& '.\.venv-data\Scripts\python.exe' -X utf8 scripts/fit_normalization.py --processed-root data/processed/cfg5 --output data/splits/normalization.json
& '.\.venv-data\Scripts\python.exe' -X utf8 scripts/check_dataset.py --processed-root data/processed/cfg5 --stats data/splits/normalization.json --split train --output docs/audit/train_dataset_full.json
& '.\.venv-data\Scripts\python.exe' -X utf8 scripts/check_dataset.py --processed-root data/processed/cfg5 --stats data/splits/normalization.json --split val --output docs/audit/val_dataset_full.json
```

**Đã tự chọn bằng cách B:**

```powershell
& '.\.venv-data\Scripts\python.exe' -X utf8 scripts/preprocess_data.py --manifest data/splits/my_manifest.json --splits data/splits/my_splits.json --split train --output-root data/processed/my_cfg5
& '.\.venv-data\Scripts\python.exe' -X utf8 scripts/preprocess_data.py --manifest data/splits/my_manifest.json --splits data/splits/my_splits.json --split val --output-root data/processed/my_cfg5
& '.\.venv-data\Scripts\python.exe' -X utf8 scripts/fit_normalization.py --splits data/splits/my_splits.json --processed-root data/processed/my_cfg5 --output data/splits/my_normalization.json
& '.\.venv-data\Scripts\python.exe' -X utf8 scripts/check_dataset.py --splits data/splits/my_splits.json --processed-root data/processed/my_cfg5 --stats data/splits/my_normalization.json --split train --output docs/audit/my_train_check.json
& '.\.venv-data\Scripts\python.exe' -X utf8 scripts/check_dataset.py --splits data/splits/my_splits.json --processed-root data/processed/my_cfg5 --stats data/splits/my_normalization.json --split val --output docs/audit/my_val_check.json
```

Nếu muốn thử trước, thêm `--limit-episodes 1 --max-steps 8` vào preprocess và chọn output root riêng. Statistics/check của smoke cần `--allow-partial-smoke`; không dùng cờ này cho full để che dữ liệu thiếu.

Output mỗi episode có `images.npy`, `state.npy`, `action.npy`, timestamp/index arrays, `segment_ids.npy`, `episode.json`. File được ghi trong `.partial` rồi đổi tên khi xong. **Cập nhật:** CLI có `--resume` để kiểm hash/shape và bỏ qua episode hoàn chỉnh khi chạy lại; `.partial` vẫn dừng để bạn xem `ERROR.json`/log, không xóa raw. Lệnh full mới xem [hướng dẫn VS Code](CFG5_VSCODE_SMOKE_VI.md).

Full statistics phải có `smoke_only:false`, không thiếu train episode, và đúng split hash. Mẫu train dự kiến3.883, val477 nếu giữ nguyên57 và decode tất cả thành công. `check_dataset.py` chỉ kiểm sample đầu/giữa/cuối; counts không có nghĩa đã đánh giá model trên toàn dataset. Để test split khóa lại cho bước đánh giá cuối, chưa dùng nó lựa model.

## 6. Mở NPY để xem và trình bày

NPY là mảng nhị phân, không mở như văn bản/Excel. Cách dễ nhất là xuất trang preview:

```powershell
& '.\.venv-data\Scripts\python.exe' -X utf8 scripts/inspect_processed.py --episode-dir data/processed/smoke_cfg5/task_0004_user_0014_scene_0002_cfg_0005 --index 0 --output outputs/my_sample_preview
Start-Process '.\outputs\my_sample_preview\index.html'
```

Thay `--episode-dir` bằng episode full của bạn và `--index` bằng mốc muốn xem. Output gồm `index.html`, `eight_cameras.png`, `sample.json`; không ghi lại arrays. JSON preview chứa **giá trị vật lý trước normalize** để dễ giải thích.

Hoặc đọc trực tiếp bằng Python trong PowerShell:

```powershell
@'
from pathlib import Path
import numpy as np
p = Path('data/processed/smoke_cfg5/task_0004_user_0014_scene_0002_cfg_0005')
for name in ['images', 'state', 'action', 'segment_ids', 'timestamps_ms']:
    a = np.load(p / (name + '.npy'), mmap_mode='r', allow_pickle=False)
    print(name, a.shape, a.dtype)
state = np.load(p / 'state.npy', mmap_mode='r', allow_pickle=False)
print('3 state dau:', state[:3])
'@ | & '.\.venv-data\Scripts\python.exe' -X utf8 -
```

Để xem sample **đúng dạng model nhận**, dùng adapter với smoke contract hiện tại:

```powershell
@'
from src.data.common import load_config, read_json
from src.data.act_dataset import ACTDataset
cfg = load_config('configs/data/cfg5_base8.json')
ds = ACTDataset('data/processed/smoke_cfg5', read_json('data/splits/cfg5_splits.json'), read_json('data/splits/smoke_normalization.json'), cfg, 'train', allow_partial_smoke=True)
s = ds[0]
for key in ['images', 'state', 'actions', 'is_pad']:
    print(key, s[key].shape, s[key].dtype)
print('target that:', (~s['is_pad']).sum())
print('mask:', s['is_pad'])
'@ | & '.\.venv-data\Scripts\python.exe' -X utf8 -
```

Kết quả expected cho sample này: images(8,3,360,640), state(8,), actions(20,8), mask(20,),5 target thật. Ảnh normalized không nên hiển thị thẳng như RGB; dùng ảnh uint8 từ processed hoặc trang preview.

## 7. Trình bày và bước tiếp theo

Một bài trình bày có thể theo thứ tự: mục tiêu pretraining → task/tiêu chí chọn →57/42 và47/5/5 → sơ đồ timestamp→frame/state →8 ảnh cùng mốc → vector8D → chunk/mask → thống kê train-only → model nhận(B,8,3,360,640) và xuất(B,20,8).

Ghi rõ pose arm là future measured-state proxy; số4.789 samples là dự kiến từ timeline, không phải4.789 lần robot hoàn thành task. Phân biệt thử chạy model với weights đã học. Dẫn nguồn giấy tờ/commit từ references, không gọi mọi quyết định adapter là nguyên bản của paper.

Bạn không cần lọc bằng tay để “tiết kiệm token”: script chạy trên máy, không gửi từng frame vào hội thoại, thường nhanh và ít lỗi hơn thao tác thủ công. Token chủ yếu dành cho viết code/giải thích/đọc kết quả; chi phí giải mã là thời gian CPU/ổ đĩa. Cách phân công hợp lý: script lọc điều kiện khách quan, bạn xem các episode cần quyết định về nội dung, rồi script chuyển đổi hàng loạt. Không cần gửi từng array qua chat.

Khi xong, kiểm các report/`smoke_only` và giữ config+manifest+split+stats cùng nhau. Bước sau mới chuẩn bị train/validate/checkpoint và chạy pretraining theo ngân sách máy. Demonstrations khuôn để sau giai đoạn này theo thứ tự bạn đã chốt.
