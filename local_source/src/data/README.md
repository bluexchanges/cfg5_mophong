# Logic đọc và chuẩn bị CFG5

Mục tiêu của package là biến từng episode raw thành sample NumPy cho ACT mà không load cả dataset vào RAM. Đường dẫn trong bảng tính từ thư mục này.

| File | Logic chính | Vì sao quan trọng? |
|---|---|---|
| `__init__.py` | Đánh dấu package `src.data`. | Không chạy pipeline khi import package. |
| `common.py` | `ROOT`, đọc/ghi JSON, hash nội dung, load config, tìm raw, kiểm output ngoài raw, đối chiếu contract. | Ngăn ghi output vào dataset gốc và trộn artifact có config khác. |
| `geometry.py` | Normalize quaternion wxyz, giữ dấu liên tục, đổi pose/rotation-vector sang ma trận4×4, kiểm transform. | q và−q cùng rotation nhưng đổi dấu đột ngột gây target học không liên tục; thứ tự quaternion phải thống nhất. |
| `cfg5.py` | Đọc NPY dictionary đáng tin cậy; kiểm timestamps/video; chọn quá khứ bằng `past_indices`; tạo timeline, manifest, fingerprint và processed episode. | Nơi quyết định episode/bước nào hợp lệ, camera nào được dùng và có cắt gap hay không. |
| `splits.py` | Nhóm episode theo(task,user,calib), xáo theo seed, phân train/val/test; kiểm group/episode không trùng và split hash. | Chia trước action chunks để tránh rò rỉ giữa train và đánh giá. |
| `act_dataset.py` | `fit_normalization` tính moments trên train; `ACTDataset` lập index theo segment, đọc memmap, normalize ảnh/state/action, tạo padding mask. | Chỉ mục t không được lấy target xuyên gap hoặc sang episode khác. |

## Từ raw đến sample

1. `build_manifest` chỉ chọn robot episode đúng mẫu tên, task và rating; yêu cầu đủ8 video. Nó kiểm frame đầu và số frame, chưa decode toàn bộ video.
2. `make_timeline` tạo lưới10 Hz trên phần thời gian giao. Ở mỗi mốc, lấy ảnh/state gần nhất không nằm trong tương lai, tuổi≤150 ms. Khoảng thiếu bị cắt; giữ segment đủ dài.
3. State lưu pose đo được+width đo được. Action lưu pose đo được+width command ở từng mốc; ý nghĩa “tương lai” xuất hiện khi dataset lấy từ t+1. Không gọi arm target này là expert arm command.
4. `preprocess_episode` kiểm fingerprint nguồn, ghi arrays vào `.partial`, decode từng camera, đổi BGR→RGB, resize, rồi rename khi hoàn tất. Nếu lỗi giữ `ERROR.json`; nếu output đã tồn tại thì từ chối ghi đè.
5. Fit mean/std từ train. State dùng các bước train; action statistics dùng các bước đủ điều kiện làm future target. Không dùng val/test để fit.
6. Với observation t, dataset lấy tối đa20 targets t+1... trong cùng segment. Phần thiếu có giá trị0 sau chuẩn hóa và `is_pad=True`; loss tương lai phải bỏ qua phần này và dùng đúng denominator.

Sample trả về: `images` float32(8,3,360,640), `state` float32(8,), `actions` float32(20,8), `is_pad` bool(20,), kèm `episode` và `timestamp_ms`. Một observation cuối segment không có future target thì không được đưa vào index.

NPY object raw dùng pickle chỉ vì đây là dataset cục bộ đã được cung cấp; processed arrays dùng `allow_pickle=False`. Cache memmap giữ một episode gần nhất. Smoke data/statistics chỉ được đọc khi bật `allow_partial_smoke`, không được dùng như tập train đầy đủ.

Xem [schema file processed](../../data/README.md), [tham số config](../../configs/data/README.md), [lệnh chạy](../../scripts/README.md).

Bổ sung `selection.py`: `select_manifest` nhận allowlist episode từ manifest đã kiểm, đối chiếu hash/ID, lưu nguồn lựa chọn và ID bị bỏ; không mutate manifest đầu vào. CLI select_cfg5.py tạo manifest mới, sau đó phải tạo split/processed/stats mới.
