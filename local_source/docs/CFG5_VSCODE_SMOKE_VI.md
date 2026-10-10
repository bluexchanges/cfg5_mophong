# Tự chuyển đổi CFG5 bằng VS Code: thử 1–2 episode trước

Hướng dẫn này dùng split full đã khóa `531 train / 66 val / 66 test`. **Cập nhật:** người dùng đã chạy xong full train/val; tài liệu giữ lại để tái lập hoặc thực hành, không cần chạy lại. ACT chưa train. Mọi đường dẫn dưới đây tính từ `D:\Download\mophong_RH20T`. Chỉ dùng robot episode đạt lọc, không chuyển human hoặc robot incomplete.

## 1. Mở đúng project và Python

Mở PowerShell của Windows, nhập:

```powershell
Set-Location 'D:\Download\mophong_RH20T'
code .
```

Máy này đã có lệnh `code` của VS Code. Trong Explorer bên trái, bạn sẽ thấy `scripts/`, `src/`, `configs/`, `data/`. Mở `scripts/preprocess_data.py` để xem **chương trình chạy**; thuật toán đồng bộ timestamp và ghi NPY nằm ở `src/data/cfg5.py`; cách ACT ghép ảnh/state/action chunk nằm ở `src/data/act_dataset.py`. Cấu hình là `configs/data/cfg5_all8.json`, danh sách663 episode đạt là `data/splits/cfg5_all_manifest.json`, phân chia531/66/66 là `data/splits/cfg5_all_splits.json`. Không cần sửa các file này để chạy thử.

Trong VS Code, chọn **Terminal → New Terminal** (mặc định PowerShell). Nếu terminal không đứng ở project root, nhập lại `Set-Location` ở trên. Nếu đã cài Python extension, nhấn **Ctrl+Shift+P → Python: Select Interpreter → Enter interpreter path**, chọn `D:\Download\mophong_RH20T\.venv-data\Scripts\python.exe`. Việc chọn interpreter giúp nút Run và gợi ý code trong editor; các lệnh dưới đây vẫn gọi Python đúng trực tiếp, **không cần Activate.ps1**. `python` trên PATH hiện trỏ tới MSYS2, vì vậy đừng dùng `python scripts/...` cho project này.

Trong terminal VS Code, chạy kiểm tra:

```powershell
Get-Location
$py = '.\.venv-data\Scripts\python.exe'
& $py -X utf8 --version
& $py -X utf8 -c "import cv2, numpy; print('OpenCV', cv2.__version__, 'NumPy', numpy.__version__)"
```

Kết quả `Get-Location` phải là `D:\Download\mophong_RH20T`; Python hiện là3.12.14. `&` trong PowerShell nghĩa là chạy chương trình ở đường dẫn `$py`.

Các biến `$py`, `$common`, `$smoke` chỉ tồn tại trong **terminal hiện tại**. Nếu đóng terminal/VS Code rồi mở lại, nhập lại các dòng khai báo biến trước khi chạy lệnh tiếp. Đừng bấm nút **Run Python File** đơn thuần cho `preprocess_data.py`: nó cần các tham số `--split`, `--manifest`, `--config` từ terminal như bên dưới.

## 2. Chạy smoke 1 episode rồi mở rộng thành 2

Copy **cả khối** này vào terminal. Các dòng `$common` chỉ lập danh sách tham số; dòng cuối mới giải mã ảnh và lưu output. `--limit-episodes 1` lấy episode train đầu tiên trong split; `--max-steps 8` chỉ lưu8 mốc10 Hz, **không** phải toàn episode. `--resume` cho phép chạy lại và kiểm/bỏ qua episode hoàn chỉnh.

```powershell
$common = @(
  '--config', 'configs/data/cfg5_all8.json',
  '--manifest', 'data/splits/cfg5_all_manifest.json',
  '--splits', 'data/splits/cfg5_all_splits.json'
)
$smoke = 'data/processed/cfg5_all_smoke'
& $py -X utf8 scripts/preprocess_data.py @common --split train --limit-episodes 1 --max-steps 8 --output-root $smoke --resume
```

Đợi lệnh xong. Kết quả nên có `steps= 8 smoke= True` và `$LASTEXITCODE` bằng0. Output được lưu tại `data/processed/cfg5_all_smoke/task_0001_user_0007_scene_0001_cfg_0005/`; **không** ghi vào `RH20T_cfg5/`.

Nếu episode thứ nhất trông đúng, nhập lệnh này để có tổng cộng2 episode. Nó sẽ xác minh/bỏ qua episode thứ nhất rồi xử lý episode thứ hai:

```powershell
& $py -X utf8 scripts/preprocess_data.py @common --split train --limit-episodes 2 --max-steps 8 --output-root $smoke --resume
Get-ChildItem $smoke -Directory | Select-Object Name
```

Bạn sẽ thấy hai thư mục episode. Mỗi thư mục có `images.npy` (8 thời điểm ×8 camera ×360×640×3 màu), `state.npy` và `action.npy` (mỗi file8×8), các file timestamp/index/segment và `episode.json`. `episode.json` phải có `steps: 8`, `smoke_only: true`, cùng hash của config/split. `images.npy` là dữ liệu thật đã giải mã; `.json` là ghi chú/kiểm tra, không phải model weights. Hai episode smoke chiếm khoảng89 MB chỉ riêng ảnh, không phải400 GB.

## 3. Mở 8 ảnh và xem “đáp án”

Chọn episode train đầu tiên từ chính split, xuất trang xem vào `outputs/`:

```powershell
$split = Get-Content 'data/splits/cfg5_all_splits.json' -Raw | ConvertFrom-Json
$ep1 = $split.splits.train[0]
& $py -X utf8 scripts/inspect_processed.py --config configs/data/cfg5_all8.json --episode-dir (Join-Path $smoke $ep1) --index 0 --output outputs/cfg5_all_smoke_preview_1
Start-Process '.\outputs\cfg5_all_smoke_preview_1\index.html'
```

Trang web cục bộ hiển thị8 ảnh ở cùng tick policy, bảng shape/dtype, state8D và số future targets còn thật. `sample.json` trong thư mục preview ghi timestamp, tuổi từng ảnh/state, target8D và mask20 bước. Ảnh là **đầu vào**; target8D ở thời điểm tương lai là “đáp án” cho ACT. Nếu muốn xem episode thứ hai, đổi `$ep1` thành `$split.splits.train[1]` và output thành `outputs/cfg5_all_smoke_preview_2`.

Smoke thành công khi: hai lệnh preprocess thoát mã0; có đúng1 rồi2 thư mục; `steps=8`, `smoke_only=true`; trang preview mở được, đủ8 góc, shape đúng, `valid_future_actions > 0` ở index0. Nếu ra `FileExistsError`, đọc tên file/thư mục trong thông báo: `--resume` chỉ bỏ qua output hoàn chỉnh; thư mục `.partial` cần xem `ERROR.json`, không chạy đè hay xóa raw. Nếu lỗi `config_hash`/`split_hash`, có thể bạn đã trộn file subset cũ với `cfg5_all`.

## 4. Khi smoke đạt: chuyển train/val full

**Không chạy ngay nếu chưa xem smoke và kiểm dung lượng.** Bản full phải dùng thư mục khác và **không** có `--max-steps`/`--limit-episodes`:

```powershell
[math]::Round((Get-PSDrive D).Free / 1GB, 1)  # GiB còn trống trên ổ D
$full = 'data/processed/cfg5_all'
& $py -X utf8 scripts/preprocess_data.py @common --split train --output-root $full --resume
& $py -X utf8 scripts/preprocess_data.py @common --split val --output-root $full --resume
```

Chạy **từng lệnh**, đợi xong và kiểm `$LASTEXITCODE` bằng0 trước lệnh sau. Output là `data/processed/cfg5_all/<episode>/`, flat theo tên episode; JSON split cho biết mỗi episode thuộc train hay val. `--resume` kiểm hash/shape của episode đã hoàn chỉnh và bỏ qua; nó **không sửa** một episode `.partial` bị lỗi. Không chuyển test lúc này. Sau khi đủ531 train và66 val, mới tính train-only normalization/kiểm dataset bằng lệnh trong [hướng dẫn split full](CFG5_FULL_SPLIT_AND_RUN_VI.md); smoke statistics không dùng để train chính thức.

Ảnh NPY không nén dự kiến **361,79 GB train +38,53 GB val**, cộng file nhỏ khác. Lần kiểm gần đây D: còn khoảng576 GiB, nhưng phải đo lại lúc chạy. Không có benchmark toàn episode trên máy này, nên không hứa số giờ: decode8 video và ghi khoảng400 GB có thể kéo dài nhiều giờ; smoke8 bước đo chủ yếu overhead nên không suy thời gian full từ smoke. Muốn có ước lượng thực tế mà không tạo bản sao thừa, sau smoke có thể chạy **10 episode full đầu tiên** vào chính `$full` với `--limit-episodes 10 --resume`, bấm giờ rồi tiếp tục lệnh full train ở trên; các episode đã hoàn chỉnh được kiểm và bỏ qua. Lệnh tùy chọn:

```powershell
$started = Get-Date
& $py -X utf8 scripts/preprocess_data.py @common --split train --limit-episodes 10 --output-root $full --resume
(Get-Date) - $started
```

Mười episode đầu có thể không đại diện cho tất cả663 episode, nhưng cho bạn số đo thật về CPU/ổ đĩa trên máy mình. Không chạy training ACT trong các lệnh trên.
