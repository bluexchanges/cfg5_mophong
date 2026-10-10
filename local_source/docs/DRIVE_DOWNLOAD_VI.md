# Tải dữ liệu Google Drive về server bằng 4 luồng

## 1. Mục tiêu và giới hạn

Script `scripts/download_drive.py` dùng Python standard library và rclone đã có
trên server; không cần PyTorch, GPU, môi trường ACT hoặc cài thư viện Python mới.

- Mặc định remote `mydrive:`; folder ID `1vzCuuAVDapO1bBW1ijk4YYlFp4uKoItI`.
- Đích `/home/user08/cfg5-pretrain/downloads/finetune_data` khi chạy bằng user08.
- `--workers 4` tương ứng `rclone --transfers 4`: tối đa 4 file tải cùng lúc.
  Đây là concurrency mạng/I/O, không phải đặt CPU affinity hoặc xin 4 core Slurm.
- Tắt multi-thread streams trong từng file để không nhân thêm luồng dữ liệu lớn.
- File tải xong được lưu ngay dưới đúng tên/thư mục; không chờ đủ dataset.
- Khi chạy lại, đối chiếu kích thước/checksum; giữ file đã khớp, tải file thiếu
  hoặc khác nội dung. File khác nội dung tại đích có thể được thay bằng bản Drive.
- Không xóa file thừa tại đích; không sửa file trên Drive.
- Copy xong còn chạy kiểm tra nguồn → đích; chỉ báo `complete` khi cả hai bước đạt.

**Resume ở mức file hoàn chỉnh, không bảo đảm resume từng byte của file dở.**
Ví dụ có 100 file, đã xong 60, đang tải 4 file thì bị ngắt: 60 file được giữ;
4 file dở có thể tải lại từ đầu cùng 36 file chưa xong. rclone dùng tên `.partial`
cho file tạm rồi đổi tên khi hoàn tất; không coi `.partial` là dữ liệu hoàn chỉnh.
Đừng dùng `--ignore-existing` để bỏ qua file dở hoặc sửa bằng `--inplace`.

Mất mạng laptop/đóng VS Code/SSH không làm mất mạng server. Worker chạy nền độc lập
bằng `start_new_session`, stdin không phụ thuộc terminal. Nếu mạng server bị gián
đoạn, rclone có timeout/retry và wrapper thử lại sau 60, 120, 240… tối đa 600 giây.
Mặc định retry không giới hạn với lỗi không được phân loại fatal. Lỗi cấu hình,
không tìm thấy thư mục hoặc fatal dừng lại để người dùng sửa; mất quyền OAuth không
thể tự chữa bằng retry. Nếu server reboot hoặc tiến trình bị kill, chạy `start` lại.
Không hứa tốc độ tăng 4 lần hoặc SSH bị ngắt là tiến trình server bị ngắt.

## 2. Chuẩn bị và upload gói

Trên server kiểm tra:

```bash
rclone version
rclone listremotes
```

Server hiện có rclone v1.75.1 và `mydrive:`. Tài khoản rclone phải có quyền đọc nội
dung thư mục. Đã liệt kê 307 thư mục `pick_place_000042`…`pick_place_000348` và
`review_8cams_summary.json`; đây chưa phải kiểm chứng đủ/chất lượng 307 episode.

Trên Windows PowerShell upload:

```powershell
scp "F:\cfg5_mophong\drive_download_bundle.zip" user08@172.16.3.8:/home/user08/cfg5-pretrain/
```

Trên terminal server giải nén code/hướng dẫn:

```bash
unzip -n "$HOME/cfg5-pretrain/drive_download_bundle.zip" \
  -d "$HOME/cfg5-pretrain/drive_download_bundle"
```

Không cần đặt script trong repo trainer, không thay code training/preprocessing.
Nếu dùng bản nguồn của dự án, entrypoint là `python3 scripts/download_drive.py`.

## 3. Khởi động 4 luồng

Nếu lệnh rclone `nohup` cũ còn chạy cùng đích, script sẽ từ chối tạo tiến trình
trùng và in PID. Có thể giữ nguyên lượt tải cũ đến khi xong; hoặc xác nhận PID
đúng là rclone tải `finetune_data` rồi dừng riêng tiến trình đó:

```bash
ps -p PID_RCLONE_CU -o pid,args
kill -TERM PID_RCLONE_CU
```

Thay `PID_RCLONE_CU` bằng số PID script báo; chờ nó kết thúc rồi mới `start`.
Không dùng `pkill python`, `killall rclone` hoặc hủy job training khác.

Khởi động (đã tự chạy nền, không cần thêm nohup hoặc dấu `&`):

```bash
python3 "$HOME/cfg5-pretrain/drive_download_bundle/download_drive.py" start
```

Kết quả in PID worker, đích và đường dẫn log. Một lock theo thư mục đích bảo vệ
không chạy hai lượt script đồng thời, kể cả các lệnh start gần nhau. Thư mục nguồn
được ghi lại để không trộn nhầm một Drive folder khác vào đích cũ.

## 4. Xem tiến độ, tiếp tục và dừng

Xem trạng thái và 22 dòng log cuối:

```bash
python3 "$HOME/cfg5-pretrain/drive_download_bundle/download_drive.py" status
```

Theo dõi liên tục, dùng đường dẫn log được in bởi `start/status`:

```bash
tail -f DUONG_DAN_LOG
```

`Ctrl+C` ở lệnh `tail` chỉ dừng xem log. Các trạng thái:

| status | Ý nghĩa |
|---|---|
| `starting` | Đang bắt đầu một vòng copy |
| `copying` | Đang tải, rclone ghi thống kê mỗi 30 giây |
| `retry_wait` | Có lỗi, đang chờ thử lại; xem exit_code và retry_in_seconds |
| `verifying` | Đang kiểm tra kích thước/hash sau copy |
| `complete` | Copy/check cùng exit=0, có verified_at và DRIVE_DOWNLOAD_OK |
| `failed` | Chưa đạt; xem log, sửa quyền/path/dung lượng/cấu hình rồi start lại |
| `stopped` | Đã dừng có chủ đích, các file đã hoàn tất vẫn ở đích |

Ngày giờ `state.json` dùng UTC; terminal rclone có thể dùng giờ server.
Nếu status còn `copying` nhưng PID không còn sống trên headnode này thì là trạng
thái cũ sau sự cố, không phải tiến trình đang tải. Worker chạy trên headnode nào
thì xem/dừng tại headnode đó; không gửi signal bằng PID đọc từ máy khác.

Tiếp tục sau sự cố/reboot: **chạy lại đúng lệnh `start` và cùng đích**. Không tạo
thư mục timestamp mới nếu muốn tái sử dụng dữ liệu đã tải.

Dừng riêng lượt do script quản lý:

```bash
python3 "$HOME/cfg5-pretrain/drive_download_bundle/download_drive.py" stop
```

Lệnh stop kiểm hostname, user và token worker trước khi gửi SIGTERM. Không dừng
ACT/Slurm hoặc tiến trình rclone cũ chạy ngoài script. Worker chờ rclone dừng và
ghi `stopped`; không xóa file hoàn chỉnh. Chờ status=`stopped` trước khi start lại.

## 5. Xác nhận tải xong

Chỉ coi xong khi status=`complete` và log có `DRIVE_DOWNLOAD_OK`.
`copy` kết thúc hoặc file/thư mục tồn tại chưa đủ chứng minh toàn bộ dữ liệu đạt.
Các file trạng thái đặt ngoài raw, dưới:

```text
$HOME/cfg5-pretrain/logs/drive_download/<mã theo đường dẫn đích>/
├── config.json          # Nguồn, đích, số luồng; không chứa OAuth token
├── download.lock        # Lock, không xóa khi đang chạy
├── download.log         # Copy/check/retry
├── state.json           # PID/hostname/status/attempt/verified_at
└── check_files.txt      # Kết quả đối chiếu từng file
```

Trong `check_files.txt`, `=` là khớp; `+` là thiếu ở đích; `*` là khác;
`!` là lỗi đọc/hash. Kiểm tra dùng `rclone check --one-way`, không tự tải toàn bộ
Drive lần nữa để so sánh. Có hash nguồn thì dùng hash; thiếu hash chung thì có
thể chỉ kiểm size. Dữ liệu thông thường trên Google Drive hỗ trợ MD5.

Có thể kiểm lại, chạy foreground và đọc đĩa/checksum (không khởi động cùng lượt tải):

```bash
python3 "$HOME/cfg5-pretrain/drive_download_bundle/download_drive.py" verify
```

`verify` không tự sửa file. Nếu thiếu/khác, dùng `start` để copy sửa rồi check lại.
Giữ thư mục Drive ổn định trong lúc copy/check; kết quả không phải snapshot bất
biến nếu người khác đang thay đổi dataset trên Drive. Sau tải vẫn cần kiểm schema,
camera/timestamp/action rồi chia episode trước khi preprocessing/fine-tune.

## 6. Dùng cho thư mục Drive khác

Các tùy chọn phải được dùng nhất quán với cùng `--dest` cho start/status/stop/verify:

```bash
python3 "$HOME/cfg5-pretrain/drive_download_bundle/download_drive.py" start \
  --remote mydrive \
  --folder 'https://drive.google.com/drive/folders/FOLDER_ID' \
  --dest "$HOME/cfg5-pretrain/downloads/another_dataset" \
  --workers 4
```

`--retry-seconds 60`: thời gian chờ ban đầu; `--max-attempts 0`: không giới hạn.
Ví dụ `--max-attempts 8` dừng sau 8 vòng chưa đạt để kiểm tra lỗi thủ công.
Không dán OAuth token vào hướng dẫn/README hoặc gửi lên GitHub.

## 7. OAuth và server dùng chung

Thông báo shared Google Drive client_id sẽ bị bỏ trong 2026 chưa đồng nghĩa tải
thất bại. Nếu vẫn copy được thì lượt tải có thể tiếp tục. Khi 401/403/invalid_client
hoặc mất quyền xuất hiện, kiểm log và sửa remote; cần client_id riêng để dùng lâu
dài. Cấu hình quyền đọc là đủ cho tải. Không tự đổi/xóa remote chung đang dùng.

Script chạy trên Linux có Python 3 và rclone; không dùng GPU và không tạo Slurm job.
Đây là tác vụ truyền dữ liệu. Nếu quy định cụm yêu cầu tải qua data-transfer node
hoặc Slurm, chạy trên node được cho phép; không dùng số core của ACT để suy ra tốc
độ tải. RAM/CPU, băng thông Drive, mạng và NFS đều có thể giới hạn tốc độ.

## 8. Kiểm chứng của gói

10 fixtures đã đạt trên Linux (WSL Ubuntu): retry sau lỗi tạm, không tuyên bố
complete trước check, check lỗi → copy sửa → check lại, fatal/auth error,
giới hạn số lần thử, verify không copy, folder binding và bảo vệ PID. Ba kiểm tra
lifecycle dùng provider giả 24 file nhỏ: lock chặn lượt trùng, worker có session
riêng, copy/check hoàn tất, dừng rồi start lại giữ byte/mtime của file đã tải,
từ chối rclone cũ cùng đích và không tự hủy tiến trình đó.
Không chạy tải thật Google Drive trong lúc viết gói; không benchmark server.
Các kiểm tra này xác minh wrapper, không chứng minh OAuth/NFS/throughput của cụm.

Nguồn chính thức:

- https://rclone.org/commands/rclone_copy/
- https://rclone.org/commands/rclone_check/
- https://rclone.org/docs/#inplace
- https://rclone.org/docs/#multi-thread-streams-int
- https://rclone.org/drive/#making-your-own-client-id
