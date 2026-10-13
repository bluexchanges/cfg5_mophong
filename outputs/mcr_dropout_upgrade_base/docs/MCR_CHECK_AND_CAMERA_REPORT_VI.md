# Kiểm tra MCR đóng băng và thống kê camera trước preprocessing

Phạm vi: toàn bộ 1.317 robot episode đã có joint; giữ nguyên raw, split 1.053 train /
132 val / 132 test, nhãn 7 góc khớp + độ mở kẹp và normalization đã tạo. Không dùng
checkpoint ACT E5/E58 dạng TCP để khởi tạo ACT joint mới. Trọng số MCR ResNet50
được tải/kiểm tra bởi script `download_mcr.py` đã có trong bundle trước.

## 1. Hai việc trong job kiểm tra

`scripts/slurm/cfg5_mcr_preflight.sbatch check` thực hiện:

1. Tạo một báo cáo HTML từ manifest đã khóa; kiểm SHA của `camera_valid.npy`
   và đối chiếu shape, coverage, metadata cho cả 1.317 episode. Đọc trạng thái
   tồn tại/kích thước MP4 và timestamp; không giải mã ảnh test hoặc tạo cache test.
2. Kiểm kích thước 94.345.861 byte và SHA256 chính thức của MCR. Nếu chưa có
   checkpoint, gọi downloader chính thức đã pin; nếu có checksum sai thì dừng.
3. Load strict ResNet50 vào policy dùng cùng code với preprocessing. Khi gọi
   `policy.train()`, backbone vẫn eval, không có tham số trainable và không đổi
   buffer BatchNorm; ACT vẫn có tham số trainable.
4. Chạy hai forward nhỏ trên ảnh tổng hợp: đầu ra (1,8,2048,7,7), finite, camera
   mask bằng false có feature zero, kết quả lặp tương đương trong tolerance 1e-6.
   Đây không phải đánh giá chất lượng gắp hay benchmark tốc độ train.
5. Đo toàn bộ home, dung lượng filesystem còn trống và ngân sách feature đầy đủ.
   Ngân sách script là 1,8 TB thập phân cộng dự phòng 10 GiB; không phải truy vấn
   quota thật. Dữ liệu của người dùng chung trong home vẫn được tính.

Job `check` không tạo cache và không train. ResNet pass không đồng nghĩa đủ đĩa;
đọc trường `full_build_fits` và `additional_space_needed_GiB` trong log/JSON.

## 2. Báo cáo task/camera trong một file

File server: `outputs/data_audit/cfg5_robot_all_camera_report.html`.

- Một bảng 104 task theo snapshot local; số task chính xác trên server được in
  sau khi kiểm manifest server.
- Mỗi serial camera có số episode thiếu cả đoạn (T), số thiếu một phần (P),
  và tỷ lệ timestamp hợp lệ tính theo tổng độ dài các episode của task.
- Bấm tên task để mở danh sách từng episode, split, serial thiếu, tỷ lệ/tick và
  nguyên nhân: file mất/rỗng, video/timestamp lỗi, hoặc không phủ timeline.
- Có ô lọc theo tên task và bảng phân bố thiếu 0 đến 8 camera.
- 4 episode không có joint nằm ngoài bảng được ghi riêng, raw không bị thay đổi.

T=0 timestamp hợp lệ trên timeline giữ lại; không nhất thiết là thư mục camera
vật lý bị mất. P=0<valid_ticks<steps: chỉ phần thiếu được mask. Tỷ lệ dựa trên
mask/timeline 10Hz và giới hạn tuổi quan sát đã có, không dựa vào đoán nhãn.
Không có bộ lọc rating hoặc loại episode thiếu camera được thêm vào.

File local `F:\cfg5_mophong\cfg5_robot_all_camera_report_local.html` được tạo từ
audit local ngày trước, có nhãn SNAPSHOT LOCAL. Nó không chứng minh split hay
raw trên server hiện tại đã giống local; dùng báo cáo job server để xác nhận.

## 3. Upload và check trên server

PowerShell local:

```powershell
scp "F:\cfg5_mophong\cfg5_mcr_preflight_bundle.zip" user08@172.16.3.8:~/cfg5-pretrain/
```

Terminal server:

```bash
cd "$HOME/cfg5-pretrain/edabk-physical-ai-robot/04-Src/software/cfg5_act"
unzip -o "$HOME/cfg5-pretrain/cfg5_mcr_preflight_bundle.zip" -d "$PWD"
J=$(sbatch --parsable --time=00:30:00 scripts/slurm/cfg5_mcr_preflight.sbatch check)
echo "Job check ResNet + báo cáo: $J"
tail -F "$HOME/cfg5-pretrain/logs/cfg5-mcr-preflight-$J.out"
```

Các marker: `CAMERA_REPORT_OK`, `MCR_RESNET_CHECK_OK`, `MCR_STORAGE_CHECK`,
`MCR_CHECK_AND_CAMERA_REPORT_JOB_OK`. Trạng thái cuối cùng kiểm bằng:

```bash
sacct -j "$J" -X --format=JobID,State,ExitCode,Elapsed
tail -n 30 "$HOME/cfg5-pretrain/logs/cfg5-mcr-preflight-$J.err"
```

JSON check riêng của job: `outputs/mcr_checks/mcr_frozen_check_<jobid>.json`.
Nếu ResNet lỗi, báo cáo camera vẫn có vì wrapper tạo báo cáo trước kiểm model.

Tải HTML server về local, chạy PowerShell:

```powershell
scp user08@172.16.3.8:~/cfg5-pretrain/edabk-physical-ai-robot/04-Src/software/cfg5_act/outputs/data_audit/cfg5_robot_all_camera_report.html "F:\cfg5_mophong\cfg5_robot_all_camera_report_server.html"
```

## 4. Bắt đầu preprocessing feature-only

```bash
P=$(sbatch --parsable scripts/slurm/cfg5_mcr_preflight.sbatch build)
echo "Job preprocessing: $P"
tail -F "$HOME/cfg5-pretrain/logs/cfg5-mcr-preflight-$P.out"
```

Mode build kiểm tra lại ResNet/báo cáo/ngân sách trước, chạy pilot 3 episode rồi
build đầy đủ train+val khi các bước trước thành công. Nếu full-build không đủ
đĩa, job dừng trước khi tạo cache; không nâng ngân sách hoặc bỏ qua `phuc`.
Mode pilot cũng đòi đủ chỗ cho build đầy đủ để tránh chỉ chạy thử rồi mắc ở build.

Cache mới: `data/features/cfg5_robot_all_mcr_features`, feature INT8 + scale FP16,
không lưu thêm JPEG, không pooling xuống 4x4, không đụng root cache cũ đã dọn.
JPEG Q90 chỉ tạo/giải mã tạm trong RAM khi trích feature. Test giữ riêng.
Success: `MCR_CACHE_READY`, `MCR_FROZEN_PREPROCESS_JOB_OK` và `READY_FROZEN.json`.
Không tự đăng ký training ACT trong job này.

## 5. Kiểm chứng local

- Trọng số MCR thật SHA đúng, load strict và hai lượt extract CPU pass; backbone
  0 tham số trainable, ACT 73.493.449 tham số trainable, BatchNorm không đổi.
- 5 test nhỏ: thống kê thiếu cả đoạn/một phần, HTML escape, dữ liệu coverage sai,
  checksum mask đổi, checkpoint sai, ngân sách tính cả home và reserve.
- Báo cáo local: 1.317 episode, 104 task, 574.493 tick; 523 episode có ít nhất
  một camera không hợp lệ cả đoạn, 1.303 có camera thiếu một phần. Hai nhóm chồng.
- Chưa kiểm CUDA của server và chưa tạo cache đầy đủ từ phiên local này.
