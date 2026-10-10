# Split ngẫu nhiên toàn CFG5 và tự chuyển đổi dữ liệu

**Tình trạng bản GitHub:** Processed output đã được dọn khỏi máy phát triển và không nằm trong repository. Khi chạy trên server, dùng [quy trình server](SERVER_REPROCESS_VI.md) để xử lý khác biệt mtime/hash nguồn trước khi dùng các lệnh dưới đây.

**Nếu bạn mới dùng VS Code/PowerShell, bắt đầu ở [hướng dẫn smoke từng bước](CFG5_VSCODE_SMOKE_VI.md).** Lệnh bên dưới là bản rút gọn cho người đã quen terminal.

**Cập nhật sau khi người dùng chạy:** Đã xử lý đủ531 train và66 validation episode vào `data/processed/cfg5_all/` (layout flat, membership theo split JSON); thống kê train-only và kiểm sample đã đạt. Test chưa xử lý, ACT chưa train. Xem [audit sau xử lý](audit/cfg5_all_processed_postrun.json). Các lệnh dưới đây giữ lại để tái lập, dùng full manifest 663 robot episode, không phải subset57 cũ.

## Split đã khóa

`data/splits/cfg5_all_splits.json` dùng seed42 và tỷ lệ mục tiêu80/10/10. Với663 episode, số nguyên gần nhất theo largest remainder là **531 train / 66 validation / 66 test** (80,09% /9,95% /9,95%). Mỗi group `(task,user,calib)` nằm trọn trong một split; 114 group được chia94/10/10. Thuật toán xáo thứ tự group bằng seed rồi chọn ngẫu nhiên trong các phương án vẫn đạt **đúng số episode**, không ghim task nào vào train. Đổi seed tạo phân bổ khác; camera của một episode không bao giờ tách split.

Năm task liên quan pick-and-place có57 episode: **42 train /12 validation /3 test** trong lần rút seed42. Riêng task14 không có nhóm ở train vì chỉ có hai group và phép rút ngẫu nhiên đưa chúng ra ngoài train. Điều này phải được ghi trong báo cáo pretraining; sampler không thể tăng trọng số cho một task không có trong train. Không xem validation/test để chọn lại seed cho “đẹp”, vì như vậy không còn một split ngẫu nhiên đã khóa. Đánh giá đặt vật vào khuôn sẽ cần dữ liệu task đích và rollout riêng sau này.

Human episode không là sample ACT 8D. Trong `RH20T_cfg5/human_robot_index.csv`, cột `robot_act_split` kế thừa split của robot cùng mã tên khi robot đó thuộc663 episode đạt bộ lọc:475 liên hệ train,66 val,65 test;588 human không có robot trong split ACT. Đây là nhãn **phân vùng**, không phải action label cho video human.

Để kiểm split mà không xử lý ảnh:

```powershell
Set-Location 'D:\Download\mophong_RH20T'
Get-Content 'data\splits\cfg5_all_splits.json' -TotalCount 40
& '.\.venv-data\Scripts\python.exe' -X utf8 scripts\create_splits.py --config configs\data\cfg5_all8.json --manifest data\splits\cfg5_all_manifest.json --output data\splits\cfg5_all_splits_recheck.json
```

File `*_recheck.json` phải có cùng nội dung/hash với split đã khóa. Không đổi seed/config khi chuyển đổi dữ liệu.

## Chuyển đổi gì, khi nào?

Raw episode chứa MP4, timestamps và NPY object; ACTDataset hiện cần mỗi episode đã xử lý với ảnh RGB uint8, state8D, action8D, segment và frame/timestamp index. `scripts/preprocess_data.py` decode **mỗi episode một lần**, ghi các mảng vào `data/processed/cfg5_all/<episode>/`. Chunk20 action được tạo khi đọc sample, không lưu lặp từng chunk. Dữ liệu gốc không bị sửa.

**Chuyển train và validation trước.** Train để fit model/thống kê; validation để chọn checkpoint nhưng không fit normalization. Giữ test khóa, chỉ chuyển và dùng sau khi đã khóa checkpoint/cấu hình. Chuyển đổi test về mặt kỹ thuật không học từ test, nhưng trì hoãn giúp tránh vô tình chọn tham số dựa vào test. Không đưa human hay658 robot chưa đạt vào lệnh này.

Lưu ý dung lượng: ảnh uint8 không nén ở 640×360×8 camera ước tính **361,79 GB train +38,53 GB validation =400,32 GB**; test thêm42,14 GB khi cần. Đây là dung lượng dẫn xuất mới, không phải copy MP4. Trước khi chạy, kiểm dung lượng D: với `Get-PSDrive D | Select-Object Name,Free`. Lần kiểm gần đây còn khoảng576 GiB, nhưng phải kiểm lại vào lúc chạy. CLI đã có `--resume`: kiểm hash/shape và bỏ qua episode hoàn chỉnh; nếu còn `.partial`, xem `ERROR.json` trước khi xử lý, không tự bỏ qua episode dở.

Chạy thử **1 train episode,8 bước** vào root riêng để kiểm thao tác, không dùng nó làm dữ liệu chính:

```powershell
$py = '.\.venv-data\Scripts\python.exe'
& $py -X utf8 scripts\preprocess_data.py --config configs\data\cfg5_all8.json --manifest data\splits\cfg5_all_manifest.json --splits data\splits\cfg5_all_splits.json --split train --limit-episodes 1 --max-steps 8 --output-root data\processed\cfg5_all_smoke --resume
```

Khi đủ dung lượng, chạy full vào **root khác**. Mỗi lệnh xử lý tuần tự các episode của một split; có thể mất lâu do decode8 video/episode. Không dùng `--limit-episodes` hoặc `--max-steps` cho bản full:

```powershell
& $py -X utf8 scripts\preprocess_data.py --config configs\data\cfg5_all8.json --manifest data\splits\cfg5_all_manifest.json --splits data\splits\cfg5_all_splits.json --split train --output-root data\processed\cfg5_all --resume
& $py -X utf8 scripts\preprocess_data.py --config configs\data\cfg5_all8.json --manifest data\splits\cfg5_all_manifest.json --splits data\splits\cfg5_all_splits.json --split val --output-root data\processed\cfg5_all --resume
& $py -X utf8 scripts\fit_normalization.py --config configs\data\cfg5_all8.json --splits data\splits\cfg5_all_splits.json --processed-root data\processed\cfg5_all --output data\splits\cfg5_all_normalization.json
& $py -X utf8 scripts\check_dataset.py --config configs\data\cfg5_all8.json --splits data\splits\cfg5_all_splits.json --stats data\splits\cfg5_all_normalization.json --processed-root data\processed\cfg5_all --split train --output docs\audit\cfg5_all_train_check.json
& $py -X utf8 scripts\check_dataset.py --config configs\data\cfg5_all8.json --splits data\splits\cfg5_all_splits.json --stats data\splits\cfg5_all_normalization.json --processed-root data\processed\cfg5_all --split val --output docs\audit\cfg5_all_val_check.json
```

Đợi từng lệnh hoàn thành và thoát mã0 rồi mới chạy lệnh tiếp theo. `fit_normalization` từ chối thiếu train episode/smoke data nếu không bật cờ đặc biệt; **không bật** `--allow-partial-smoke` cho bản chính thức. `check_dataset` chỉ kiểm vài sample đầu/giữa/cuối, chưa phải đánh giá toàn bộ model. Xem [hướng dẫn mở NPY](CFG5_HANDS_ON_VI.md#6-mở-npy-để-xem-và-trình-bày) để xem ảnh8 camera và action mask.

## 10 Hz và đồng bộ nghĩa là gì?

**10 Hz có căn cứ từ RH20T paper**, không phải chọn tùy tiện: bảng III ghi RGB, joint angle/torque và gripper width danh nghĩa10 Hz; Cartesian pose và force/torque100 Hz. Mục IV-A ghi ACT chunk20 ứng với2 giây tại10 Hz và resize ảnh640×360. Tuy nhiên MP4/timestamp trên máy không đều đúng100 ms (mẫu đã xem có median khoảng68 ms), nên **lưới đều 100 ms là quyết định adapter của dự án** để theo cấu hình thí nghiệm ACT; không dùng FPS container làm đồng hồ. `max_observation_age_ms=150` và `max_state_age_ms=150` là ngưỡng chúng ta chọn, không phải con số paper đảm bảo.

Tại tick `t`, code chọn cho từng camera frame mới nhất có timestamp `≤t`, và chọn TCP/gripper state mới nhất `≤t`; tất cả phải không quá cũ150 ms. Nếu một view hay state quá cũ, tick bị loại và gap chia segment. Vì thế 8 camera và robot **được ghép theo timestamp**, nhưng không chụp cùng một microsecond; hai nguồn có thể lệch tới150 ms theo tiêu chí hiện tại. Đây là đồng bộ gần đúng có giới hạn, không phải chứng minh đồng bộ hardware trigger hay bù latency.

Quan trọng: ACT adapter dùng **measured TCP/gripper** làm state/target; nó **không đọc lệnh motor của 7 khớp hoặc lực/torque** để làm nhãn điều khiển. Vì vậy có thể nói ảnh được căn thời gian với **trạng thái robot đã ghi**, không thể nói đã kiểm chứng đồng bộ với **lệnh/động cơ tay máy**. Trước tác vụ đặt chốt vào lỗ nhỏ, cần đo thêm phân bố lệch timestamp và kiểm tra chính sách/controller trong MuJoCo; offline alignment không tự bảo đảm robot đặt chính xác.

Nguồn: [RH20T paper](../references/papers/RH20T_paper_compressed%20(1).pdf), bảng III/mục IV-A; implementation ở `src/data/cfg5.py::make_timeline`, config ở `configs/data/cfg5_all8.json`.
