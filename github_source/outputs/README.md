# Kết quả chạy mô phỏng

Đây là artifact sinh khi chạy script, không phải source code hay tập demonstrations đã chuẩn bị cho ACT. Các run mô phỏng hiện tại là scripted controller/kiểm render. Đã thêm trang rà soát CFG5 và processed preview; ACT integration report ở docs/audit. Chưa có checkpoint ACT đã học ở đây.

## Mỗi thư mục hiện chứa gì?

| Thư mục | Nội dung | Cách hiểu |
|---|---|---|
| `placement_logic_review/` | `episode_42.json`, `episode_43.json`, `episode_44.json`, `episodes.csv`, `summary.json`. | Kết quả sau sửa evaluator/alias,3/3 episode đặt đủ3 vật. Đây là nhóm kết quả placement hiện hành. |
| `controller_evaluation/` | Cùng bộ tên episode/CSV/summary cho seeds42/43/44. | Kết quả controller lượt3 trước sửa evaluator, giữ làm lịch sử. |
| `three_pegs_demo/` | `controller_demo.mp4`, `final_closeup.png`, `result.json`, `trace.json`. | Video/controller lượt3 trước sửa placement, không tự chứng minh evaluator hiện tại. |
| `stage3/` | `eight_cameras.png`, `overview.png`, `task_closeup.png`, `video_midpoint.png`. | Ba ảnh đầu được `check_mujoco.py` tạo/cập nhật; video_midpoint là ảnh trích kiểm video cũ. Không mặc định tất cả ảnh cùng một run. |
| `stage3_probe/` | `final_closeup.png`, `result.json`, `trace.json`. | Lần chạy thăm dò khi xây dựng controller; xem failure/config bên trong trước khi dùng. |
| `stage3_probe2/` | Cùng ba loại file như probe. | Một lần thăm dò khác, không phải tập test độc lập của ACT. |

## Đọc file nào trước?

`summary.json` ghi mode, seeds, số episode, successful_episodes/success_rate, config_hash và phạm vi. Đọc số episode trước khi diễn giải tỷ lệ;3 seed smoke là kiểm kỹ thuật nhỏ.

`episode_<seed>.json` hoặc `result.json` ghi seed, mode, `act_model_used`, checkpoint, thời gian mô phỏng, failure, từng stage và metrics. `metrics.objects` có grasp/lift/place/task success, lý do thất bại và các phép đo hình học/contact. Bản mới có expected/actual protrusion, sai số đáy và lực đỡ; file lịch sử có thể dùng schema cũ. Tìm config qua snapshot nếu có, hoặc đối chiếu config_hash với audit của run.

`episodes.csv` là bảng tổng hợp tiện so sánh; JSON giữ chi tiết hơn. `trace.json` là dấu vết chạy controller theo bước/pha, phục vụ debug; chưa phải schema demonstrations đầy đủ RGB/action/split cho training. `final_closeup.png` chỉ là ảnh cuối, không thay evaluator. MP4 cho phép xem diễn biến nhưng không thay kiểm contact/ổn định.

Khi chạy mới, chọn `--output outputs/<ten_lan_chay>` để giữ kết quả trước. Ví dụ từ project root:

```powershell
& '.\.venv-data\Scripts\python.exe' -X utf8 scripts/evaluate_mujoco.py --seeds 42 43 44 --output outputs/my_evaluation
```

File lớn/run outputs bị bỏ qua trong Git; README này được giữ. Bằng chứng nhỏ cần tái lập đặt trong [docs/audit](../docs/audit/README.md), kèm config/hash và phạm vi. Không tính ảnh/video đẹp hoặc controller success thành ACT success.

Bổ sung: `cfg5_review/` có review.html, summary.json và selection.json cho bạn chọn ID; selected_manifest_example.json/selected_splits_example.json chỉ là kiểm CLI giữ nguyên57 ID, không thay manifest/split chính. `processed_preview_train/` có index.html, eight_cameras.png, sample.json của một mốc train smoke; dùng để đọc/trình bày dữ liệu.
