# Bằng chứng kiểm tra nhỏ

Các file ở đây ghi kết quả quan sát tại một lần chạy. Hash xác định nội dung tương ứng, không tự chứng minh thuật toán đúng. Báo cáo cũ có thể không khớp config hiện tại; giữ để truy vết, không gộp mọi kết quả thành một thí nghiệm.

| File | Nội dung và phạm vi |
|---|---|
| `project_inventory.json` | Kiểm kê raw theo metadata/tên episode/file; số task, robot/human, dung lượng nếu có scan. Không xác nhận mọi episode train được. |
| `project_inventory_after_move.json` | Kiểm kê sau move; tổng file/byte vẫn bằng trước khi thêm metadata index. |
| `cfg5_all_organization_plan.json` | Mỗi episode: phân loại, lý do, đường nguồn/đích, quality, fingerprint, số file/byte và ID thư mục; hash bảo vệ plan. |
| `cfg5_all_organization_progress.json` | Tiến độ move cuối cùng; dùng để đối chiếu resume, không phải kiểm định độc lập. |
| `cfg5_all_organization_verified.json` | Xác nhận2515 folder ở đích, không còn root episode, identity/count/byte khớp; không copy payload. |
| `camera_rig_check_after_move.json` | Kiểm8 camera và12 đối chiếu TCP sau khi đổi layout; không phải ACT rollout. |
| `cfg5_all_processed_postrun.json` | Sau người dùng preprocess: đối chiếu597 folder với split531 train/66 val,4.776 header NPY, metadata/hash/steps và byte từng split. Không quét toàn pixel hoặc chạy model. |
| `cfg5_all_train_check.json`, `cfg5_all_val_check.json` | Dựng ACTDataset và đọc3 sample đầu/giữa/cuối mỗi split; báo tổng sample index58.313/6.214, chưa phải validation loss. |
| `payload_samples.json` | Cấu trúc/giá trị mẫu NPY payload để xác minh schema. Chỉ là mẫu, không phải toàn dataset. |
| `upstream_sampling_probe.json` | Fixture tái hiện padding của RH20T baseline và mật độ sample AirExo; không chạy model. |
| `verification.json` | Kiểm chứng lượt1: parse script, hash PDF/upstream, chặn output raw và liên kết tài liệu. |
| `stage2_verification.json` | Snapshot lượt2: môi trường, source hash, split, smoke dataset và camera số học. |
| `train_dataset_smoke.json` | Counts và các sample đầu/giữa/cuối được kiểm trên train smoke. Không phải train loss. |
| `val_dataset_smoke.json` | Cùng loại kiểm trên val smoke; statistics vẫn từ train. Không phải validation ACT. |
| `camera_rig_check.json` | Kiểm roundtrip transform và đối chiếu TCP từ calibration; tách biệt với render. |
| `stage3_verification.json` | Snapshot lượt3 trước sửa placement: code/assets,14 tests tại thời điểm đó và controller/render. |
| `mujoco_check.json` | Báo cáo check scene/camera của lượt chạy trước: phiên bản, số actuator, shape RGB, projection. Muốn tái sử dụng phải đối chiếu config hash. |
| `placement_unit_tests.txt` | Log27 tests đạt sau sửa placement, bao gồm dữ liệu/mô phỏng/logic hình học. |
| `placement_camera_check.json` | Render8 camera sau sửa, shape/type và marker projection; sai số tối đa khoảng0.2213 pixel. |
| `placement_logic_verification.json` | Audit sau sửa: config/alias, kích thước suy ra,27 tests, controller seeds42/43/44, metric từng chốt, hash source/config/tests và liên kết camera report. |

Kết quả placement hiện dùng `placement_logic_verification.json`; episode chi tiết ở `../../outputs/placement_logic_review/`. Cả3 seed đều thành công trong controller smoke, chưa phải success rate ACT hoặc bằng chứng tổng quát hóa.

Các script kiểm có thể ghi lại output mặc định. Khi kiểm phiên bản mới, chọn tên report/thư mục output mới nếu cần giữ lịch sử; ghi đủ seed/config/command/phạm vi. Không sửa tay Boolean hoặc metric để khớp kỳ vọng. Tài liệu mới không làm những audit lịch sử tự được chạy lại.

Bổ sung lượt4a: `act_integration_check.json` ghi full ACT inference trên1 train-smoke sample và backward cấu hình giảm,0 optimizer steps; `act_integration_verification.json` ghi tests/dependencies/source hashes và kiểm selection/preview; `act_unit_tests.txt` là log35 tests. Các báo cáo này không phải kết quả pretraining.
