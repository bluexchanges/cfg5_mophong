# Test mô hình ACT sau khi hoàn tất 80 epoch

## Dừng ngay theo yêu cầu mới 2026-10-07

Gói `cfg5_pretrain_test_now_bundle.zip` hỗ trợ dừng lượt BS8 và test ngay
checkpoint best đã lưu, không cần chờ80:

```bash
python3 "$HOME/cfg5-pretrain/cfg5_pretrain_test_now_bundle/scripts/slurm/submit_pretrain_test.py" --stop-now
```

Launcher đọc đúng `act_bs8_to80.jobid`, đối chiếu user/name/state trong queue,
chỉ dừng job đó. Nếu đã đăng ký testafterok nguồn này và test cònPD, chỉ hủy
job test có đúngdependency nguồn rồi gửi một testmới khôngdependency. Jobkhác
nhưT27 giữ nguyên. Chờ training rời queue rồi mới gửi test. Epoch đang chạy
chưa lưu bị bỏ; không xóa hoặc sửa best/latest/history. Report của chế độ này
ở `outputs/act_pretrain_bs8_to80/test_final/test_report.json`; selection ghi
sốepoch đã lưu thực tế và bestepoch, không ghi rằng đã trainđủ80.

Nếu dừng đúng khoảng sau khi atomicbest được lưu mà history chưa commit,
chấp nhận best của epoch liền tiếp theo khi val thấp hơn history. Trường hợp
khác không khớp thì báo lỗi. Test giữ eval/inference_mode và optimizer_steps0.
Các phần dưới mô tả chế độ mặc định chờ đủ80, vẫn được hỗ trợ.

Test dùng 66 episode giữ riêng của CFG5. Không dùng test để chọn checkpoint,
learning rate, batch size hoặc cấu hình. Chốt lượt 80 epoch và chọn `best.pt`
theo val loss nhỏ nhất, rồi khóa bản sao checkpoint cùng model/data config,
split, manifest và normalization trong `selection.json` trước khi đọc test.
Nếu muốn thay đổi mô hình dựa trên báo cáo này, tập 66 episode đã trở thành
dữ liệu phát triển; không tiếp tục gọi kết quả trên cùng tập là test chưa thấy.

## Mô hình không học trong test

- `policy.eval()` tắt dropout và dùng hành vi evaluation của các layer.
- Toàn bộ lượt forward nằm trong `torch.inference_mode()`.
- Không tạo optimizer, không `backward()`, không `optimizer.step()`.
- Không fit lại normalization; dùng snapshot statistics của 531 train episode.
- SHA256 checkpoint kiểm trước/sau; test fixture cũng so sánh toàn bộ
  parameter/buffer và xác nhận không có gradient sau evaluation.
- Không sửa checkpoint `best.pt`, `latest.pt` hoặc training history.

## Hai nhóm kết quả

1. **Inference:** đầu vào chỉ có ảnh 8 camera và state, latent prior `z=0`.
   Dự đoán chunk20, bỏ vị trí padding khỏi metric. Báo normalized L1,
   khoảng cách TCP trung bình mm, MAE xyz mm, gripper MAE mm và góc quaternion
   trung bình độ. Quaternion chuẩn hóa và coi `q`/`-q` tương đương; quaternion
   norm gần0 bị loại khỏi metric góc, ghi rõ số lượng loại.
2. **Posterior reconstruction:** truyền target action vào CVAE encoder để đo
   `L1 + 10*KL` giống cách tính loss validation. Seed cố định, batch8, worker2.
   Đây là reconstruction loss có điều kiện trên đáp án, không dùng để tuyên bố
   khả năng điều khiển robot. Truyền target để đo loss không cập nhật trọng số.

Các metric inference tính trung bình theo toàn bộ future target hợp lệ của
các chunk chồng lấp; sample/batch/episode/valid step/element đều được báo.
Action TCP tương lai trong CFG5 là measured-state proxy, không phải lệnh
điều khiển cánh tay đã ghi. Đây là đánh giá offline; chưa có simulation
rollout, success rate hoặc kết luận sim-to-real.

## Chạy trên server

Upload `cfg5_pretrain_test_bundle.zip`, giải nén vào
`$HOME/cfg5-pretrain/cfg5_pretrain_test_bundle`, rồi chạy:

```bash
python3 "$HOME/cfg5-pretrain/cfg5_pretrain_test_bundle/scripts/slurm/submit_pretrain_test.py"
```

Launcher gửi một job 1 GPU/8 CPU/64GiB/12h. Nếu lượt BS8 đang chạy,
job test dùng `afterok:<training_jobid>`: chỉ chạy sau khi training thành công.
Nếu training đã kết thúc, chỉ cho gửi khi history lưu đủ80 epoch. Không gửi
lặp khi đã có job test hoặc report. Training thất bại thì kiểm lỗi trước.

Job khóa best model, preprocess66 test episode vào thư mục processed_test
riêng, rồi đánh giá đầy đủ. Nếu preprocessing gián đoạn, episode hoàn chỉnh
được kiểm trước khi bỏ qua; thư mục `.partial` yêu cầu kiểm ERROR.json.
Không chạy lại preprocessing train/val và không giải nén raw trong script.
Nếu raw test còn thiếu hoặc fingerprint khác manifest, job dừng để báo lỗi.

Output: `outputs/act_pretrain_bs8_to80/test_epoch80/`:

- `best.pt`: bản sao checkpoint đã khóa (khoảng321MiB).
- `selection.json`: quy tắc chọn, epoch, val loss, SHA256, config/stat/split.
- `processed_test/`: NPY của test.
- `preprocessing.json`: danh sách test episode đã xử lý.
- `test_report.json`: hai nhóm metric và optimizer_steps=0.

Theo dõi:

```bash
J=$(cat "$HOME/cfg5-pretrain/act_pretrain_test.jobid")
squeue -j "$J" -o "%.18i %.20j %.2t %.10M %R"
tail -n 20 "$HOME/cfg5-pretrain/logs/cfg5-act-test-$J.out"
tail -n 20 "$HOME/cfg5-pretrain/logs/cfg5-act-test-$J.err"
```

Nếu không còn trong queue:

```bash
sacct -j "$J" -X --format=JobID,State,ExitCode,Elapsed
```

Khi thành công có `CFG5_PRETRAIN_TEST_OK`. Đã kiểm fixture CPU với ACT thu
nhỏ; full ACT trên A100 và66 raw test episode do người dùng chạy trên server.
