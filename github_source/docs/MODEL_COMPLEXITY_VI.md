# Độ lớn và khối lượng tính toán ACT CFG5

Đếm ngày 2026-10-06, cấu hình `configs/act/rh20t_all8.json`: ảnh8camera360x640; state8D; chunk20x8D; hidden512, FFN3200, image encoder4, decoder7, CVAE encoder4. Không tải checkpoint hoặc dữ liệu.

Số tham số: **83.873.929**. ResNet18 backbone11.166.912 (shared8camera); Transformer55.031.680; CVAE encoder17.332.736; projection/embedding/head còn342.601. Tham số khác FLOPs: một trọng số có thể được dùng ở rất nhiều pixel/token/camera.

| Chế độ | Batch1 | Batch8 |
|---|---:|---:|
| Inference forward |249,724GFLOPs|1.997,789GFLOPs|
| Train/val forward có action encoder |250,489GFLOPs|2.003,913GFLOPs|
| Train backward |492,297GFLOPs|3.938,377GFLOPs|
| Train forward+backward |742,786GFLOPs|5.942,289GFLOPs|

1GFLOP=1tỷ phép tính; nhân+cộng tính2FLOPs theo công thức counter. Với58.313train/6.214val, train+val một epoch khoảng44,87PFLOPs, chưa optimizer/I/O và các phép elementwise. Batch lớn giảm số update nhưng không giảm tương ứng tổng tính toán xử lý cùng số sample.

Phương pháp: `FakeTensorMode` chạy cấu trúc model thật với tensor shapes, `FlopCounterMode` PyTorch2.7.1 đếm global conv/matmul cho forward và backward; không chạy training số học toàn model. Đã đối chiếu với forward/backward thật của ACT thu nhỏ32x32/hidden32/1layer và số đếm trùng nhau1.742.212.352FLOPs. Các phép softmax, layer norm, activation, loss, bias, optimizer, copyCPU-GPU không được đếm đầy đủ. Chạy rawDETRVAE để bỏ wrapper kiểmfinite; maskdummy allvalid để loss không cần boolean-index dynamic shapes. Padding vẫn xử lý attentiondense, không đổi các chiều chính.

Báo cáo máy đọc: `docs/audit/model_complexity.json`; script tái lập: `scripts/count_model_complexity.py`. Đây là ước lượng thuật toán từ config local, không phải latency/FLOP thực thi measured trên serverGPU. Nếu server đổi modelconfig/image/chunk thì cần đếm lại.

Nguồn phương pháp: [PyTorch flop_counter 2.7.1](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/utils/flop_counter.py), [PyTorch profiler](https://docs.pytorch.org/docs/2.7/profiler.html).
