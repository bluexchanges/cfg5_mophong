# Phân tích paper RH20T và ACT

Ngày khảo sát: 2026-09-27. Đây là báo cáo lượt đầu, chưa phải kết quả training.

**Cập nhật lượt2:** lựa chọn hiện tại là base-frame TCP8D và8 camera, CFG5 để pretraining cho task khuôn. Xem [DECISIONS.md](DECISIONS.md); các đề xuất chưa chốt bên dưới là lịch sử khảo sát.

## 1. Tài liệu thực sự đã đọc

- `references/papers/RH20T_paper_compressed (1).pdf`: 15 trang, do người dùng cung cấp. Trọng tâm: mục III về dữ liệu; mục IV và bảng IV–V về thí nghiệm ACT.
- `references/papers/2304.13705v1.pdf`: 18 trang, paper ACT/ALOHA bản v1 do người dùng cung cấp. Trọng tâm: mục IV, thí nghiệm mục V–VI và phụ lục C.

Đã trích text toàn bộ hai PDF và kiểm tra trực quan trang 3, 5–6 của RH20T, trang 5–6 của ACT để đối chiếu bảng và công thức. Số trang trong báo cáo là thứ tự trang PDF. Nguồn và hash nằm trong [SOURCES](../references/SOURCES.md) và manifest PDF.

## 2. RH20T đóng góp điều gì?

**Theo paper:** RH20T là dataset robot manipulation đa dạng về task, robot, môi trường, góc camera và modality. Dataset đầy đủ có hơn 110.000 robot sequences, human demonstrations tương ứng, 147 tasks thuộc 42 skills. Các số này mô tả dataset trong paper, không phải số dữ liệu CFG5 trên máy.

Mục tiêu dài hạn là học kỹ năng mới từ rất ít demonstration. Tuy nhiên, mục IV nói rõ tác giả dùng thí nghiệm **few-shot transfer với ACT** để minh họa ích lợi của dataset; paper chưa huấn luyện một foundation model tổng quát thực hiện mọi kỹ năng one-shot. Không nên lấy tên paper làm bằng chứng rằng baseline đã giải quyết hoàn chỉnh one-shot learning.

Quy trình thu thập sử dụng teleoperation với phản hồi lực, camera calibration, timestamp và xử lý về các hệ tọa độ nhất quán. Bảng II, trang 3 xác định **Cfg 5: robot Franka, gripper Franka, force sensing Franka; không có fingertip tactile**. Việc chúng ta dùng Panda phù hợp với cấu hình dữ liệu, nhưng vẫn cần kiểm chứng frame/tool và controller khi chuyển sang MuJoCo.

**Theo bảng III:** RGB, joint angles và gripper width được liệt kê ở 10 Hz; TCP và force/torque ở 100 Hz. Đây là tần số công bố. Không dùng chúng thay cho timestamp thực tế: mẫu trên máy có median khoảng cách timestamp RGB 68 ms, và `high_freq_data.npy` của mẫu robot đầu tiên là dictionary rỗng.

## 3. Thí nghiệm ACT trong RH20T

Nguồn: mục IV.A–B, trang 5–6, đã đối chiếu bảng với ảnh PDF.

| Thành phần | Nội dung paper |
|---|---|
| Robot đánh giá | Flexiv, gripper Dahuan-95, một RealSense phía trước |
| Task đích | Gắp **khối hộp**, đặt nó **lên cân** |
| Dữ liệu môi trường mới | 75 robot demonstrations; thêm điều kiện few-shot 40 và 10 |
| Pretraining | 335 trajectories cùng task; thêm 195 trajectories từ 3 task tương tự |
| Biến đổi domain | Camera pose, table cover, object và robot embodiment khác nhau |
| Kiến trúc | Hidden dimension 512; feedforward dimension 3200 |
| Pretraining schedule | 10 epochs, learning rate 2×10⁻⁵ |
| Fine-tuning schedule | Mô tả chính: 750 epochs, learning rate 10⁻⁵; bảng IV còn có 500 epochs |
| Chunk | 20 actions ở 10 Hz, tương ứng 2 giây |
| Ảnh | Rộng 640, cao 360 trong train và test |
| Temporal ensembling | Hệ số 0,01 |
| Đánh giá | Reach, Pick, Place success rate; tối đa 60 giây mỗi trial |
| Số trial | Bảng IV: 20 trial mỗi cấu hình; bảng V đổi object/table cover: 10 trial |

Ví dụ đúng phạm vi bảng IV, cùng 75 demonstrations và 750 epochs:

| Pretraining | Reach | Pick | Place |
|---|---:|---:|---:|
| Không | 55% | 5% | 0% |
| Cùng task | 80% | 20% | 15% |
| Cùng và các task tương tự | 80% | 25% | 25% |

Không sử dụng các tỷ lệ này làm “mức phải đạt” cho Panda/MuJoCo: robot, dữ liệu fine-tuning, task geometry, controller và domain đều khác. Muốn tái hiện định lượng bảng IV phải tái lập các điều kiện tương ứng, bao gồm tập demonstration mới và protocol 20 trial.

Paper nói tăng sample density bằng cách đưa mọi sub-trajectory hợp lệ vào mỗi epoch. Paper không đủ để suy ra chính xác validation loader, split, checkpoint criterion và action encoding của code được phát hành; các mục đó phải đọc source. Phân tích source nằm trong [ACT_IMPLEMENTATION_REVIEW](ACT_IMPLEMENTATION_REVIEW.md).

## 4. ACT hoạt động thế nào?

Imitation learning học ánh xạ từ observation sang action dựa trên demonstration. ACT dự đoán **một chuỗi action tương lai**, thay vì chỉ action tại một bước.

Viết observation là `o_t = (I_t^1, ..., I_t^M, s_t)`, với ảnh RGB từ M camera và trạng thái robot `s_t`. Model dự đoán chunk `A_t` gồm K vector action, mỗi vector có d phần tử. Dạng tensor tổng quát:

```text
images:   [B, M, 3, H, W]
state:    [B, d_state]
actions:  [B, K, d_action]
is_pad:   [B, K]           # True tại phần tử padding
```

B là batch size. Tên biến `qpos` trong code ACT không tự chứng minh dữ liệu đó là joint angle: baseline RH20T truyền TCP vào tham số này.

### Observation và action trong ACT gốc

Paper ACT dùng 4 ảnh 640×480 và 14 giá trị state/action cho hai tay ALOHA: mỗi tay gồm 6 khớp cánh tay và 1 giá trị gripper. Không phải Panda có 14 khớp. Observation lấy trạng thái follower; target action lấy joint position của **leader**, là lệnh đích cho follower.

Trang 4 nhấn mạnh leader target khác measured follower state: sai lệch này liên quan tới lực qua controller. Vì vậy, nếu sau này dùng `q(t+1)` đo được làm target cho Panda, phải gọi đó là **future-state proxy**, không giả định đó là command đã ghi của chuyên gia.

### Kiến trúc CVAE

1. Khi training, encoder nhận `[CLS]`, state hiện tại và K target actions; Transformer sinh `mu` và `logvar` của latent `z`.
2. Lấy mẫu `z = mu + exp(0.5 logvar) * epsilon`, với `epsilon` Gaussian chuẩn.
3. ResNet18 trích đặc trưng từng ảnh; vị trí không gian được thêm vào feature. Transformer kết hợp feature ảnh, state và latent.
4. Transformer decoder sinh K action vectors bằng các query tương ứng với vị trí trong chunk.
5. Khi inference, không đưa target actions vào encoder; dùng `z = 0` rồi dự đoán chunk từ observation.

CVAE giúp mô hình học các cách thực hiện khác nhau từ demonstration của con người. Không phải reinforcement learning: pipeline baseline không tối ưu reward qua tương tác MuJoCo.

### Loss và padding

Mô tả implementation ở trang 6 ACT dùng:

`L = L_reconstruction + beta * D_KL(q(z | state, actions) || N(0, I))`.

Reconstruction dùng L1; KL buộc phân bố latent gần Gaussian chuẩn. Với diagonal Gaussian, KL mỗi sample là `0.5 * sum(mu² + exp(logvar) - 1 - logvar)`.

**Chi tiết cần phân biệt:** Algorithm 1 trang 5 của PDF v1 ghi MSE, nhưng phần văn bản trang 6 nói L1; cả ba repository đã đọc dùng L1. Khi triển khai, ưu tiên L1 theo implementation và ghi rõ sự khác biệt trong bản v1.

Nếu chunk đến cuối episode, chỉ một phần target còn tồn tại. Ví dụ K=4 và còn 2 actions thật, mask phải là `[False, False, True, True]`. Phần padding không được coi là demonstration thật. Source hiện dùng `.mean()` sau masking nên denominator vẫn bao gồm vị trí pad; đây là một lựa chọn loss khác với chia cho số phần tử hợp lệ. Cần ghi rõ cả loss dùng training lẫn metric dùng chọn checkpoint.

### Chunk alignment và temporal ensembling

Quy ước target phụ thuộc nguồn dữ liệu. Với measured state dùng để dự đoán tương lai, có thể đặt `A_t = [s_(t+1), ..., s_(t+K)]`. Với command được ghi đồng bộ, phải xác định offset đúng qua timestamp, không sao chép offset từ ALOHA.

Ví dụ 10 Hz và K=20 nghĩa là dự đoán 2 giây **chỉ khi đã resample đều 10 Hz**. Danh sách frame đã bỏ các đoạn ít chuyển động có thể không còn khoảng cách 0,1 giây; 20 frames khi đó không đồng nghĩa 2 giây.

Temporal ensembling kết hợp nhiều dự đoán cho **cùng một thời điểm thực thi**, không trung bình các action thuộc thời điểm khác nhau. Paper ACT v1 định nghĩa trọng số `exp(-m*i)`, i=0 là dự đoán cũ nhất. Cần giữ rõ thứ tự này khi triển khai để không vô tình đảo ưu tiên. Với quaternion, trung bình trực tiếp còn cần xử lý dấu, chuẩn hóa và trường hợp gần vector 0; đây là bổ sung bắt buộc nếu dùng TCP pose.

## 5. Training, validation, test và checkpoint

- **Training:** tối ưu tham số trên train data; log reconstruction/KL/total loss và kiểm tra NaN/Inf.
- **Offline validation:** không cập nhật gradient; dùng để chọn checkpoint/config. Phải biết số observation–chunk được xét, không chỉ số episode.
- **Offline test:** giữ riêng, chỉ dùng sau khi khóa checkpoint và config; dùng đúng preprocessing/normalization của train.
- **Simulation rollout:** observation phụ thuộc những action đã thực thi và dynamics. Offline loss thấp không tự bảo đảm gắp thành công.

ACT gốc chọn model có validation loss thấp nhất rồi rollout. Code validation có target actions nên vẫn dùng posterior CVAE và lấy mẫu latent; `eval()` không tự biến nó thành inference `z=0`. Đề xuất báo cáo riêng metric posterior và metric prior `z=0`, cố định sample manifest/seed khi so sánh checkpoint.

Phân biệt **pretrained image backbone** với **ACT policy checkpoint đã học robot task**. Các backbone đã đọc bật pretrained ResNet; chúng có thể tải weights khi khởi tạo. Không có policy checkpoint phù hợp CFG5/Panda được xác minh hoặc tải trong lượt này. Code tải `state_dict` không đồng nghĩa hỗ trợ resume đầy đủ optimizer, RNG và epoch.

## 6. Reproduction của dự án sẽ khác paper ở đâu?

| Phần | Paper RH20T | Dự án dự kiến |
|---|---|---|
| Dữ liệu | Các subset chọn cho pretrain + dữ liệu môi trường Flexiv mới | CFG5 hiện có; chưa xác định subset/task chính |
| Robot đánh giá | Flexiv thật | Panda trong MuJoCo |
| Task | Khối hộp đặt lên cân | Pick-and-place khối hộp, cần chốt task và geometry |
| Action representation | Paper không mô tả đủ; public baseline dùng TCP trong camera frame + gripper | Đề xuất giữ TCP 8D; joint 8D là phương án khác cần xác nhận |
| Modality | Dataset đa modality; baseline thí nghiệm dùng ảnh/action | RGB + state ở baseline đầu, không tự thêm force/depth |
| Kết quả | Few-shot real-robot transfer | Offline imitation và kiểm thử chuyển policy sang simulation |

Code baseline chính thức đã tìm thấy ở [rh20t/act_baseline](https://github.com/rh20t/act_baseline/tree/107ce1287a46be3a1a7ea954f8bbdec3bb103288). Tuy nhiên, README mô tả ví dụ pretraining; chưa đủ bằng chứng rằng repository chứa trọn bộ dữ liệu, fine-tuning và evaluator của thí nghiệm bảng IV. Không kết luận “paper không có code”, cũng không khẳng định “chạy repo là tái hiện bảng IV”.

Một ACT học trên ảnh thật có thể thất bại trên ảnh render vì domain gap. Cần giữ ba loại kết quả riêng: controller thủ công, offline test CFG5 và ACT rollout MuJoCo. Nếu bổ sung demonstrations mô phỏng/fine-tuning về sau, ghi thành thí nghiệm riêng.
