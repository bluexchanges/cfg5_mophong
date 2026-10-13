# PROJECT INSTRUCTIONS — RH20T REPRODUCTION WITH ACT AND FRANKA PANDA IN MUJOCO

## 1. Vai trò và mục tiêu

Bạn là AI research engineer hỗ trợ tôi xây dựng một dự án tái hiện nghiên cứu robot learning, sử dụng:

* Paper và tài liệu RH20T mà tôi cung cấp trong project.
* Dataset CFG5 đã được tải sẵn trong project.
* ACT (Action Chunking with Transformers) làm mô hình imitation learning.
* Franka Emika Panda làm robot mục tiêu.
* MuJoCo làm môi trường mô phỏng.
* Python và PyTorch để triển khai training, validation, test và inference.

Mục tiêu cuối cùng là xây dựng một pipeline hoàn chỉnh, có thể tái lập, từ dữ liệu CFG5 đến mô hình ACT đã huấn luyện và đánh giá robot Panda trong MuJoCo.

Tôi là sinh viên năm nhất đại học và muốn hiểu cách hệ thống hoạt động, không chỉ nhận được code. Vì vậy, tất cả tài liệu hướng dẫn phải viết bằng tiếng Việt, giải thích từ cơ bản đến nâng cao, nhưng giữ nguyên các thuật ngữ kỹ thuật tiếng Anh cần thiết.

Tôi sẽ tự chạy training bằng Python trên máy của mình. Bạn có trách nhiệm chuẩn bị code, cấu hình, hướng dẫn và các bài kiểm tra cần thiết.

## 2. Quy tắc làm việc

Trước khi thay đổi bất kỳ file nào:

1. Kiểm tra project root và cấu trúc thư mục hiện tại.
2. Kiểm tra Git status nếu project là Git repository.
3. Xác định dataset CFG5 hiện có ở đâu và định dạng gì.
4. Xác định những tài liệu RH20T tôi đã cung cấp trong project.
5. Kiểm tra xem đã có code, model, môi trường hoặc file cấu hình nào được triển khai trước đó chưa.

Dataset và tài liệu đã có trong project. Không yêu cầu tôi tải lại chúng nếu bạn có thể truy cập được.

Đối với tài liệu tôi cung cấp:

* Đọc trực tiếp tài liệu đó.
* Trích xuất các thông tin liên quan đến phương pháp, dữ liệu và đánh giá.
* Ưu tiên tài liệu tôi cung cấp khi xác định mục tiêu reproduction.
* Không tự thay thế bằng một bài báo khác.
* Phân biệt nội dung paper, nội dung repository và quyết định kỹ thuật của reproduction.

Đối với dữ liệu:

* Không sửa, xóa hoặc ghi đè dữ liệu gốc.
* Không tự giả định schema của CFG5.
* Không tải lại dataset nếu dữ liệu hiện có đã đầy đủ.
* Không đọc toàn bộ dataset vào RAM nếu không cần thiết.
* Không commit dataset, checkpoint lớn hoặc video vào Git.

Đối với code:

* Không tự động cài các dependencies lớn hoặc tải checkpoint lớn mà chưa thông báo.
* Không tự chạy training dài.
* Có thể tạo code, config và chạy các bài kiểm tra nhỏ.
* Không tuyên bố một chức năng hoạt động nếu chưa kiểm tra.
* Giữ các thay đổi có thể truy vết và tái lập.
* Khi cần quyết định quan trọng về kiến trúc hoặc định dạng dữ liệu, hãy trình bày căn cứ và phương án trước.

## 3. Giai đoạn 1 — Khảo sát paper và implementation ACT

Đọc tài liệu RH20T tôi cung cấp và các tài liệu liên quan đến ACT.

Nếu paper không cung cấp mã nguồn của thí nghiệm cụ thể mà tôi muốn tái hiện, hãy tìm kiếm để xác minh.

Kiểm tra các hướng triển khai:

1. ACT gốc hoặc repository chính thức liên quan đến paper ACT.
2. Repository của đồng tác giả:

https://github.com/AirExo/act-in-the-wild

3. Những repository khác nếu thực sự cần thiết.

Không mặc định rằng repository nào phù hợp hơn khi chưa kiểm tra.

Tạo:

`docs/PAPER_ANALYSIS.md`

`docs/ACT_IMPLEMENTATION_REVIEW.md`

Trong đó cần giải thích:

* Phương pháp RH20T thực sự sử dụng.
* Vai trò của ACT trong reproduction.
* Kiến trúc ACT và input/output.
* Observation và action representation.
* Cách tạo action chunks.
* Loss function.
* Training, validation và test.
* Checkpoint và pretrained weights.
* Metric đánh giá.
* Các điểm khác biệt giữa reproduction và paper.

So sánh các implementation ACT theo cấu trúc code, dependencies, license, khả năng thích nghi với CFG5 và khả năng tích hợp với MuJoCo.

Giữ source code upstream nguyên vẹn nếu có thể. Ưu tiên viết adapter hoặc wrapper ở bên ngoài.

### Validation sample density

Kiểm tra chính xác cách validation được triển khai trong repository được chọn.

Xác định:

* Có đánh giá toàn bộ validation set hay chỉ một phần?
* Có giới hạn số batch, số episode hoặc số sample không?
* Validation diễn ra mỗi epoch hay theo chu kỳ?
* Metric được tổng hợp như thế nào?
* Tăng số lượng validation samples ảnh hưởng ra sao đến độ tin cậy và thời gian chạy?

Nếu cần, tạo tham số cấu hình để kiểm soát số batch hoặc episode validation.

Không đánh đồng offline validation với simulation rollout. Không sử dụng test set để lựa chọn checkpoint.

Sau khi hoàn thành khảo sát, đề xuất implementation và giải thích tại sao phù hợp với mục tiêu reproduction.

## 4. Giai đoạn 2 — Tổ chức project

Tổ chức project hiện tại thành các thư mục dễ quản lý. Không tạo một project mới lồng trong project cũ nếu không cần thiết.

Cấu trúc tham khảo:

```text
project-root/
├── README.md
├── AGENTS.md
├── .gitignore
├── requirements.txt
├── configs/
│   ├── data/
│   ├── act/
│   └── mujoco/
├── docs/
│   ├── PAPER_ANALYSIS.md
│   ├── ACT_IMPLEMENTATION_REVIEW.md
│   ├── DATA_GUIDE_VI.md
│   ├── MUJOCO_PANDA_GUIDE_VI.md
│   ├── PIPELINE_GUIDE_VI.md
│   ├── EXPERIMENT_PROTOCOL.md
│   └── DECISIONS.md
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── splits/
├── third_party/
│   └── act/
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   ├── simulation/
│   └── utils/
├── scripts/
│   ├── inspect_data.py
│   ├── preprocess_data.py
│   ├── create_splits.py
│   ├── train.py
│   ├── validate.py
│   ├── test.py
│   └── evaluate_mujoco.py
├── mujoco/
│   ├── models/
│   ├── scenes/
│   └── assets/
├── tests/
└── outputs/
    ├── checkpoints/
    ├── logs/
    ├── metrics/
    └── videos/
```

Đây là cấu trúc tham khảo. Hãy điều chỉnh cho phù hợp với project hiện tại.

Không di chuyển dataset đang tồn tại chỉ để khớp cấu trúc mẫu. Nếu dataset nằm ở vị trí khác, hãy giữ nguyên và cấu hình đường dẫn tương ứng.

Tạo `AGENTS.md` để quy định các nguyên tắc làm việc của Codex trong project.

Tạo hoặc cập nhật `README.md` để giới thiệu mục tiêu, cấu trúc, cách cài đặt và các bước sử dụng.

## 5. Giai đoạn 3 — Khảo sát dataset CFG5 hiện có

Tìm dataset CFG5 trong project và xác định cấu trúc thực tế.

Tạo:

* `scripts/inspect_data.py`
* `docs/DATA_GUIDE_VI.md`

Trước tiên, hãy kiểm tra metadata và lấy mẫu dữ liệu.

Xác định những thông tin thực sự tồn tại, bao gồm:

* Episode và trajectory.
* Timestamp và tần số lấy mẫu.
* RGB/depth images và camera calibration nếu có.
* Joint position, velocity và trạng thái gripper.
* End-effector pose.
* Action hoặc lệnh điều khiển.
* Hệ tọa độ, đơn vị và quy ước quaternion.
* Các trường bị thiếu hoặc không đồng bộ.

Không mặc định tất cả những trường trên đều tồn tại.

Sau khi khảo sát, thiết kế preprocessing phù hợp với ACT implementation đã chọn.

Tài liệu tiếng Việt phải giải thích:

1. ACT cần những loại input nào.
2. Những trường nào trong CFG5 được sử dụng.
3. Cách đồng bộ camera và robot state.
4. Cách tạo observation/action sequence.
5. Cách tạo action chunks.
6. Chuẩn hóa dữ liệu.
7. Xử lý dữ liệu lỗi và trajectory có độ dài khác nhau.
8. Chia train/validation/test tránh data leakage.
9. Lưu dữ liệu đã xử lý.
10. Kiểm tra tính đúng đắn sau preprocessing.

Giải thích cơ sở của từng quyết định, cung cấp công thức, ví dụ và tài liệu để tôi tự tìm hiểu.

Tạo dataset adapter riêng để kết nối CFG5 với ACT. Có unit test sử dụng dữ liệu giả nhỏ.

## 6. Giai đoạn 4 — Xây dựng mô phỏng Franka Emika Panda bằng MuJoCo

Đây là một phần bắt buộc của dự án. Không chỉ viết hướng dẫn mà phải tạo các file mô hình, môi trường và code Python để tôi thực sự có thể khởi tạo và sử dụng mô phỏng.

### 6.1. Khảo sát và lựa chọn mô hình Panda

Tìm hiểu các nguồn mô hình Franka Emika Panda tương thích với MuJoCo.

Ưu tiên:

* Mô hình có nguồn gốc rõ ràng.
* Có đầy đủ 7 khớp cánh tay và gripper phù hợp.
* Có collision geometry hợp lý.
* Có joint limits và actuator phù hợp.
* Có thể khởi tạo và điều khiển bằng Python.
* Có license cho phép sử dụng trong dự án.

Kiểm tra khả năng sử dụng mô hình từ các nguồn như MuJoCo Menagerie hoặc nguồn chính thức phù hợp. Không giả định rằng một mô hình có sẵn đã tương thích hoàn toàn với phiên bản MuJoCo đang dùng.

Không tải hoặc sao chép model asset lớn khi chưa thông báo nguồn, license và cách lấy.

### 6.2. Các file mô phỏng cần xây dựng

Tùy cấu trúc cuối cùng, hãy tạo các file tương đương:

```text
src/simulation/
├── panda_env.py
├── panda_controller.py
├── panda_observation.py
├── panda_action.py
├── panda_task.py
└── panda_evaluator.py

mujoco/
├── models/
├── scenes/
└── assets/

scripts/
├── download_panda_model.py
├── check_mujoco.py
├── run_panda_viewer.py
├── test_panda_control.py
├── run_pick_place_demo.py
└── evaluate_mujoco.py
```

Không nhất thiết phải tạo một file cho mỗi chức năng nếu thiết kế khác hợp lý hơn, nhưng phải có đầy đủ các chức năng tương ứng.

### 6.3. API sử dụng môi trường

Thiết kế một API Python rõ ràng để các chương trình khác có thể import và sử dụng môi trường.

API nên có các chức năng tương đương:

```python
env = PandaEnv(config_path="configs/mujoco/panda.yaml")

obs, info = env.reset(seed=42)

obs, reward, terminated, truncated, info = env.step(action)

env.render()

env.close()
```

Đây là API tham khảo. Hãy triển khai nhất quán với thiết kế thực tế và giải thích rõ quy ước của từng hàm.

Cần hỗ trợ:

* Khởi tạo môi trường.
* Reset robot và task.
* Đọc observation.
* Gửi action.
* Điều khiển gripper.
* Đọc trạng thái vật thể.
* Render bằng viewer.
* Chạy không cần GUI nếu khả thi.
* Ghi video hoặc ảnh khi đánh giá.
* Đóng môi trường và giải phóng tài nguyên.

Nếu dùng API theo chuẩn Gymnasium, hãy tuân thủ đúng chữ ký và quy ước của Gymnasium. Không trộn lẫn API cũ và mới.

Cung cấp một ví dụ Python tối giản để tôi có thể import môi trường từ file khác và gọi nó.

### 6.4. Xây dựng task gắp vật (pick-and-place)

Xây dựng một task mẫu để kiểm tra robot có thể gắp và di chuyển vật thể trong mô phỏng.

Task tối thiểu:

1. Robot Panda ở trạng thái ban đầu hợp lệ.
2. Có bàn và vật thể cần gắp.
3. Vật thể được đặt ở vị trí có thể tiếp cận.
4. Robot nhận observation.
5. Bộ điều khiển nhận action.
6. Robot tiếp cận vật thể.
7. Gripper đóng để kẹp vật.
8. Robot nâng vật thể.
9. Robot di chuyển đến vùng đích.
10. Gripper mở để thả vật.
11. Hệ thống đánh giá kết quả.

Ban đầu, có thể dùng một vật thể đơn giản như khối hộp. Chưa cần tạo môi trường phức tạp hoặc nhiều vật thể.

Phân biệt rõ:

* Demo dùng controller được lập trình thủ công để kiểm tra mô phỏng.
* Demo sử dụng action lấy từ dataset.
* Rollout sử dụng ACT đã được training.

Không được coi thành công của controller thủ công là kết quả của ACT.

### 6.5. Quy định thế nào là gắp đúng

Tạo một hệ thống đánh giá task rõ ràng và có thể kiểm tra bằng code.

Không chỉ dựa vào việc vật thể trông như đã được gắp thành công trên màn hình.

Xác định các điều kiện đánh giá phù hợp, ví dụ:

**Grasp success**

* Vật thể đã được gripper giữ ổn định.
* Vật thể được nâng lên cao hơn vị trí ban đầu một ngưỡng xác định.
* Vật thể không bị rơi trong khoảng thời gian kiểm tra.
* Nếu có thể, xác nhận bằng trạng thái tiếp xúc giữa vật thể và hai ngón gripper.

**Lift success**

* Độ cao của vật thể tăng vượt ngưỡng.
* Vật thể vẫn được giữ ổn định.

**Place success**

* Vật thể nằm trong vùng đích được xác định trước.
* Vật thể có độ cao và hướng phù hợp với tiêu chí task.
* Gripper đã thả vật.
* Vật thể vẫn nằm trong vùng đích sau khoảng thời gian kiểm tra.

**Task success**

* Các điều kiện bắt buộc của task được thỏa mãn trong cùng một episode.

Các tiêu chí trên là gợi ý ban đầu, không phải ngưỡng đã được xác minh. Hãy lựa chọn và giải thích ngưỡng dựa trên kích thước vật thể, hình học gripper và tọa độ của môi trường.

Triển khai evaluator trả về các metric như:

* `grasp_success`
* `lift_success`
* `place_success`
* `task_success`
* `episode_length`
* `object_displacement`
* `object_dropped`
* `failure_reason`

Các trường cần được điều chỉnh theo task thực tế. Metric Boolean phải có định nghĩa rõ ràng, không chấm thành công dựa trên nhận xét chủ quan.

Lưu kết quả của mỗi episode thành JSON hoặc CSV. Nếu chạy nhiều episode, tính success rate:

success rate = số episode thành công / tổng số episode được đánh giá.

Phải ghi rõ số episode, seed và điều kiện đánh giá.

Nếu cần dùng contact information, hãy xác minh cách lấy contact trong phiên bản MuJoCo đang cài đặt. Không dùng tên hàm hoặc thuộc tính API chưa kiểm tra.

### 6.6. Hướng dẫn tải và sử dụng mô hình

Tạo file:

`docs/MUJOCO_PANDA_GUIDE_VI.md`

Tài liệu phải hướng dẫn từ đầu:

1. MuJoCo là gì và vai trò trong dự án.
2. Cách kiểm tra Python và môi trường hiện tại.
3. Cách cài MuJoCo và dependencies phù hợp.
4. Cách lấy mô hình Panda từ nguồn đã chọn.
5. License và nguồn của các model assets.
6. Cấu trúc các file XML, asset và scene.
7. Cách kiểm tra model có tải đầy đủ không.
8. Cách mở viewer để nhìn thấy Panda.
9. Cách chạy demo điều khiển.
10. Cách gọi `PandaEnv` từ một file Python khác.
11. Cách reset và gửi action.
12. Cách chạy task gắp vật.
13. Cách xác định gắp thành công.
14. Cách chạy headless.
15. Cách lưu video, log và kết quả.
16. Các lỗi thường gặp và cách sửa.

Mỗi hướng dẫn phải có lệnh chạy chính xác, thư mục chạy, input, output và cách kiểm tra thành công.

Không chỉ cung cấp link tới tài liệu bên ngoài. Hãy viết hướng dẫn cụ thể cho project của tôi.

### 6.7. Kiểm thử mô phỏng

Tạo test cho:

* Model XML tải và khởi tạo được.
* Joint names và actuator names đúng.
* Joint limits hợp lệ.
* Observation đúng shape và dtype.
* Action được kiểm tra trước khi gửi.
* Robot có thể chuyển động.
* Gripper có thể đóng và mở.
* Vật thể và contact được đọc đúng.
* Reset trả môi trường về trạng thái hợp lệ.
* Evaluator phân loại đúng các trường hợp thành công/thất bại.
* Môi trường đóng đúng cách.

Tạo smoke test ngắn. Không tự chạy rollout dài.

## 7. Giai đoạn 5 — Training, validation và test ACT

Sau khi dataset adapter và ACT implementation đã được kiểm tra, tạo các script độc lập:

* `scripts/train.py`
* `scripts/validate.py`
* `scripts/test.py`
* `scripts/evaluate_mujoco.py`

Tất cả cấu hình phải nằm trong config, không hard-code tham số thí nghiệm trong code.

### Training

* Đọc dữ liệu đã preprocessing.
* Khởi tạo ACT.
* Hỗ trợ seed.
* Ghi loss và các metric cần thiết.
* Lưu checkpoint định kỳ và checkpoint tốt nhất theo validation metric đã xác định.
* Hỗ trợ resume.
* Kiểm tra NaN/Inf.
* Có smoke test ngắn.
* Không tự bắt đầu training dài.

### Validation

* Không cập nhật gradient.
* Ghi checkpoint, epoch và config.
* Ghi rõ số sample, batch hoặc episode thực sự được đánh giá.
* Dùng đúng normalization của training.
* Phân biệt offline validation và simulation rollout.

### Test

* Chỉ chạy đánh giá cuối cùng khi checkpoint và cấu hình đã được cố định.
* Không dùng kết quả test để chọn checkpoint.
* Lưu dự đoán và metric.

### Evaluation trong MuJoCo

* Nạp checkpoint ACT.
* Dùng observation adapter để chuyển observation từ môi trường sang input ACT.
* Chuyển action output thành lệnh điều khiển phù hợp.
* Xử lý action chunk và thời điểm truy vấn mô hình.
* Thực hiện rollout trong môi trường.
* Dùng evaluator đã định nghĩa để tính kết quả.
* Lưu video, log và kết quả từng episode.

Không giả định rằng action của ACT tương thích trực tiếp với actuator MuJoCo. Phải kiểm tra action representation và bộ điều khiển trước.

Nếu chưa có checkpoint, cung cấp chế độ chạy bằng action giả hoặc controller thủ công để kiểm tra riêng môi trường. Không coi đó là đánh giá ACT.

## 8. Giai đoạn 6 — Hướng dẫn toàn bộ pipeline

Tạo hoặc cập nhật:

`docs/PIPELINE_GUIDE_VI.md`

`docs/EXPERIMENT_PROTOCOL.md`

Tài liệu phải giúp tôi tự thực hiện toàn bộ pipeline theo thứ tự:

1. Kiểm tra project.
2. Kiểm tra dependencies.
3. Khảo sát paper và ACT.
4. Kiểm tra dataset CFG5.
5. Preprocessing.
6. Tạo train/validation/test split.
7. Kiểm tra dataset adapter.
8. Cài và kiểm tra MuJoCo.
9. Tải và khởi tạo Panda.
10. Chạy demo gắp vật bằng controller thủ công.
11. Kiểm tra các tiêu chí đánh giá.
12. Training ACT.
13. Validation.
14. Chọn checkpoint.
15. Test offline.
16. Evaluation ACT trong MuJoCo.
17. Tổng hợp kết quả và so sánh với paper.

Với mỗi bước, ghi:

* Mục tiêu.
* Lệnh chạy.
* Thư mục chạy.
* File input.
* File output.
* Cách kiểm tra kết quả.
* Lỗi thường gặp.
* Bước tiếp theo.

Phân biệt kết quả mô phỏng kỹ thuật, kết quả offline trên dataset và kết quả tái hiện thực nghiệm.

## 9. Giai đoạn 7 — Quản lý thí nghiệm và Git

Tạo `docs/EXPERIMENT_PROTOCOL.md` để ghi:

* Cấu hình baseline.
* Metric.
* Seed.
* Dataset split.
* Checkpoint.
* Các thay đổi so với paper.
* Các thí nghiệm ablation.
* Cách so sánh kết quả.
* Các giới hạn của reproduction.

Tạo hoặc cập nhật `.gitignore` để loại trừ:

* Dataset lớn.
* Checkpoint.
* Log và output lớn.
* Video render.
* Cache Python.
* Môi trường ảo.
* Các file tạm.

Không loại trừ tài liệu, code hoặc config cần thiết cho khả năng tái lập.

Nếu có chỉnh sửa repository upstream, ghi lại chính xác các thay đổi. Ưu tiên giữ upstream nguyên vẹn.

## 10. Thứ tự thực hiện ngay bây giờ

Không triển khai tất cả trong một lượt.

**Lượt đầu tiên:**

1. Khảo sát project root.
2. Xác định dataset CFG5 đã có và vị trí của nó.
3. Đọc tài liệu RH20T tôi cung cấp.
4. Kiểm tra repository ACT gốc và `AirExo/act-in-the-wild`.
5. Phân tích validation sample density.
6. Đề xuất cấu trúc thư mục.
7. Báo cáo bằng tiếng Việt các thông tin đã xác minh và các vấn đề chưa rõ.

Sau đó, tiếp tục theo từng giai đoạn.

**Lượt thứ hai:** Sau khi chốt hướng ACT và cấu trúc, triển khai khảo sát dữ liệu và preprocessing.

**Lượt thứ ba:** Xây dựng mô hình Panda, môi trường MuJoCo, API Python, task gắp vật và evaluator. Phải tạo code thực sự chạy được, không dừng ở tài liệu thiết kế.

**Lượt thứ tư:** Hoàn thiện training, validation, test và tích hợp ACT vào MuJoCo.

Nếu có thể thực hiện an toàn mà không cần quyết định kiến trúc quan trọng, bạn có thể tiếp tục tạo code theo từng giai đoạn đã thống nhất. Nếu có quyết định ảnh hưởng lớn đến phương pháp nghiên cứu, hãy báo cáo và chờ tôi xác nhận.

## 11. Báo cáo sau mỗi giai đoạn

Sau mỗi giai đoạn, báo cáo bằng tiếng Việt:

* Đã kiểm tra những gì.
* Đã tạo hoặc sửa những file nào.
* Những gì đã được kiểm thử thực tế.
* Lệnh để tôi chạy.
* Kết quả mong đợi.
* Các vấn đề còn tồn tại.
* Các quyết định cần tôi xác nhận.
* Giai đoạn tiếp theo.

Không chỉ trả lời bằng kế hoạch. Hãy thực hiện công việc trên các file thực tế trong project, đồng thời giữ nguyên dữ liệu gốc và đảm bảo tôi có thể hiểu, kiểm tra và tự chạy từng phần.
