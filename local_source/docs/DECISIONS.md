# Quyết định dự án

### 2026-10-09: job14119 thiếu MuJoCo runtime hoặc import nhầm

User gửi AttributeError module mujoco không có __version__ tại kiểm dependency,
job failed sau4s, chưa tới model evaluation. Repository có thư mục mujoco/ chứa
asset, nên khi venv không cài package thật thì import vẫn có thể thành công dưới
dạng namespace. Không coi hasattr version là chứng minh runtime đầy đủ.
Đổi check local Slurm sang importlib.metadata.version và bắt buộc MjModel;
thêm diagnostic module file/path. Hướng dẫn cài mujoco==3.14.0 bằng đúng venv,
kiểm metadata+MjModel trước khi submit lại. V1/v2 check cũ cũng chạy được sau khi
cài đúng package có __version__; không yêu cầu upload lại chỉ để xử lý lần này.

### 2026-10-09: sửa dependency của bundle E5 v2

Server báo thiếu prerequisite khi cài E5. Log user bị cắt trước tên file;
chưa xác nhận file thiếu cụ thể. Đối chiếu launcher pretrain_test cho thấy
evaluator cũ chạy từ bundle ngoài repo. Gói E5 cũ đã giả định sai file evaluator
phải tồn tại trong repo. V2 đưa InferenceMetrics sang module transfer_metrics
đi kèm, giữ đúng công thức/masking và kiểm regression với evaluator cũ.
Test fixture không là prerequisite runtime; bổ sung nếu thiếu, giữ nếu đã có.
Installer liệt kê toàn bộ runtime thiếu và repo root trước khi ghi bất kỳ file.
Gói/folder v2 riêng tránh unzip -n giữ lại installer và manifest v1.

## 2026-10-09: diagnostic E5 và visual backbone MCR

User chọn E5 pretrain để đánh giá transfer 10 val demo và MuJoCo, không học;
E58 vẫn là minimum validation toàn lượt. Job14111 đã được user dừng.
Thêm transfer_probe ngoài upstream: snapshot/SHA E5, giữ stats/camera/10Hz,
offline metrics tách closed-loop, reset scene+seed với initial-state guard,
first action mỗi tick, clip workspace/kẹp có raw trace, video và report lỗi hạ tầng.
Không đụng split hoặc checkpoint cũ; held-out test chưa dùng cho phát triển này.

User đã chọn MCR trước sau khi phân biệt DROID Franka với CrossFormer 30 embodiments.
MCR official ResNet50 SHA21560db9... đã tải 94.345.861 byte và kiểm strict-load;
HF revision057651ca0b6a27aba390449603f674cd0977f87c. Nguồn/giới hạn trong EPOCH5_MCR_VI.md.
Adapter MCR2048 channels riêng: giữ ACT E5 tương thích, visual projection mới;
freeze cả gradient và BN running stats. Cache giữ spatial7x7 INT8+FP16 per-channel
scale, JPEG90(SQLite) được decode trước extraction để frozen/full có cùng ảnh đầu vào.
MCR preprocessing center224 theo tác giả; crop, JPEG và INT8 có rủi ro với precision task.
Cache tiết kiệm dung lượng, không phải chỉ weights; payload sim train+val ~317GiB
chưa cộng JPEG. Pilot3 train episode, quota soft1.8TB bao gồm raw/oldcache/checkpoint,
train/val only, atomic per-episode, SHA/chunk+numeric và context guard.
Frozen ACT train CFG5 trước, full fine-tune own split đọc JPEG, không seek MP4;
chưa tuyên bố tốc độ/chất lượng tốt hơn. Archive mỗi epoch nhánh mới để truy xuất.
Code+Panda assets có license đóng gói epoch5_mcr_bundle, server tự tải MCR94MB.
Local chỉ fixture và MuJoCo2steps; không chạy full cache/train/10rollout local.

**Cập nhật 2026-10-07 — user chốt dừng ngay để test:** Bổ sung --stop-now, scancel scope đúng UID/name/ID lượtBS8 sau khi kiểm checkpoint/history. Chỉ thay pendingtest cóafterok nguồn bịhủy; chờ nguồn rờiqueue, khônghủyT27/jobkhác. Khóa bestval trong các epoch đãlưu, cho phép atomicbest ởepochliềnkếhist nếu loss giảm, ghi sốepochthực vào selection; outputtest_final. Khôngđợi80, khôngđào tạo trong test. Có fixture chọn58tronghistory76, atomicboundary vàcancel scope/dependency.

**Cập nhật 2026-10-07 — test pretraining sau80epoch:** Người dùng yêu cầu test tập pretrain và nhấn mạnh mô hình không học. Chuẩn bị jobafterok lượtBS8-to80, chọn best theo val sau đủ80, khóa checkpoint/config/split/train-only normalization trước test66episode. Preprocess test vào outputriêng, eval/inference_mode khôngoptimizer/backward, đo inference ảnh/state latent0 cùng posterior reconstruction loss phân biệt rõ. Metrics target thật bỏpadding, mm/độ ngoài normalizedL1; khôngrollout hoặc successrate. Fixture so weights/buffers trước/sau bằng ACTthu nhỏ; chưa chạy testfullA100. Xem [hướng dẫn test](PRETRAIN_TEST_VI.md).

**Cập nhật 2026-10-05 — giảm sau QoS chặn hai GPU:** User cung cấp job13988 PD/QOSMaxGRESPerJob rồi yêu cầu batch8/worker cũ. Gói thay thế xin1GPU/8CPU/64GiB, lấy worker từ metadatafast40, forkBS8 sang outputs/act_pretrain_bs8_to80 để tiếp80tổng; giữ optimizer/RNG/contract/LR. Launcher chỉ thay13988 nếu đúng user/name vàPD, scancel lọcPENDING tránh race, không hủyjob khác, không gửiACT trùng. Có test/probe trênGPU trước train. Xem [hướng dẫn BS8](BS8_TO80_VI.md).

**Cập nhật 2026-10-05 — tiếp từ 40 đến 80 với batch 16 và hai GPU:** Người dùng báo hoàn thành40 và yêu cầu batch16, thêmGPU, tăngworker. Chuẩn bị một job xin16CPU/128GiB/2GPU, loader8worker, batch16 tổng (khoảng8/GPU). Adapter DataParallel gộp L1 theo valid elements và KL theo samples, giữ bare model keys/optimizer, fork sang outputs/act_pretrain_bs16_w8_gpu2; giữ contract/LR/precision và epoch/global_step/RNG nguồn, GPU mới seeded theo run. DataParallel có replicate/scatter overhead; chưa cam kết speedup, DDP là hướng tối ưu tiếp. Launcher chống job ACT trùng, không hủy job; test hai GPU và probe3batch thực tế trước fork/train dài. Đã chạy38 fixture:33pass,5CUDA skip trên máy CPU; gate CUDA chạy ở server. Tài liệu [chạy](BS16_GPU2_TO80_VI.md) và [học đa nhân](MULTICORE_LEARNING_VI.md).

**Cập nhật 2026-10-04 — tăng batch/CPU có đo:** Người dùng yêu cầu script tăng batch/core để tận dụng GPU. Job mới xin8CPU/64GB/1GPU, thử4/8/16/32 với2/4worker và CUDA prefetch. Probe fullACT3+20batch trong subprocess riêng, OOM/timeout bỏcặp; chọnsamples/s cópeakreserved≤85%VRAM, ưu tiêncặpnhỏhơntrong3%. Probeweights khônglưu; fork từcheckpointgốc sangoutputs/act_pretrain_fast, giữAdamW/RNG/epoch/globalstep vàLR, tổng40epoch. Controller mới tiếp quảnwatcherworkers2/prefetch quaUID/script vàlock, chờepoch đãlưu rồiđổijob, khônghủyT27. CPUfixtures kiểmscope/boundary/selection; chưabenchmarkGPU. Xem [hướng dẫn](FAST_GPU_TRAIN_VI.md).

**Cập nhật 2026-10-04 — CUDA prefetch tùy chọn:** Người dùng yêu cầu che thời gian CPU → GPU. Đo server batch4 pinned84,38MiB chuyển4,45ms so bước244ms; không coi H2D là nút chính. Thêm `cuda_prefetch` mặc địnhfalse, copy batch kế trên stream riêng với lifetime/record_stream/wait_stream đúng thứ tự, chặn smoke trước lookahead. Nhánh output riêng giữ batch4/workers2 và checkpoint state; watcher chuyển sau epoch đã lưu. CPU fixtures kiểm migration/scope/serial; ba CUDA fixtures phải chạy trong job mới trước khi cài code. Chưa xác minh overlap/throughput trên GPU; không cam kết hết khoảng nghỉ. Xem [hướng dẫn](CUDA_PREFETCH_VI.md).

**Cập nhật 2026-10-04 — dashboard terminal:** Sau khi `a100top` không có trên server, chuẩn bị bản riêng `a100top_cfg5.py` dùng stdlib/ANSI. Dashboard chỉ đọc queue, log/history, PID CPU và GPU UUID; srun step dùng mpi=none trong job ACT đang chạy, không gửi/hủy job hoặc thay training. Phân biệt TIME_LEFT với ETA phase, RSS main với job RAM, epoch đang chạy với history đã lưu. Kiểm6 fixture và CLI help; chưa mở live dashboard trên server. Xem [hướng dẫn](A100TOP_CFG5_VI.md).

**Cập nhật 2026-10-04 — giảm chờ dữ liệu:** Sau phép đo server batch 4 (GPU 0–93%, CPU main khoảng 1–1,2 lõi, VRAM khoảng 10,2 GiB), người dùng yêu cầu giảm GPU chờ. Thêm lượt workers2 với prefetch2/worker, persistent workers, spawn và timing host cho loader/step/throughput. Chuyển ở checkpoint boundary bằng watcher scope đúng job/user/name; cài code có checksum trong compute job mới, giữ parent output, fork sang `outputs/act_pretrain_bs4_w2`, tổng 40 epoch. Giữ model/LR/precision/split/stats. Fixture kiểm worker thực tế đọc đủ mẫu qua 2 epoch và quy trình cancel/submit/install; chưa benchmark workers2 full trên GPU, không cam kết hết khoảng chờ. Xem [hướng dẫn](WORKERS2_CONTINUE_VI.md).

**Cập nhật 2026-10-04 — tăng batch trên server:** Người dùng yêu cầu batch size 4 trên A100 80 GB và chạy tiếp đến tổng 40 epoch. Đã thêm `fork_training_run.py` để tạo lượt mới từ `latest.pt`, giữ model/AdamW/RNG/epoch/global_step và best/history; chỉ cho thay batch_size, epochs, output_dir, còn contract và resume thường giữ kiểm tra nghiêm. Output mới `outputs/act_pretrain_bs4`; không đổi LR, workers, precision hoặc dữ liệu. Launcher Slurm kiểm job ACT đang chạy/chờ và giữ backup code khi cài. Fixture CPU kiểm giữ optimizer state và resume batch 4; chưa benchmark hoặc chạy full batch 4 trên GPU. Xem [hướng dẫn](BATCH4_CONTINUE_VI.md). Kết quả server người dùng cung cấp trước đó xác nhận full training batch 1 đã chạy; các mục ghi “chưa train” bên dưới là trạng thái lịch sử.

**Cập nhật 2026-09-30 — trainer pretraining:** Đã thêm trainer riêng dùng `ACTDataset`/`RH20TACTPolicy`, validation weighted theo target thật, `best.pt`/`latest.pt` và resume ranh giới epoch. Config mới `rh20t_all8.json` trỏ toàn CFG5. Đã kiểm bằng fixture và một optimizer step với ACT thu nhỏ; chưa chạy optimizer step của full ACT với CFG5 trên server GPU, chưa pretrain, chưa có rollout ACT. Server phải preprocess lại train+val và fit normalization train-only theo [hướng dẫn](PRETRAINING_VI.md). Tránh dùng `third_party/rh20t_act/train.py` nguyên xi vì nó dùng dataset/1 camera và split riêng upstream.

**Cập nhật 2026-09-29:** Bản đưa lên GitHub chỉ giữ code/metadata nhỏ. Processed train+validation đã được xóa ở máy phát triển sau khi kiểm; server sẽ preprocess lại từ raw cung cấp riêng. Các đoạn dưới đây mô tả thời điểm kiểm trước khi dọn dữ liệu.

**Cập nhật 2026-09-28 — toàn CFG5:** Theo yêu cầu mới, đã phân loại cả 2.515 episode/104 task, không dừng ở subset5 task. Chuyển nguyên folder episode bằng rename cùng ổ sang `RH20T_cfg5/training_data/task_XXXX/` (663 robot đạt ACT) hoặc `missing_data/task_XXXX/robot_incomplete/` (658 robot) và `missing_data/task_XXXX/human_for_future_one_shot/` (1.194 human). Không copy payload; số file/byte và directory identity đã kiểm. Human có chỉ mục ghép theo mã tên ở `RH20T_cfg5/human_robot_index.csv` và JSON ngoài raw; đây **không** là cặp đồng bộ hay nhãn action human. Paper báo cáo ACT dùng robot trajectories, trong khi human–robot hierarchy là hướng cho one-shot. Xem [báo cáo lọc toàn bộ](CFG5_FULL_FILTER_VI.md). Config `cfg5_all8.json`/manifest `cfg5_all_manifest.json` là phạm vi hiện hành cho bước chuẩn bị pretraining; split57 cũ chỉ là thí nghiệm lịch sử. Chưa full preprocess, chưa train.

**Cập nhật split toàn CFG5:** theo yêu cầu tỉ lệ chuẩn/ngẫu nhiên, dùng seed42 và nhóm `(task,user,calib)`, không ghim task. Thuật toán xáo group rồi lấy một phân bổ khả thi bằng DP để đạt đúng số episode nguyên gần80/10/10: **531 train /66 val /66 test**. Artifact [cfg5_all_splits.json](../data/splits/cfg5_all_splits.json), hash `aa8318240bba9d072b95a011e60a29b05322001d12a5977dd34ab4330f573825`. Nhóm5 task liên quan có42/12/3; task14 không có train do kết quả chia ngẫu nhiên theo group, không chọn lại seed theo kết quả test. Human liên hệ với robot đạt ACT thừa hưởng split của robot trong CSV index. Chưa preprocess full; người dùng tự chạy theo [hướng dẫn mới](CFG5_FULL_SPLIT_AND_RUN_VI.md). 10 Hz theo tần số danh nghĩa và thí nghiệm ACT của paper, nhưng lưới100 ms/giới hạn tuổi150 ms là adapter; ảnh căn với measured TCP/gripper theo timestamp, **không** chứng minh đồng bộ lệnh motor.

**Cập nhật sau khi người dùng preprocess:** `data/processed/cfg5_all/` có đủ531 train+66 val episode (không có test/partial), lưu flat để `ACTDataset` tra split qua JSON; không copy/move400GB thành hai folder. Kiểm4.776 header NPY và metadata đúng,65.427 bước train/6.968 bước val, không lỗi. `data/splits/cfg5_all_normalization.json` được tính **chỉ từ train**:65.427 state và58.313 action thật, không thiếu episode và không dùng smoke. `check_dataset` đọc3 sample mỗi split:58.313 sample train,6.214 sample val; đây là kiểm adapter, **chưa chạy ACT training hoặc đánh giá model**. Audit [cfg5_all_processed_postrun.json](audit/cfg5_all_processed_postrun.json), [train](audit/cfg5_all_train_check.json), [val](audit/cfg5_all_val_check.json).

Cập nhật 2026-09-27: **RH20T dùng để pretraining; mục tiêu chính là ACT điều khiển Panda gắp và đặt vật đúng vào khuôn nhỏ.** Người dùng giao quyền chọn phương án phù hợp CFG5 và yêu cầu 8 camera trong mô phỏng. Đề xuất camera-frame ở lượt 1 được thay bằng quyết định dưới đây; không còn chờ chọn representation.

## Phương án đã chọn

Rà soát placement sau lượt3: [PLACEMENT_LOGIC_REVIEW_VI.md](PLACEMENT_LOGIC_REVIEW_VI.md). Config hộp cũ chỉ còn alias; schema2 đo **độ nhô target=L−depth**, đồng thời kiểm nghiêng, phần cắm trong lỗ, đáy đỡ thật và released. Task hiện tại vẫn35−12=23mm; không thay bằng tolerance0.5mm.

| Thành phần | Quyết định và căn cứ |
|---|---|
| Upstream ACT | Lõi `rh20t/act_baseline`, commit `107ce1287a46be3a1a7ea954f8bbdec3bb103288`; adapter riêng. Source chạy đã tải đầy đủ vào third_party/rh20t_act; adapter ở src/models/rh20t_act.py. Full inference và backward nhỏ đã kiểm, chưa training. |
| State/action | 8D: TCP **aligned robot base frame**, xyz theo mét, quaternion **wxyz**, độ mở gripper theo mét. Một action dùng chung cho cả 8 góc nhìn. |
| Tại sao TCP | Phù hợp pose đã căn chỉnh của RH20T và mục tiêu đặt vật; không phụ thuộc thứ tự joint upstream. MuJoCo vẫn cần controller/IK và chuyển aligned TCP sang site thực tế. |
| Target | Pose đo được ở bước tương lai + độ mở gripper command tương lai. Arm dùng **measured-state proxy**, không phải raw expert command. |
| Pretraining | Task 4, 8, 13, 14, 17: di chuyển/đặt block, thao tác trên cân, đặt bút vào ống. Subset liên quan để kiểm chứng transfer, không gọi là toàn RH20T. |
| Camera | 8 RGB serial trong config; 6 external cố định, 2 in-hand theo RH20T. Rig lấy calibration `1640071679624`. |
| Đồng bộ/chunk | 10 Hz; sample-and-hold causal, tuổi ảnh/state ≤150 ms; cắt segment tại gap. K=20, target bắt đầu t+1, mask theo target thật. |
| Split | Group `(task,user,calib)`, tỷ lệ mục tiêu 80/10/10, seed 42. Không phải user-heldout hay calibration-heldout toàn cục. |
| Normalization | Train-only cho state/action; ảnh ImageNet normalization đúng một lần. Smoke statistics không dùng cho thí nghiệm chính thức. |
| Task đích | Theo yêu cầu lượt3: **3 chốt trụ Ø6×35 mm**,3 lỗ Ø7 mm sâu12 mm, đúng vị trí theo màu; thay đề xuất khối hộp lượt2. Config chính `panda_three_pegs.json`. Đã kiểm vật lý/controller. |
| So sánh chính | ACT học task đích từ đầu so với pretrain CFG5 rồi fine-tune; cùng demonstrations, split, seeds và ngân sách fine-tuning. |

CFG5 không thay thế dữ liệu task đích. Khác biệt mục tiêu, vật thể, gripper và miền hình ảnh khiến lợi ích transfer phải được kiểm chứng bằng rollout. Controller phải đối chiếu width với giới hạn gripper Panda; không gửi trực tiếp vector 8D vào actuator.

## Kết quả lượt 2

Trong 99 robot episode thuộc 5 task, 57 qua bộ lọc, 42 bị loại: 39 thiếu camera bắt buộc, 2 rating dưới ngưỡng, 1 không đủ khoảng giao thời gian. Đây là lý do đầu tiên phát hiện mỗi episode. Human episode không dùng trong adapter này.

Split: **47 train / 5 val / 5 test**. Sau lọc: 5.473 bước, 684 segment, 4.789 observation có future target. Median segment chỉ 6 bước; 275 sample có đủ 20 future targets. Yêu cầu đủ 8 camera làm giảm dữ liệu đáng kể. Có thể mở thêm task hoặc nghiên cứu missing-view masking sau; không tự chèn ảnh giả hay nối gap.

Đã chạy 8 unit tests; decode 8 bước của 1 train và 1 val episode; kiểm tra adapter với train-only statistics. Rig qua 8 roundtrip và 12 đối chiếu TCP camera cố định trên hai calibration của train. Chưa chạy model, training, test loss hoặc render MuJoCo.

## Phân kỳ

**Lượt3 đã hoàn tất:** Panda Menagerie Apache-2.0, MuJoCo3.14.0, PandaEnv API, controller IK, khuôn có cavity thật,8 policy camera +2 debug camera, evaluator100Hz. 14 unit tests đạt;3/3 seed controller smoke đặt đủ3 vật. Xem [STAGE3_REPORT_VI.md](STAGE3_REPORT_VI.md). TCP ảo tại grasp center Menagerie là quy ước mô phỏng được ghi rõ, chưa đo xác nhận factory tool transform của CFG5; không dùng demo làm bằng chứng ACT/sim-to-real.

Theo mục 10 [yêu cầu gốc](../references/requests/PROJECT_INSTRUCTIONS_VI.md): “Không triển khai tất cả trong một lượt”.

1. Lượt 1 đã xong: paper, nguồn ACT, inventory và validation.
2. Lượt 2 đã xong: adapter, preprocessing, split, smoke test, action contract và camera rig số học.
3. Lượt 3 đã xong: model Panda/license, MuJoCo với8 camera, controller,3 chốt/khuôn, evaluator, viewer/headless, demo và kiểm thử.
4. Lượt4 theo thứ tự người dùng chốt2026-09-28: tích hợp ACT trước (đã có model + smoke), rồi người dùng chọn/preprocess CFG5, sau đó trainer/pretraining. Demonstrations khuôn/fine-tuning và rollout checkpoint làm sau. Training dài do người dùng chạy.

Không cần xác nhận thêm cho các lựa chọn đã được giao. Raw giữ tại `RH20T_cfg5/`; PDF mới bỏ vào `references/papers/`, ghi chú vào `references/notes/`. Project chưa phải Git repository; upstream không sửa.

## Lượt4a — ACT trước dữ liệu full

Theo yêu cầu2026-09-28, chưa thu demonstrations khuôn. Đã tải26 file source đúng commit, giữ nguyên; adapter8 camera thêm camera identity positional embedding, explicit device/no implicit download, loss valid-only. Model83.873.929 parameters full inference trên một sample train smoke; backward nhỏ, optimizer_steps=0. Backbone/ACT đều chưa nạp pretrained weights. CPU Torch dùng để kiểm tích hợp, không quyết định GPU training. Hướng dẫn tự chọn/preprocess/mở dữ liệu ở [CFG5_HANDS_ON_VI](CFG5_HANDS_ON_VI.md); kiến trúc và giới hạn ở [ACT_SETUP_VI](ACT_SETUP_VI.md).

## 2026-10-08 — tải dữ liệu task đích từ Drive

Theo yêu cầu người dùng, thêm `scripts/download_drive.py` và
[DRIVE_DOWNLOAD_VI.md](DRIVE_DOWNLOAD_VI.md), gói độc lập `drive_download_bundle.zip`.
Dùng remote rclone có sẵn `mydrive:`, tải trực tiếp vào `~/cfg5-pretrain/downloads/finetune_data`
với 4 file đồng thời; không đồng nghĩa chiếm 4 lõi CPU. Chạy nền trên Linux, lock
theo đích, bảo vệ PID, retry, checksum và check nguồn/đích trước trạng thái complete.
Resume ở mức file đã hoàn tất; file đang dở có thể tải lại từ đầu. Không đổi raw
CFG5/trainer hoặc xác nhận schema dữ liệu task đích. 10 fixtures đạt trên WSL Linux,
gồm lifecycle với provider giả 24 file nhỏ; chưa tải dataset thật hoặc benchmark cụm.

## 2026-10-08 — demo task đích từ hai nguồn, fine-tune riêng

Người dùng cung cấp `pick_place_000042` và `pick_place_000487` ở local để khảo sát
hai nguồn server `downloads/finetune_data` và `downloads/dataset_moi`. Metadata và
NPZ xác nhận MuJoCo controller demo success, state measured, action commanded,
pre-step cùng tick; T=497/503, 8 RGB 640×360, 10 Hz, xyz+wxyz+gripper mét. Cảnh 42 fixed,
487 rich. Giữ payload nguyên vẹn; adapter dùng numeric NPZ, đối chiếu timestamp
8 video, không đưa privileged object pose/stages vào input policy.

Thêm `src/data/sim_demos.py`, `scripts/prepare_finetune_data.py`, hướng dẫn và 4 sbatch.
Manifest tham chiếu cả hai raw root; tên trùng cần nội dung giống nhau (gồm video)
để bỏ bản trùng; tên trùng nhưng nội dung khác thì dừng. Group nối theo seed hoặc
trajectory giống nhau; phân bổ theo variation profile, seed 42, mục tiêu 80/10/10.
Split trước chunks và khóa hash; chưa tạo split thật từ chỉ hai mẫu. Mặc định demo
lỗi làm audit dừng để đọc lý do.

ACTDataset thêm action_target_offset có mặc định 1 giữ CFG5 cũ, demo mới offset 0:
obs(t)→command(t:t+20), padding và gap mask. Loader/data contract hỗ trợ representation
demo riêng. Khởi tạo `--init-checkpoint` từ best pretrain, strict architecture/camera
và numeric normalization; optimizer/epoch mới. Resume vẫn yêu cầu contract của
chính run fine-tune. Kế thừa CFG5 train stats từ metadata best epoch 58, ghi provenance;
không fit val/test hoặc đổi mean/std rồi giữ nguyên head. Không sửa source upstream.

Preprocessing memmap uint8 từng episode, process CPU, kiểm disk free +10% +1 GiB;
~2.56/2.59 GiB ảnh cho mỗi mẫu thật. Checkpoint/source hash best 58 là
7567df2b01291fbd8f096cce5693bde542fbfc9709b9d5b2d7f18d722863ac9e.
Đã nạp strict model thật 83.873.929 params và decode 4 tick/mẫu; 29 tests đạt, kiểm
alignment/padding/gap, group/profile leakage, duplicates, timestamp, safe pickle,
raw write guard, fresh optimizer/resume và CLI 2 process chuẩn bị/check/resume.
Bộ cài kiểm checksum/phiên bản trước khi viết, sao lưu file cũ; đã kiểm fixtures
refuse custom version/corruption/path escape và cài lại. Cú pháp 4 bash sbatch đạt.
Chưa full audit/preprocess server, chưa fine-tune GPU, chưa đánh giá task success.

### Sửa split sau audit server — fixed chỉ có một nhóm

Job14094 báo 913 episode qua audit, 111 bị loại (24 demo thất bại, 87 lỗi/thiếu
video theo điều kiện kiểm); đã xác nhận một số color.mp4 thiếu ở finetune_data.
Người dùng chọn tiếp tục với exclude-invalid. Job14095 dừng khi stratum fixed
chỉ có một connected group seed/trajectory; chưa ghi artifact split/config.

Sửa split giữ toàn bộ profile ít hơn 3 group ở train. Phần còn lại phải có >=2
group độc lập cho val/test. Ưu tiên exact counts mục tiêu toàn bộ dataset; nếu
không khả thi, phân bổ gần tỷ lệ và báo actual counts, không chia nhỏ group.
Ghi group_summary, train_only_profiles, assignment_notes, requested_episode_counts.
Val/test không đại diện cho fixed độc lập khi fixed chỉ thuộc train. Không đổi
seed, bỏ group hoặc tách camera/frame để vượt kiểm leakage. Thêm 2 regression
fixtures: 913 episode với fixed200 dùng cùng trajectory và rich713 độc lập;
fixed dùng cùng seed cùng pool có group size2 khiến exact counts bất khả thi.
12 tests sim fine-tune đạt; chuẩn bị patch riêng để người dùng cài và chạy lại.

### Preprocessing 16 CPU / 128G — sau khi server khóa split thành công

Theo yêu cầu người dùng, bổ sung runner và hai Slurm job check/full CPU16, mem128G,
workers16. Giới hạn OpenCV/FFmpeg/BLAS một luồng mỗi process trước import; tối đa
16 episode đang chạy/đợi, ghi memmap từng episode và kiểm lại array/contract.
Preflight kiểm locked hashes, best SHA, raw fingerprint/timestamp/header/frame đầu,
output đã hoàn tất và disk free cho pending +10%+1GiB. Không đọc toàn bộ video trong
preflight; full decode kiểm từng frame. Quota riêng vẫn cần đọc bằng quota/admin.
Không dùng GPU8GB trong CPU decoder; không đổi 10Hz, ảnh 8x640x360 hay split/stats.
Log throughput theo tick/view/byte ảnh logic, elapsed và ETA. Khóa file để không
chạy hai job check/full mới cùng output. Episode partial không tự xóa/tiếp từ frame.
Bộ cài chỉ thêm file, guard checksum/prerequisite, idempotent và từ chối local edit.
2 tests video fixture 2 spawned process đạt; installer fixtures add/idempotent/
refuse custom/prerequisite và bash -n hai sbatch đạt. Chưa benchmark full16 trên
cluster; tốc độ có thể bị giới hạn bởi NFS/ổ đĩa. Bundle finetune_preprocess16_bundle.zip
được chuẩn bị để người dùng upload, kiểm storage rồi đăng ký job trên server.

### Kiểm cấp phát Slurm — job14104 và lỗi phân giải tên node

Job thử xin GPU1/CPU16/mem128G được bố trí trên dgx02; log có NumCPUs=16,
SLURM_JOB_CPUS_PER_NODE=16, MinMemoryNode=128G, SLURM_MEM_PER_NODE=131072MiB,
GPU index5. Job srun step bị lỗi getaddrinfo dgx02; chưa đọc được step affinity
hoặc cgroup memory. Sửa check_job_allocation.sbatch chạy Python ngay trong batch
đã được Slurm đặt trên compute node, không tạo srun step. Ghi đúng BATCH_AFFINITY,
có thể rộng hơn AllocCPUS nếu cluster không bind batch; không suy ra số CPU cấp
từ affinity rộng. Không sửa DNS/slurm.conf trên server. Cấp phát16 đã được log
xác nhận; enforced memory limit sẽ chỉ kết luận sau khi đọc cgroup trong job sửa.

### Fine-tune MP4 + bounded episode cache — 2026-10-08

Job14105 COMPLETED0:0, direct batch affinity16logicalCPU trên8physicalcore với SMT2;
Slurm nhận GPU1/CPU16/mem128G. Cgroup memory đọc được unlimited sentinel nên không
kết luận enforced128G hoặc RAM vô hạn. Tài liệu cluster do user gửi xác nhận home
soft1.8TB/hard2TB. Phương án full decoded NPY train/val ~2.05TiB không phù hợp quota.

Theo user yêu cầu, thêm MP4 backend giữ nguyên locked split/normalization và model
pretrain best58. Metadata chỉ lưu state/action/times/segments + source reference,
không ghi ảnh. Iterable prebatched stream phân episode riêng cho worker; decode
8MP4 tuần tự một lần vào RGBuint8 cache mỗi epoch, shuffle observation trong episode,
same-tick command20target/gap/padding theo ACTDataset. Batch tối đa8, đuôi episode
giữ lại không drop/duplicate; batch count khác ceil(total/8). Giới hạn cache3.5GiB
mỗi worker, bỏ cache trước episode mới và khi kết thúc iterator. Train16worker,
val2worker, prefetch1, decoder/BLAS1thread, CUDAprefetch giữ từ trainer hiện hữu.

Epoch shuffle generator tách worker base-seed generator: persistent worker và
loader mới sau resume có số draw base seed khác nhau; không để chúng đổi RNG đã
checkpoint. Spawn2worker kiểm resume epoch2/3 cùng thứ tự và state RNG. Backend
mặc định vẫnNPY, runconfig MP4 mới có signature riêng, output riêng và freshoptimizer.
Prepare/check chỉ trainval; không chuẩn bị/đọc ảnh test. InitSHA/stats provenance
và contract cũ được kiểm trước train; runtime không đổi mean/std/camera/10Hz.

Đã chạy33 CPU fixture tests:30đạt,3CUDA-only bỏ qua; có ACT reduced backward thật,
MP4/NPY equality, fail decode/raw changed/numeric corruption/cache cap, exact
worker ownership/sample coverage, locked prepare/check/resume2process, installer
guards, NPY trainer/checkpoint regression. bash-n hai Slurm jobs đạt. Bộ cài kiểm
baseline/chksum và sao lưu2corefiles, không sửaupstream/raw/split/checkpointcũ.
Không mirrorgithub_source/pushGitHub trong lượt này theo user muốntraintrước.
SSH BatchMode server172.16.3.8 bị Permission denied do thiếuxácthực; chưa upload
hoặc chạyserver. Bundle finetune_mp4_bundle.zip sẵnđểscp, jobGPUcheck1smokebatch
train+val vàbenchmark16worker; jobfull chỉ chạyafterok check, không2jobđồngthời.
Fullserverthroughput/RAM/NFS và20epoch chưa kiểm; không tuyênbốGPUluôn100%.

### ETA từ log/history — 2026-10-09

Theo yêu cầu tính thời gian xong, thêm script read-only estimate_mp4_training_time.py,
không importmodel hay thayconfig/job. Đọc20windowtốc độ gần nhất từngphase, weighted
theo samples/time; tươnglai fullphase dùngmean3epochgầnnhất, phầnepochhiện tại dùng
remaining samples/rate. Cộng riêngtrainval, resetprogress theo epochJSON/headerresume,
không tínhlại train nếu đangval. Thiếusốđoval không bịa ETA toànrun; chỉđưa trainETA.
InUTC+7dựkiếnxong, walltimecòn từsqueue; không đồngnhất72htổng với72hcònlại.
Chưa cộngsavecheckpoint/xếphàng/rerun epoch interrupted.5purestdlibtestsđạt.
Scriptchỉ thêm vào local_source, useruploadriêngkhi jobđangtrain; chưaSSHupload.

### Tối ưu MP4: batch lớn, uint8 H2D và thread prefetch — 2026-10-09

Theo ảnh/log job14111: VRAM19.4/80GB, RAM57.9/128GB, CPU459%/16logical;
data_wait0.5–175.2ms, step385.4–1499.8ms. Không kết luận decode là nút thắt
chính từ GPU0% snapshot. Step gồm CPU dispatch/compute/H2D/scalar sync.

Thêm runtime opt-in image_transfer_dtype=uint8 (chỉ MP4) và cuda_prefetch_threaded.
Giữ normalize mean/std khóa, normalize float32 trên GPU; fixture khớp CPU trong
1e-6, ảnh uint8 H2D giảm4lần số byte. Không scan finite trên uint8 (mọi giá trị
hữu hạn); state/action và ảnh float vẫn kiểm. Thread nền fetch+H2D/copy stream,
compute chờ event riêng batch hiện tại; giữ CPU refs tới event và record_stream
GPU tensors. Không sửa model/upstream, không đổi control10Hz, split hay stats.

Benchmark8case baseline8/16, fast8/16, fast16và32 với8/16/24worker, cùng train
seed/checkpoint. Ít nhất một đợt đổi episode trước kết thúc cửa sổ đo. Chỉ đo
optimizer tạm, không lưu weights benchmark, không dùng test. Resident batch đo
sau dừngworkers để phân biệt pipeline với chi phí step; không gọi là kernel-only.
Chọn samples/s cao nhất trongbatch16/32+worker24 theo yêu cầu tăngcảhai, còn20%VRAM
dự phòng; reportghi thêm best_any; không cải thiệnso baseline thì dừng,
không submit train chạy với cấu hình thua. Worker24 vẫn chỉ có16CPUlogical;
cache24max thực~62.4GiB cộng RAM khác, cần benchmark thật.

Fork mở các runtime flags, giữ initialization best58 và optimizer/RNG/epoch,
output mới và strictsignature mới. Watcher chỉ đọc JSON/Slurm trênheadnode, chờ
epochfull+latest/history rồi cancel đúngjob/user, install với backup/baseline
guard, submitbenchmark vàtrain afterok. Chưa đổi job14111 hoặc uploadserver.

27tests local:22đạt,5CUDA-onlyskip; bổ sung kiểm fork runtime/provenance2testsđạt.
Real bundle installplan từ baselinezip/idempotent đạt; Bash/Python syntax đạt.
GPUgate gồm serial/threadoptimizer equality và2spawnworker phải qua trênserver.
GPU memory/throughput/1–3hbenchmark chưa đo, không hứa speedup hoặc batch32fit.
Zip finetune_mp4_fast_bundle.zip và docs/MP4_FAST_VI.md chuẩn bị để userupload.

### 2026-10-09 — chuyển MCR warmup sang dữ liệu mới, dọn ảnh CFG5 có kiểm tra

Người dùng sửa hướng: không tạo feature cache/warmup lại CFG5; lấy ACT E5 và
ResNet50 MCR để frozen warmup rồi full fine-tune trên dữ liệu gắp thả đã split.
Pilot mới job14129 thành công, 407776 train+val ticks, feature payload317.33GiB,
estimated pending462.17GiB (gồm JPEG+dự phòng). User gửi build mới job14130.
Các job CFG5 cũ14127/14128 được yêu cầu huỷ; chưa tự xác nhận chúng đã dừng.

Theo yêu cầu dọn khoảng400GB, thêm cleanup_cfg5_images.py. Mặc định audit,
--apply chỉ unlink images.npy thuộc531train+66val trong root CFG5 cũ theo locked
membership, không recursive delete. Guard đường dẫn/symlink/header/metadata,
root mới disjoint, cache mới dùngMP4, checkpointE5 và Slurm job whitelist.
Giữ raw, test, numeric, split/stats/checkpoint và các file khác. Journal ghi rõ
từng file; ảnh cũ cần tái tạo nếu muốn quay lại NPY pipeline.6fixtures nhỏ đạt,
gồm realCLI audit/apply trên597trainval+66test giả; chưa xoá dữ liệu thật.
