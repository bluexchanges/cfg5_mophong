# Danh mục nguồn đã kiểm tra

Ngày khảo sát: 2026-09-27. Ưu tiên hai PDF do người dùng cung cấp; website/repository dùng để đối chiếu implementation và định dạng dữ liệu.

## Paper được cung cấp

1. **RH20T: A Comprehensive Robotic Dataset for Learning Diverse Skills in One-Shot**. Bản cục bộ: `papers/RH20T_paper_compressed (1).pdf`, 15 trang. Bảng II–III trang 3; thí nghiệm ACT mục IV trang 5–6; bảng IV–V trang 6; danh sách task ở phụ lục trang 9–15. [Trang dự án](https://rh20t.github.io/), [arXiv](https://arxiv.org/abs/2307.00595).
2. **Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware**. Bản cục bộ `papers/2304.13705v1.pdf`, 18 trang, arXiv v1. ACT ở mục IV trang 4–6; đánh giá trang 6–10; kiến trúc chi tiết ở phụ lục C. [Trang ACT/ALOHA](https://tonyzhaozh.github.io/aloha/), [arXiv v1](https://arxiv.org/abs/2304.13705v1).

SHA-256 của chính hai file đã đọc nằm trong `papers/manifest.json`. Không dùng bản arXiv mới hơn thay cho PDF được cung cấp.

## Mã nguồn đã khóa phiên bản để khảo sát

| Repository | Commit | Vai trò | License root |
|---|---|---|---|
| [tonyzhaozh/act](https://github.com/tonyzhaozh/act/tree/742c753c0d4a5d87076c8f69e5628c79a8cc5488) | `742c753c0d4a5d87076c8f69e5628c79a8cc5488` | Implementation ACT gốc và môi trường ALOHA | MIT |
| [AirExo/act-in-the-wild](https://github.com/AirExo/act-in-the-wild/tree/a11e8e69da317b9a64cc39a85b48bc2f397e0475) | `a11e8e69da317b9a64cc39a85b48bc2f397e0475` | ACT với dữ liệu AirExo, tăng mật độ sample | MIT |
| [rh20t/act_baseline](https://github.com/rh20t/act_baseline/tree/107ce1287a46be3a1a7ea954f8bbdec3bb103288) | `107ce1287a46be3a1a7ea954f8bbdec3bb103288` | Baseline ACT trực tiếp cho RH20T | MIT |
| [rh20t/rh20t_api](https://github.com/rh20t/rh20t_api/tree/aa3124434729ed622109a29b2cbb9f3bbb1c5eeb) | `aa3124434729ed622109a29b2cbb9f3bbb1c5eeb` | Schema, cấu hình robot và API dữ liệu | MIT |

Manifest chứa URL raw theo commit và SHA-256 của từng file. Giữ cả license và các attribution trong file, kể cả code DETR được kế thừa. Snapshot chưa được chỉnh sửa hoặc cài đặt.

## Dữ liệu

[Trang download và data format RH20T](https://rh20t.github.io/#download) phân biệt các gói RGB, depth, low-dimensional state và calibration; vì vậy có timestamps depth không chứng minh máy đã có video depth.

Theo [mục License của RH20T](https://rh20t.github.io/#License), scene 0001–0005 thuộc CC BY-SA 4.0, scene 0006–0010 thuộc CC BY-NC 4.0. License code MIT là một nội dung riêng. Dự án hiện phục vụ nghiên cứu/học tập; chưa phát hành dataset hoặc weights.

Không tải lại dataset, weights hoặc model asset trong lượt khảo sát này.

## Model mô phỏng — bổ sung lượt3

[MuJoCo Menagerie Panda](https://github.com/google-deepmind/mujoco_menagerie/tree/c96a32d28fb5da84da38c1da4d749e7a13212855/franka_emika_panda), commit `c96a32d28fb5da84da38c1da4d749e7a13212855`, Apache-2.0. Đã tải riêng71 file Panda vào `mujoco/models/franka_emika_panda/`; SHA-256 và URL từng file nằm trong manifest tại đó. Giữ nguyên upstream XML/license, scene dẫn xuất tạo ở `mujoco/scenes/`. Không tải checkpoint ACT.

Engine/API: [MuJoCo Python](https://mujoco.readthedocs.io/en/stable/python.html), [camera XML/intrinsics](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-camera), [mj_contactForce](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-contactforce). Phiên bản đã chạy3.14.0, dependency lock `requirements-sim-lock.txt`. Các nhận xét viewer, contact, projection trong báo cáo lượt3 có kiểm tra thực tế, không chỉ dựa vào tài liệu.

## ACT chạy thực tế — lượt4a

Bản source đầy đủ26 file cùng commit RH20T baseline đã tải vào third_party/rh20t_act, manifest riêng; MIT root và Apache-2.0 DETR giữ nguyên. Cặp torch2.7.1/torchvision0.22.1 CPU theo [bảng phiên bản PyTorch](https://pytorch.org/get-started/previous-versions/#v271). Adapter dự án thêm8-camera identity embedding và valid-only loss; không nạp pretrained ACT/backbone weights. Xem docs/ACT_SETUP_VI.md.
