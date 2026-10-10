# Model robot và scene mô phỏng

`mujoco/` chứa mô tả vật lý/XML và mesh. Logic Python ở [src/simulation](../src/simulation/README.md); tham số task ở [configs/mujoco](../configs/mujoco/README.md).

| File/thư mục | Làm gì? | Nguồn và cách sửa |
|---|---|---|
| `models/franka_emika_panda/panda.xml` | Mô tả robot7 khớp, hai ngón, inertial/collision/actuator và tham chiếu mesh. | Bản upstream Menagerie; giữ nguyên. |
| `models/franka_emika_panda/assets/` | Các mesh hình học được Panda XML sử dụng. | Asset tải riêng, không commit; thiếu mesh có thể làm XML không load được. Từng tên/hash nằm trong manifest. |
| `models/franka_emika_panda/manifest.json` | Repository, commit, URL và SHA-256 các file tải. | Bằng chứng model đến từ đâu; dùng downloader để khôi phục đúng bản. |
| `models/franka_emika_panda/LICENSE` | License Apache-2.0 của model. | Giữ attribution; không suy license này cho dataset/PDF. |
| `models/franka_emika_panda/README.md` | Hướng dẫn gốc của model Panda. | Giữ nguyên tài liệu upstream. |
| `models/franka_emika_panda/CHANGELOG.md` | Lịch sử model tại snapshot đã tải. | Không phải changelog của code dự án. |
| `scenes/panda_three_pegs.xml` | Scene dẫn xuất: Panda, TCP policy, bàn,3 chốt có free joint, khuôn có lỗ rỗng,8 policy camera+2 debug camera. | File sinh tự động; sửa config/builder rồi regenerate. |
| `scenes/panda_three_pegs.manifest.json` | Hash config, rig, builder, XML nguồn và XML sinh ra, commit model. | PandaEnv đối chiếu để phát hiện scene cũ hoặc XML bị sửa. |

Nguồn model đã khóa commit `c96a32d28fb5da84da38c1da4d749e7a13212855`;71 file upstream đã tải, manifest cục bộ ghi chi tiết. Lỗ được tạo bằng các mảnh thành bao quanh khoảng rỗng và đế dưới; hình vẽ lỗ trên mặt không đủ để mô phỏng insertion.

Luồng: downloader cung cấp model → builder sao chép/dẫn xuất XML và dựng task → MuJoCo load XML → PandaEnv điều khiển. Model có position actuators; action8D phải qua controller/IK, không chép trực tiếp vào `data.ctrl`.

Từ project root:

```powershell
& '.\.venv-data\Scripts\python.exe' -X utf8 scripts/build_panda_scene.py
& '.\.venv-data\Scripts\python.exe' -X utf8 scripts/check_mujoco.py
```

Build/check có ghi lại scene/report/preview. Khi cần giữ kết quả một thí nghiệm, lưu config/hash/report trước khi đổi tham số. Không thêm README riêng vào từng asset hoặc sửa README upstream; file này giải thích phần tích hợp của dự án.
