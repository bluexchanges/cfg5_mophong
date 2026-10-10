# Source bên ngoài dùng trực tiếp

Khác với snapshot khảo sát trong `references/upstream`, source ở đây được adapter model import để chạy ACT. Không sửa source gốc; phần thích nghi nằm ở `src/models/rh20t_act.py`.

| File/thư mục | Vai trò |
|---|---|
| `rh20t_act/` | Toàn bộ26 file repository `rh20t/act_baseline` tại commit107ce1287a46be3a1a7ea954f8bbdec3bb103288, tổng103.547 byte. Không có weights. |
| `rh20t_act_manifest.json` | Commit, path, URL, kích thước, Git blob SHA và SHA-256 từng file. Downloader/model dùng kiểm toàn vẹn. |
| `rh20t_act/models/detr/models/` | Lõi DETRVAE, Transformer, backbone, positional encoding; adapter dùng trực tiếp các class/hàm cần thiết. |
| `rh20t_act/models/policy.py`, `models/detr/main.py` | Wrapper/factory upstream được giữ để tra cứu; adapter không gọi factory ép CUDA/argparse/download. |
| `rh20t_act/dataset/`, `train.py`, `command_pretrain.sh` | Pipeline gốc để đối chiếu, không phải pipeline CFG5 đã đổi của dự án. Không chạy trực tiếp để thay trainer chưa tích hợp. |
| `rh20t_act/LICENSE`, `rh20t_act/models/detr/LICENSE` | MIT root và Apache-2.0 DETR; giữ copyright/attribution nguồn. |
| `rh20t_act/README.md`, `requirements.txt`, `models/detr/setup.py`, `detr.egg-info/` | Tài liệu/metadata package gốc. File tồn tại không có nghĩa đã cài package hoặc requirements này tương thích môi trường hiện tại. |

Tải lại/kiểm file: `python scripts/download_act_baseline.py` bằng interpreter `.venv-data` từ root. Script không ghi đè file khác hash, không tải checkpoint. Cài dependency theo [ACT_SETUP_VI](../docs/ACT_SETUP_VI.md), không theo requirements upstream cũ.
