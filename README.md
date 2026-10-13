# CFG5 → ACT / MCR → Panda trong MuJoCo

Repository lưu code, cấu hình, tài liệu và các gói triển khai của dự án robot learning trên CFG5, với ACT, backbone ResNet/MCR và Franka Panda trong MuJoCo.

## Bắt đầu

- [Hướng dẫn dự án chính](local_source/README.md).
- [Các quyết định kỹ thuật và cập nhật theo thời gian](local_source/docs/DECISIONS.md).
- [Rule cho AI agent](AGENTS.md) và [yêu cầu dự án gốc](local_source/references/requests/PROJECT_INSTRUCTIONS_VI.md).
- [Sơ đồ AI có giải thích](https://cfg5-act-ai-flow.icy-cat-0191.chatgpt.site/) hoặc [bản HTML trong repo](ai-flow-site/dist/index.html).
- [Báo cáo đối chiếu và đồng bộ GitHub ngày 2026-10-13](docs/GITHUB_SYNC_20261013_VI.md).

Một số README/báo cáo trong source mô tả trạng thái ở các mốc cũ. Đọc quyết định, config, checkpoint metadata và báo cáo thực tế của run cần làm việc; không xem trạng thái lịch sử là kết luận về mọi phiên bản hiện tại.

## Các thư mục chính

| Vị trí | Nội dung |
| --- | --- |
| `local_source/` | Code/config/docs/tests làm việc chính; dữ liệu raw và payload lớn nằm ngoài bản GitHub. |
| `github_source/` | Snapshot source công khai từ giai đoạn trước. |
| `cfg5_masked_mcr_bundle*` | Gói bổ sung CFG5 missing-camera mask và staged MCR cache; từng phiên bản có manifest riêng. |
| `cfg5_mcr_continue_bundle/` | Gói tiếp tục run MCR. |
| `cfg5_mcr_three_splits_dropout_bundle/` | Gói ba split và dropout, kèm installer, docs và tests. |
| `epoch5_mcr_bundle*`, `epoch5_transfer_probe_14120/` | Các gói và metadata liên quan transfer từ ACT epoch 5. |
| `finetune_*_bundle/`, `finetune_split_fix/` | Chuẩn bị và tối ưu fine-tuning pick-place. |
| `drive_download_bundle/` | Công cụ lấy dữ liệu từ Drive. |
| `checkpoints/` | Checkpoint đã được chọn dùng Git LFS và history/metadata liên quan. |
| `local_source/outputs/` | Các báo cáo/ảnh preview nhỏ được chọn; không chứa payload numeric hay video trong Git. |
| `outputs/mcr_dropout_upgrade_base/` | Snapshot source baseline để kiểm tra nâng cấp dropout, không phải run training mới. |
| `ai-flow-site/` | Source HTML của trang giải thích AI; không cần submodule khi clone GitHub. |
| `server_pretrain_*.py` | Snapshot trainer server để so sánh, không thay thế tự động trainer trong source chính. |

Không áp dụng tất cả bundle liên tiếp một cách mặc định. Đọc prerequisites, manifest và hướng dẫn của bundle phù hợp với source/run đang dùng.

## Clone và weights

```bash
git clone https://github.com/bluexchanges/cfg5_mophong.git
cd cfg5_mophong
git lfs pull
```

Git LFS cần được cài trước khi lấy các weights đã được chọn. Không tự chạy training sau khi clone; bắt đầu từ hướng dẫn của pipeline/run cần sử dụng.

`local_source_(1)/` được bỏ qua theo yêu cầu của người dùng. Môi trường ảo, raw dataset, cache payload, video, archive bundle và các clone repo cũ không thuộc bản đồng bộ source. Rule gốc edabk được giữ riêng trong [tư liệu rule AI](docs/reference_agent_rules/README.md).
