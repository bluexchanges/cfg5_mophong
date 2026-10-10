# Dữ liệu và metadata

Bản mã nguồn để đưa lên GitHub **không chứa raw hoặc processed episode**. processed/ có README để giữ cấu trúc thư mục; sau khi preprocess trên server, output mới sẽ nằm trong processed/cfg5_all/<episode>/. Không commit các file NPY/ảnh/video.

splits/cfg5_all_manifest.json và splits/cfg5_all_splits.json là bản tham chiếu của lần xử lý trên máy phát triển: 663 robot episode đạt bộ lọc, được chia 531 train / 66 validation / 66 test theo nhóm (task,user,calib) và seed 42. splits/cfg5_all_normalization.json là thống kê **chỉ từ train** của lần đó; không dùng nếu tạo split mới có hash khác. Các file cfg5_* không có all và smoke_normalization.json thuộc thí nghiệm nhỏ trước đây.

index/cfg5_human_robot_pairs.json và index/human_robot_index.csv giữ liên hệ human–robot theo tên episode; không phải nhãn action theo frame. index/cfg5_all_split_membership.csv là bảng dễ đọc của lần xử lý 597 episode train/validation, không phải input model. Nguồn chuẩn cho split vẫn là JSON.

Một processed episode gồm images.npy dạng RGB uint8 (T,8,360,640,3), state.npy và action.npy 8D, timestamps_ms.npy, segment_ids.npy, frame_indices.npy, camera_timestamps_ms.npy, state_timestamps_ms.npy và episode.json. Loader chỉ tạo action chunk trong lúc đọc sample, không lưu lặp từng chunk. Xem [hướng dẫn dữ liệu](../docs/DATA_GUIDE_VI.md) và [chạy lại trên server](../docs/SERVER_REPROCESS_VI.md).
