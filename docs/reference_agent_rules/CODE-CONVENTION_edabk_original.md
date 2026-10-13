# Quy ước dự án / Project Conventions — edabk-physical-ai-robot

Áp dụng cho mọi thành viên. Khi quy ước mâu thuẫn với code cũ, code mới theo quy ước này; code cũ được sửa dần khi chạm tới.

## Mục lục

1. [Quy tắc chung đặt tên](#1-quy-tắc-chung-đặt-tên--general-naming-rules)
2. [Đặt tên thư mục và file](#2-đặt-tên-thư-mục-và-file--directory--file-naming)
3. [Tài liệu tham khảo](#3-tài-liệu-tham-khảo--references)
4. [Báo cáo, biên bản, log](#4-báo-cáo-biên-bản-log--reports)
5. [Công bố](#5-công-bố--publications)
6. [Coding style — Python](#6-coding-style--python)
7. [Coding style — C model](#7-coding-style--c-model)
8. [Coding style — Verilog / SystemVerilog](#8-coding-style--verilog--systemverilog)
9. [Testbench và golden vector](#9-testbench-và-golden-vector)
10. [Script flow (TCL, shell, Makefile)](#10-script-flow-tcl-shell-makefile)
11. [Git](#11-git)
12. [Tài liệu hành chính trong repo](#12-tài-liệu-hành-chính-trong-repo--administrative-documents)
13. [Quy trình thí nghiệm và merge](#13-quy-trình-thí-nghiệm-và-merge--experiment--merge-workflow)

---

## 1. Quy tắc chung đặt tên / General naming rules

| Quy tắc | Đúng | Sai |
|---|---|---|
| Chỉ dùng ASCII: `a–z A–Z 0–9 - _ .` | `spike_encoder.v` | `bộ_mã_hóa.v`, `file (1).v` |
| Không dấu cách, không ký tự `[ ] ( ) , & #` | `R12_2021_Mack_TCAD_RANC.pdf` | `[12] RANC.pdf` |
| Ngày theo ISO 8601 `YYYY-MM-DD` hoặc `YYYYMMDD` | `2026-10-05_weekly-minutes.md` | `5-10-26_minutes.md` |
| Phiên bản `v<major>.<minor>` hoặc `_v<n>` | `D2_report_v1.2.docx` | `report_final_final2.docx` |
| Không đặt tên `new`, `final`, `old`, `test2` | `top.v` + git tag | `top_new.v` |
| Bản cũ chuyển vào `_archive/`, không xóa | `_archive/top_v1.v` | — |

Phân cách: `_` ngăn cách các **trường** (ngày, loại, chủ đề); `-` nối các **từ** trong cùng một trường.

## 2. Đặt tên thư mục và file / Directory & file naming

| Loại | Quy ước | Ví dụ |
|---|---|---|
| Thư mục cấp 1 | `NN-PascalCase` | `04-Src`, `06-Reports` |
| Thư mục cấp dưới | `lower-kebab-case` hoặc `snake_case` theo tên khối RTL | `flow-rtl-sim`, `spike_encoder` |
| Python module | `snake_case.py` | `preprocess_ppg.py` |
| Notebook | `<stt>_<động-từ>_<đối-tượng>.ipynb` | `01_preprocess_ppg.ipynb` |
| C model | `<khối>.c/.h` | `snn_core.c` |
| RTL module | tên file = tên module, `snake_case.v/.sv` | `neuron_core_64x32.v` |
| Testbench | `tb_<module>.sv` hoặc `test_<module>.py` (cocotb) | `tb_spike_encoder.sv` |
| Constraint | `<top>_<mode>.sdc` | `soc_top_50mhz.sdc` |
| Test vector | `<khối>_<tín-hiệu>_<dataset>_<định-dạng>.txt` | `encoder_spike_mitbih_bin.txt` |
| Tài liệu thiết kế | `<khối>_<loại-tài-liệu>.md` | `snn_core_spec.md` |
| Hình | `<khối>_<nội-dung>.svg/.png` + nguồn `.drawio`/`.json` | `soc_dataflow.drawio` |

## 3. Tài liệu tham khảo / References

```
R<nn>_<Year>_<FirstAuthorSurname>_<VenueAbbrev>_<Short-Title>.pdf
P<nn>_<Year>_<Assignee>_<PatentNo>_<Short-Title>.pdf          (patent)
```

- `R<nn>` cấp tăng dần, không tái sử dụng; là khóa chính trong bảng survey (`02-References/`).
- `<VenueAbbrev>`: `JSSC`, `TCAS-I`, `TBioCAS`, `TCAD`, `ISSCC`, `VLSI`, `DAC`, `DATE`, `ISCAS`, `Access`, …
- Số liệu trong bảng survey ghi nguyên văn kèm đơn vị; ghi rõ **đo silicon** hay **mô phỏng post-synthesis/post-layout**, và technology node.
- Trích dẫn chuẩn **IEEE**; BibTeX key `<firstauthor><year><firstword>` chữ thường, ví dụ `mack2021ranc`.

## 4. Báo cáo, biên bản, log / Reports

| Loại | Vị trí | Quy ước |
|---|---|---|
| Biên bản họp tuần | `06-Reports/meeting-minutes/` | `YYYY-MM-DD_weekly-minutes.md` (theo `_TEMPLATE_weekly-minutes.md`) |
| Biên bản họp chuyên đề | như trên | `YYYY-MM-DD_<chủ-đề>-meeting.md` |
| Báo cáo deliverable | `06-Reports/deliverables/` | `D<n>_<Tên-deliverable>_v<x.y>_<YYYYMMDD>.docx` |
| Log huấn luyện | `06-Reports/experiment-logs/training/<config>/` | config, seed, split, metric, commit hash |
| Log mô phỏng | `06-Reports/experiment-logs/rtl-sim/` | `YYYYMMDD_<tb>_<dataset>.log` |
| Báo cáo PPA | `06-Reports/experiment-logs/synthesis/` | `YYYYMMDD_<top>_<tech>_<corner>_<freq>/` + `ppa_summary.xlsx` |
| Excel số liệu | cạnh log tương ứng | `<nội-dung>_summary.xlsx`, có sheet `Methodology` |

## 5. Công bố / Publications

```
07-Publish/<Year>_<VenueAbbrev>_<Short-Title>/
    latex/ | docx/     figures/     submitted/     review/     camera-ready/     published/
```

Patent: `07-Publish/<Year>_Patent-VN|US_<Short-Title>/`. Năm là năm nộp lần đầu; không đổi tên thư mục khi bài được chấp nhận.

## 6. Coding style — Python

- PEP 8, `black` (line length 100), `isort`.
- Không dùng đường dẫn tuyệt đối; tính từ gốc repo bằng `pathlib`:
  ```python
  from pathlib import Path
  PROJECT_ROOT = Path(__file__).resolve().parents[3]   # 04-Src/software/<sub>/file.py
  ```
- Tham số thí nghiệm để trong file YAML/JSON cấu hình; cố định seed và ghi seed vào log.
- Notebook chỉ để khám phá; logic dùng lại chuyển sang module `.py`; xóa output lớn trước commit (`nbstripout`).
- Lượng tử hóa: ghi rõ định dạng `Q<m>.<n>`, kiểu làm tròn, bão hòa hay tràn.

## 7. Coding style — C model

- C99, `stdint.h`; không dùng `float` trong đường dữ liệu bit-accurate.
- Mỗi khối RTL có một hàm C cùng tên, cùng thứ tự port; phép dịch/làm tròn/bão hòa viết bằng hàm riêng để khớp RTL.
- Biên dịch sạch với `-Wall -Wextra -Werror`.

## 8. Coding style — Verilog / SystemVerilog

| Hạng mục | Quy ước | Ví dụ |
|---|---|---|
| Module, file | `snake_case`, 1 module/file | `spike_mapper` |
| Port | hậu tố `_i` / `_o` | `data_i`, `valid_o` |
| Active-low | hậu tố `_n` | `rst_n` |
| Thanh ghi / next-state | `<tên>_q` / `<tên>_d` | `cnt_q`, `cnt_d` |
| Parameter | `UPPER_SNAKE_CASE` | `DATA_W` |
| Handshake | `valid` / `ready` | `in_valid_i`, `in_ready_o` |

- `always_ff` / `always_comb` (SV); không trộn `=` và `<=` trong một khối; không latch, không `initial`, không `#delay` trong RTL tổng hợp.
- Header mỗi file: module, mô tả, tác giả, ngày, định dạng fixed-point của port.
- SRAM/macro bọc trong wrapper để thay theo công nghệ.
- Lint sạch `verilator --lint-only -Wall` trước commit.

## 9. Testbench và golden vector

- Mỗi khối có test bit-accurate so với golden output của C/Python model.
- Đường dẫn vector truyền qua plusarg, không hard-code: `$value$plusargs("VEC_DIR=%s", vec_dir)`.
- Mỗi file vector kèm `.meta.json`: độ rộng bit, dấu, số mẫu, dataset, commit hash sinh ra.
- Dòng cuối kết quả cố định: `TEST <name> PASS mismatches=0 samples=<N>` hoặc `FAIL`.

## 10. Script flow (TCL, shell, Makefile)

- Mỗi `05-Experimental/flow-*/` có `README.md` và `Makefile` với target `run`, `report`, `clean`.
- Đường dẫn tương đối theo vị trí script; không dùng đường dẫn tuyệt đối của máy cá nhân hay Colab.
- Output ghi vào `run_<YYYYMMDD_HHMM>/` (gitignored); kết quả tổng hợp sao chép sang `06-Reports/experiment-logs/`.
- License server, password, token đặt trong biến môi trường, không đưa vào script.

## 11. Git

- Nhánh `main` (ổn định, được bảo vệ) và nhánh thành viên dài hạn `<ho-va-ten>-k<khoa>` (ví dụ `tran-thi-ha-k67`); commit `<scope>: <mô tả> (task #<số>)`. Quy trình đầy đủ kèm lệnh: README §6.
- Không commit trực tiếp lên `main`; mọi thay đổi vào `main` qua PR bàn giao từ nhánh thành viên, trưởng nhóm review, PI merge bằng merge commit (README §6.7).
- Pre-commit chạy `tools/check-repo.sh --staged`; CI chạy `--all`.
- File > 50 MB: Git LFS hoặc GitHub Release; > 100 MB bị GitHub chặn.
- Thư viện bên thứ ba (cocotb, …): cài bằng pip/package manager hoặc `git submodule` có `.gitmodules`; không commit repo lồng.
- Mã public bên ngoài (khung SoC như Caravel, IP mở, thư viện, mô hình upstream): khai báo trong `08-external/deps.tsv` (URL, commit 40 ký tự, giấy phép), kéo về bằng `bash 08-external/fetch.sh`; không commit mã kéo về. File sửa trên mã bên ngoài đặt trong `04-Src/` theo đúng đường dẫn tương đối (overlay) hoặc dạng patch; flow áp vào trước khi chạy.
- Không commit: PDK/IP có NDA, dữ liệu bệnh nhân chưa được phép phân phối, credential.

## 12. Tài liệu hành chính trong repo / Administrative documents

1. Bản đầy đủ lưu tại `_private/<loại>/` (không vào git).
2. Bản đưa vào git lược bỏ hẳn (không dùng Track Changes): kinh phí, dự toán, số điện thoại, ngày sinh, CCCD, số tài khoản.
3. `README.md` của thư mục ghi rõ phần đã lược bỏ và vị trí bản đầy đủ.
4. `tools/check-repo.sh --staged` phải PASS trước commit.

## 13. Quy trình thí nghiệm và merge / Experiment & merge workflow

Mọi thí nghiệm (mô phỏng RTL, synthesis, huấn luyện/đánh giá mô hình, benchmark) chạy bằng script; không thao tác tay không ghi lại.

### Thí nghiệm tự động / Automated experiments

1. Một lệnh tái lập toàn bộ thí nghiệm từ repo sạch: `make run` trong `05-Experimental/flow-*/` (§10) hoặc `bash run_experiment.sh configs/<config>.yaml` (§6).
2. Script ghi vào log: config, seed, commit hash (`git rev-parse --short HEAD`), thời điểm chạy.
3. Output thô ghi vào `run_<YYYYMMDD_HHMM>/` (gitignored, không commit); kết quả tổng hợp kèm đơn vị sao chép sang `06-Reports/experiment-logs/` (§4).
4. Chưa có script cho thí nghiệm cần chạy: viết script + file cấu hình, commit, rồi mới chạy. Không chạy lệnh ad-hoc rồi báo kết quả không tái lập được.

### Trình tự làm việc / Working sequence

Theo README §6 (kèm lệnh và ví dụ):

1. **Nhánh**: mỗi thành viên một nhánh dài hạn `<ho-va-ten>-k<khoa>` tạo từ `main` (§6.2).
2. **Task**: mỗi việc là một issue nhãn `task`/`experiment` trong GitHub Project; Status `Todo` → `In progress` → `Done` (§6.3).
3. **Commit**: `<scope>: <mô tả> (task #<số>)` sau khi `tools/check-repo.sh --staged` báo 0 FAIL (§6.4).
4. **Báo cáo**: chạy flow tự động, commit kết quả tổng hợp, comment báo cáo vào issue (§6.5).
5. **Review**: trưởng nhóm chạy lại; đạt thì đóng issue → `Done` (§6.6).
6. **Bàn giao**: sau 1–2 tháng hoặc cuối học kỳ, PR từ nhánh thành viên vào `main`, PI merge bằng merge commit (§6.7).
