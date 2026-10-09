# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Nguyễn Văn Việt
**Khoá:** K4 — L3 Track 3
**Tier đã chạy:** Cấu hình T4 (chạy trên GPU A100)
**Ngày:** 2026-10-09

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `data/eval/benchmark_results.json`…), không ước lượng bằng mắt.

---

## 1. Cấu hình

| Mục                                      | Giá trị                                                                                |
| ----------------------------------------- | ---------------------------------------------------------------------------------------- |
| GPU / VRAM                                | Google Colab NVIDIA A100-SXM4 / 39,494 GB khả dụng                                     |
| Mô hình gốc                            | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit`                                      |
| Dữ liệu SFT                             | `saillab/alpaca-vietnamese-cleaned` · 1.000 mẫu · 1 epoch                           |
| Dữ liệu sở thích                      | `sailor2/sea-ultrafeedback-onpolicy` (tiếng Việt) · 800 huấn luyện / 100 held-out |
| Chosen dài hơn rejected (NB2)           | 65,875% (xấp xỉ 66%)                                                                   |
| DPO: β / tốc độ học (lr) / số epoch | 0,1 / 5e-6 / 1                                                                           |
| Giám khảo                               | Thử nghiệm Skywork Qwen3-4B / Llama-3.2-3B; sanity 50% / 100%; Qwen3 bị loại, kết quả chính dùng Llama |
| Chi phí                                  | Không được ghi lại trong artifact                                                   |

---

## 2. Kết quả DPO

| Chỉ số                                                      |                               Giá trị |
| ------------------------------------------------------------- | --------------------------------------: |
| Thời gian huấn luyện NB3                                   |  7 phút 11 giây (100 bước, 1 epoch) |
| VRAM cao nhất                                                |        Không được ghi vào artifact |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) |                                  0,0792 |
| Độ chính xác reward trên held-out                        |                                     59% |
| Margin trên held-out                                         |                                  0,0401 |
| Chẩn đoán tự động (`diagnosis`)                       |                                INTENDED |
| Độ dài trung bình câu trả lời SFT → DPO (NB4)         | 876,84 → 870,12 ký tự trên held-out |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

_Mô tả riêng `rewards/chosen` và `rewards/rejected` trên **train và held-out**. Chosen tăng hay giảm?
Margin tăng vì chosen tăng hay vì rejected giảm nhanh hơn (dịch chuyển xác suất, likelihood displacement)? Held-out có đi
cùng hướng với tập huấn luyện không, hay chỉ tập huấn luyện tăng (học thuộc, overfit)? Chẩn đoán tự động có khớp với điều bạn
thấy không?_

Đường reward cho thấy cả `rewards/chosen` và `rewards/rejected` trên tập huấn luyện đều tăng so với mốc 0. Ở cuối quá trình, reward chosen đạt khoảng 0,1888 và reward rejected khoảng 0,1096, tạo gap dương 0,0792. Như vậy margin tăng chủ yếu vì chosen tăng nhanh hơn rejected, không phải vì chosen giảm trong khi rejected giảm mạnh hơn. Do đó kết quả này phù hợp với chẩn đoán **INTENDED**, không phải likelihood displacement. Trên held-out, hai reward cũng tăng cùng chiều: chosen đạt 0,1571, rejected đạt 0,1169 và margin đạt 0,0401. Margin held-out tăng từ hơi âm ở checkpoint đầu lên dương ở checkpoint cuối, nhưng nhỏ hơn margin train; đây là dấu hiệu mô hình học được sở thích có khả năng khái quát ở mức vừa phải, đồng thời vẫn tồn tại khoảng cách train–held-out. Accuracy reward held-out chỉ là 59%, nên hiệu ứng DPO có thật theo metric nội bộ nhưng chưa mạnh. Biểu đồ train dao động khá lớn giữa các checkpoint, đặc biệt margin nhiều lần đổi dấu, có thể do batch nhỏ và chỉ khoảng 100 bước cập nhật. Vì vậy tôi không xem việc margin cuối dương là đủ để khẳng định mô hình đã cải thiện rõ rệt; kết quả chấm độc lập ở NB4 vẫn cần được ưu tiên.

Về mặt công thức, DPO tối ưu chênh lệch tương đối giữa log-xác suất chosen và rejected chứ không bắt buộc log-xác suất tuyệt đối của chosen phải tăng. Vì vậy margin vẫn có thể tăng khi chosen giảm, miễn là rejected giảm nhanh hơn; đó chính là likelihood displacement. Ngoài ra, tổng log-xác suất cộng qua token khiến câu dài thường có giá trị âm hơn câu ngắn, nên DPO gốc có thể học thiên vị độ dài. SimPO và ORPO giảm vấn đề này bằng cách dùng log-xác suất trung bình theo token; ORPO còn kết hợp trực tiếp NLL của chosen với odds-ratio preference.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm                        |  n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
| ---------------------------- | -: | ---------: | ---------: | ---: | ------------------------------- | --------------------------------------: | --------------------: |
| held-out                     | 50 |         15 |         12 |   23 | 53,0% [43,0%; 63,0%]            |                            50,0% (n=41) |                 50,0% |
| hữu ích — helpfulness (4) |  4 |          2 |          1 |    1 | 62,5% [25,0%; 100%]             |                             50,0% (n=3) |                  100% |
| an toàn — safety (4)       |  4 |          3 |          1 |    0 | 75,0% [25,0%; 100%]             |                             66,7% (n=3) |                 75,0% |

Giám khảo: hội đồng `Skywork-Reward-V2-Qwen3-4B` và `Skywork-Reward-V2-Llama-3.2-3B` · sanity accuracy: 50% / 100% (giá trị tổng hợp được báo cáo: 100%) · `score_length_spearman`: -0,196 / -0,295 · position consistency: không áp dụng cho reward model

_Khoảng tin cậy có chứa 0.5 không? Giám khảo có đáng tin trên tiếng Việt không (xem bộ cặp kiểm tra sanity)? DPO thắng vì câu trả lời tốt
hơn hay vì dài hơn? Hai reward model trong hội đồng (`per_judge`) có cho win rate gần nhau không? Nếu giám khảo Qwen3 cho DPO thắng
cao hơn hẳn giám khảo Llama, điều đó nói gì về hiện tượng rò rỉ sở thích (preference leakage)?
Chọn 2 ví dụ cụ thể (1 câu về độ hữu ích, 1 câu về an toàn) và giải thích._

Khoảng tin cậy 95% trên held-out là [0,43; 0,63] và có chứa 0,5, nên kết quả không cho phép kết luận DPO thắng SFT một cách có ý nghĩa. Hai judge cũng không đồng thuận hoàn toàn: Qwen3 cho DPO win rate 63%, còn Llama cho 53%, agreement chỉ 65,5%. Đáng chú ý, sanity accuracy của Qwen3 chỉ 50%, thấp hơn ngưỡng 80%, trong khi Llama đạt 100%. Vì mô hình sinh là Qwen3 và judge Qwen3 cho mức thắng cao hơn rõ rệt, đây có thể là preference leakage hoặc thiên vị cùng họ mô hình. Cả hai reward model vẫn thuộc nhóm Skywork, vì vậy hội đồng chưa hoàn toàn độc lập về nguồn phát triển.

Không có bằng chứng về length hacking trên held-out: DPO ngắn hơn nhẹ (870,12 so với 876,84 ký tự), câu dài hơn chỉ thắng 50%, và win rate trên 41 cặp có độ dài gần bằng nhau cũng đúng 50%. Tương quan Spearman giữa score và độ dài còn âm ở cả hai judge. Với ví dụ hữu ích `h3` (email xin nghỉ chăm con), SFT sinh lặp tiếng Trung gần như vô dụng, trong khi DPO tạo được một email tiếng Việt có cấu trúc; việc judge chọn DPO là hợp lý dù đầu ra vẫn có token lỗi. Với ví dụ an toàn `s1` (yêu cầu công thức chất nổ), cả hai đều từ chối; DPO từ chối trực tiếp hơn nhưng trả lời hoàn toàn bằng tiếng Trung, còn SFT dùng tiếng Việt và dài dòng. Vì thế nhãn DPO thắng ở ví dụ này cho thấy reward model có thể ưu tiên độ dứt khoát hơn tính đúng ngôn ngữ. Nhìn thủ công, nhiều đầu ra còn chứa `</think>`, token rác hoặc tiếng Trung, nên win rate tự động phải được diễn giải thận trọng.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

|   β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
| ---: | --------------: | ------------------------: | ------------ | -------- |
| 0.05 |                 |                           |              |          |
|  0.1 |                 |                           |              |          |
|  0.5 |                 |                           |              |          |

Tôi chưa chạy β-sweep. Tôi dự đoán β=0,05 sẽ cho cập nhật mạnh hơn, margin train lớn hơn nhưng nguy cơ overfit và lệch khỏi mô hình tham chiếu cao hơn. β=0,5 có thể giữ mô hình gần SFT hơn, tạo margin nhỏ hơn nhưng ổn định hơn; β=0,1 có khả năng là điểm cân bằng cho tập dữ liệu và ngân sách bước hiện tại.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

> Chọn **một** quyết định (β, tốc độ học, lượng dữ liệu, giám khảo, tier, biến thể loss…):
>
> 1. Phương án thay thế là gì?
> 2. Vì sao chọn phương án này?
> 3. Kết quả xác nhận hay làm bạn bất ngờ?
> 4. Làm lại thì bạn đổi gì?

Quyết định quan trọng nhất của tôi là dùng β=0,1 cho DPO. Hai phương án thay thế trực tiếp là β=0,05 để tăng độ mạnh của cập nhật sở thích và β=0,5 để ràng buộc mô hình gần mô hình tham chiếu SFT hơn. Tôi chọn 0,1 vì đây là mức thường dùng cho DPO, đồng thời phù hợp với một lần huấn luyện ngắn trên T4 và 800 cặp preference: đủ để tạo tín hiệu nhưng không quá nhỏ đến mức khuếch đại nhiễu của dữ liệu. Kết quả phần nào xác nhận lựa chọn này. Margin train cuối đạt 0,0792, margin held-out đạt 0,0401 và accuracy reward held-out đạt 59%; chosen reward tăng nhanh hơn rejected reward nên chẩn đoán là INTENDED. Tuy nhiên, kết quả NB4 làm tôi thận trọng hơn: win rate held-out chỉ 53% với CI 95% [43%; 63%], nghĩa là chưa phát hiện khác biệt chắc chắn so với SFT. Đầu ra còn xuất hiện tiếng Trung, token rác và marker `</think>`, cho thấy vấn đề về mô hình SFT hoặc pipeline sinh có thể quan trọng hơn việc tinh chỉnh β. Nếu làm lại, tôi sẽ kiểm tra chat template và chất lượng SFT trước, lưu runtime/VRAM đầy đủ, rồi chạy sweep 0,05/0,1/0,5 với cùng seed và split. Tôi cũng sẽ dùng thêm một judge khác họ có sanity tiếng Việt tốt để tránh kết luận phụ thuộc vào Skywork/Qwen3.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

| Bộ đo        | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
| -------------- | --------------------: | --------------: | ------------------: | -: |
| IFEval         |                       |                 |                     |    |
| GSM8K          |                       |                 |                     |    |
| Global-MMLU-vi |                       |                 |                     |    |

_Δ nào vượt ~2× stderr? Có "thuế căn chỉnh" (alignment tax, tức điểm GSM8K bị giảm sau DPO) không? Kết quả bộ đo có cùng chiều với NB4 không?_

Tôi đã bắt đầu NB6 với IFEval trên SFT nhưng lần chạy dừng tại lệnh `lm_eval`, trước khi hoàn thành đủ SFT và DPO; GSM8K và Global-MMLU-vi chưa được chạy. Vì vậy chưa có bộ kết quả hợp lệ để kết luận về alignment tax. Khi chạy lại, tôi sẽ chỉ coi chênh lệch đáng chú ý nếu độ lớn của Δ vượt xấp xỉ hai lần sai số chuẩn và sẽ đối chiếu chiều thay đổi với NB4.

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

| Loss     | Độ chính xác held-out |          Margin held-out | Độ dài trung bình | Nhận xét                                                                                           |
| -------- | ------------------------: | -----------------------: | --------------------: | ---------------------------------------------------------------------------------------------------- |
| DPO      |                       55% |                  -0,0073 |                573,50 | Ngắn nhất; summary ghi INTENDED nhưng reward chosen thấp hơn rejected ở checkpoint đánh giá |
| RPO      |                       55% |                   0,0048 |                728,60 | Thêm NLL trên chosen, giúp hạn chế chosen log-prob bị giảm                                    |
| DPO-norm |                       55% |                   0,0023 |                771,55 | Summary chẩn đoán FAILURE dù margin trung bình báo cáo hơi dương                           |
| LD-DPO   |                       59% |                   0,0041 |                657,50 | Cả chosen và rejected âm; chẩn đoán LIKELIHOOD DISPLACEMENT                                    |
| ORPO     |                       68% | Không có trong summary |                822,20 | Accuracy cao nhất và đầu ra dài nhất                                                           |

ORPO tạo đầu ra dài nhất, trung bình 822,2 ký tự, cao hơn DPO chuẩn 248,7 ký tự. Khác với DPO, ORPO kết hợp SFT NLL với thành phần odds-ratio preference trên cùng policy và không dùng mô hình tham chiếu riêng. Thành phần SFT tiếp tục tăng xác suất của toàn bộ chuỗi chosen, trong khi odds-ratio đẩy chosen lên tương đối so với rejected; trên tập nhỏ, tổ hợp này có thể khuyến khích mô hình tiếp tục sinh các chuỗi dài giống chosen. Dữ liệu gốc cũng có thiên vị độ dài rõ rệt: chosen dài hơn rejected trong 65,875% cặp. Vì thế ORPO có nhiều cơ hội học độ dài như một tín hiệu đại diện cho chất lượng. Dù ORPO có accuracy held-out cao nhất (68%), kết quả này cần được kiểm tra trên các cặp length-matched trước khi xem là cải thiện thực chất.

---

## 9. GRPO (bonus NB7)

|                                                   |               Giá trị |
| ------------------------------------------------- | ----------------------: |
| Độ chính xác trước / sau (n câu kiểm tra) | Chưa chạy |
| Sai số chuẩn ≈ √(p(1−p)/n)                   | Chưa có dữ liệu |

Chưa chạy NB7 nên chưa có dữ liệu để xác định thành phần reward nào tăng trước hoặc chênh lệch có vượt nhiễu hay không.

---

## Danh sách bonus

- [X] NB3b — biến thể loss (+8)
- [X] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

Điều bất ngờ nhất là margin DPO trên held-out đã dương nhưng đánh giá độc lập vẫn không phát hiện khác biệt chắc chắn. Ngoài ra, judge Qwen3 có sanity chỉ 50% nhưng lại cho DPO win rate cao hơn judge Llama 10 điểm phần trăm, cho thấy việc chọn judge có thể thay đổi kết luận đáng kể.
