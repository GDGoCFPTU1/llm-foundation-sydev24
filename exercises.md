# Ngày 1 — Bài Tập & Phản Ánh
## Nền Tảng LLM API | Phiếu Thực Hành

**Thời lượng:** 1:30 giờ  
**Cấu trúc:** Lập trình cốt lõi (60 phút) → Bài tập mở rộng (30 phút)

---

## Phần 1 — Lập Trình Cốt Lõi (0:00–1:00)

Chạy các ví dụ trong Google Colab tại: https://colab.research.google.com/drive/172zCiXpLr1FEXMRCAbmZoqTrKiSkUERm?usp=sharing

Triển khai tất cả TODO trong `template.py`. Chạy `pytest tests/` để kiểm tra tiến độ.

**Điểm kiểm tra:** Sau khi hoàn thành 4 nhiệm vụ, chạy:
```bash
python template.py
```
Bạn sẽ thấy output so sánh phản hồi của GPT-4o và GPT-4o-mini.

---

## Phần 2 — Bài Tập Mở Rộng (1:00–1:30)

### Bài tập 2.1 — Độ Nhạy Của Temperature
Gọi `call_openai` với các giá trị temperature 0.0, 0.5, 1.0 và 1.5 sử dụng prompt **"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Khi temperature thấp, phản hồi thường chính xác, an toàn và lặp lại hơn. Khi temperature cao hơn, câu trả lời trở nên sáng tạo, đa dạng và đôi khi có thể khác biệt hơn so với dữ liệu mẫu ban đầu.

**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Tôi sẽ chọn temperature khoảng 0.2–0.4 cho chatbot hỗ trợ khách hàng vì cần câu trả lời ổn định, ít sáng tạo quá mức và có tính nhất quán cao để giảm sai sót.

---

### Bài tập 2.2 — Đánh Đổi Chi Phí
Xem xét kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người thực hiện 3 lần gọi API, mỗi lần trung bình ~350 token.

**Ước tính xem GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này:**
> Tổng token hàng ngày là 10.000 * 3 * 350 = 10.500.000 token. Với giá GPT-4o là 25 USD/1M token, chi phí khoảng 262.5 USD. Với GPT-4o-mini là 0.75 USD/1M token, chi phí khoảng 7.875 USD. Như vậy GPT-4o đắt hơn xấp xỉ 33 lần.

**Mô tả một trường hợp mà chi phí cao hơn của GPT-4o là xứng đáng, và một trường hợp GPT-4o-mini là lựa chọn tốt hơn:**
> GPT-4o xứng đáng khi ứng dụng cần phân tích chuyên sâu, trả lời chính xác cho nội dung pháp lý, y tế hoặc lập luận phức tạp. GPT-4o-mini phù hợp hơn cho chatbot FAQ, trợ lý tìm kiếm nhanh hoặc các tác vụ đơn giản cần tiết kiệm chi phí mà vẫn đảm bảo chất lượng đủ dùng.

---

### Bài tập 2.3 — Trải Nghiệm Người Dùng với Streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng khi người dùng tương tác trực tiếp và cần phản hồi nhanh, ví dụ như chatbot nói chuyện, hỗ trợ trực tuyến hoặc trả lời câu hỏi dài. Non-streaming phù hợp hơn khi đầu ra cần được tạo hoàn chỉnh trước khi hiển thị, hoặc với các tác vụ batch/định kỳ nơi độ trễ một vài giây không ảnh hưởng đến trải nghiệm.


## Danh Sách Kiểm Tra Nộp Bài
- [ ] Tất cả tests pass: `pytest tests/ -v`
- [ ] `call_openai` đã triển khai và kiểm thử
- [ ] `call_openai_mini` đã triển khai và kiểm thử
- [ ] `compare_models` đã triển khai và kiểm thử
- [ ] `streaming_chatbot` đã triển khai và kiểm thử
- [ ] `retry_with_backoff` đã triển khai và kiểm thử
- [ ] `batch_compare` đã triển khai và kiểm thử
- [ ] `format_comparison_table` đã triển khai và kiểm thử
- [ ] `exercises.md` đã điền đầy đủ
- [ ] Sao chép bài làm vào folder `solution` và đặt tên theo quy định 
