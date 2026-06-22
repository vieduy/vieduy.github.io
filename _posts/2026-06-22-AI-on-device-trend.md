---
title: "Tương Lai Của AI On-Device: 7 Xu Hướng Sẽ Định Hình Ngành Trong 5 Năm Tới"
date: 2026-06-22 10:00:00 +0700
categories: [AI, SLM]
toc: false
---

{% include lang-toggle.html %}

<div class="lang-content" data-lang="en" markdown="1">

## The Future of On-Device AI: 7 Trends That Will Shape the Field Over the Next 5 Years

> _English translation coming soon. Replace this block with the translated article — keep it inside this `<div ... markdown="1">` so Markdown still renders._

</div>

<div class="lang-content" data-lang="vi" markdown="1">

> Năm 2025 đánh dấu một threshold quan trọng: lần đầu tiên, on-device AI inference đạt sub-20ms latency trên điện thoại tầm trung cho production computer vision models. Không phải trên server, không phải trên flagship đắt tiền — trên hardware mà phần lớn người dùng đang cầm trên tay. Đây không còn là công nghệ tương lai. Nó đang xảy ra. Và từ điểm này trở đi, tốc độ thay đổi sẽ nhanh hơn nhiều so với 5 năm trước.

---

## 1. NPU Trở Thành Công Dân Hạng Nhất Của Silicon

Trong thập kỷ trước, NPU là "tính năng thêm vào" trên chip — một block nhỏ ở góc die, được nhắc đến trong slide marketing nhưng ít ai thực sự dùng đến. Năm 2025–2026 đánh dấu sự đảo ngược hoàn toàn: NPU trở thành thành phần được đầu tư nhiều nhất trên SoC.

**Những con số cụ thể:**

Qualcomm Snapdragon X2 Elite Extreme đạt 80 TOPS — gần gấp đôi thế hệ trước và vượt qua Apple M4 (38 TOPS) về raw NPU throughput. Snapdragon 8 Elite Gen 5, chip sẽ có mặt trong flagship Android 2026, cải thiện 37% so với thế hệ trước và xử lý được **220 tokens/giây** cho các tác vụ AI tổng quát, tăng hơn 3 lần so với 70 tokens/giây của thế hệ trước. Với VLM được optimize, con số này lên đến 11.000+ tokens/giây ở prefill phase.

IDC dự báo đến 2028, **94% máy tính mới xuất xưởng sẽ có NPU**. Năm 2026, AI PC sẽ chiếm hơn 50% tổng doanh số PC. Đây không còn là câu hỏi "có NPU không" mà là "NPU của bạn mạnh đến đâu".

**Implication cho AI Engineer:** Kỹ năng accelerator targeting — viết code chạy đúng trên NPU thay vì fallback về CPU — sẽ từ "nice to have" trở thành điều kiện tuyển dụng cơ bản. Khoảng cách giữa CPU inference và NPU inference có thể lên đến 20x latency. Engineer nào không biết cách exploit NPU sẽ ship sản phẩm chậm hơn đối thủ theo đúng nghĩa đen.

---

## 2. Cuộc Cách Mạng 1-Bit: Khi LLM Không Còn Cần GPU

Nếu có một nghiên cứu năm 2025 có khả năng thay đổi toàn bộ cách chúng ta nghĩ về on-device LLM, đó là **BitNet b1.58 2B4T** — model 2 tỷ tham số đầu tiên được train từ đầu với ternary weights {-1, 0, +1} trên 4 nghìn tỷ tokens, phát hành tháng 4/2025.

**Những con số nói lên tất cả:**

| | BitNet b1.58 2B | Gemma 3 1B | Llama 3.2 1B |
|---|---|---|---|
| Kích thước (non-embedding) | **0.4 GB** | 1.4 GB | ~1.2 GB |
| Decode latency trên CPU | **29ms** | ~80ms | ~90ms |
| Energy per inference | **6x thấp hơn** Gemma 3 1B | baseline | tương đương |

Điểm khác biệt cốt lõi: khi weights chỉ là -1, 0, hoặc +1, matrix multiplication biến thành phép cộng và trừ thuần túy — không có phép nhân. Điều này không chỉ nhanh hơn mà còn tiêu thụ ít energy hơn đáng kể, và **bất kỳ CPU nào cũng làm được tốt**, không cần GPU hay NPU đặc biệt.

Hệ sinh thái đang theo kịp nhanh hơn dự kiến. bitnet.cpp ra mắt GPU inference kernels vào tháng 5/2025. Bản cập nhật tháng 1/2026 tăng thêm 1.15–2.1x CPU speedup. llama.cpp đã hỗ trợ BitNet qua `-IQ1_S` format. Quan trọng hơn: **MediaTek Dimensity 9500** trở thành chipset di động đầu tiên có native hardware support cho BitNet 1.58-bit processing, giảm 33% power consumption — tín hiệu cho thấy hardware manufacturers đang bắt đầu đặt cược vào hướng đi này.

Giới hạn hiện tại: đến đầu 2026, chưa có native BitNet model nào lớn hơn 2B tham số. Microsoft cho biết đang nghiên cứu scale lên 7B và 13B, nhưng chưa có timeline chính thức. Đây vẫn là early adoption — nhưng với tốc độ phát triển hiện tại, cửa sổ "early" sẽ đóng lại sớm hơn nhiều người nghĩ.

---

## 3. Speculative Decoding — Từ Paper Nghiên Cứu Thành Chuẩn Mực Production

Ba năm trước, speculative decoding là kỹ thuật thú vị trong các bài báo ML. Đến 2025, nó đã trở thành standard practice trong mọi serious on-device LLM deployment.

**Cơ chế:** một draft model nhỏ (thường 1B) chạy trước để predict nhiều tokens, sau đó model lớn verify song song. Khi draft đúng — throughput tăng đột biến. Khi draft sai — không mất gì thêm so với baseline.

**Kết quả thực tế:**
- llama.cpp với 1B draft + 8B target trên MacBook M1: **180+ tokens/giây**
- EdgeLLM (IEEE Transactions on Mobile Computing): **9.3x speedup** trên mobile
- MNN-LLM của Alibaba: **8.6x prefill speedup** so với llama.cpp trên CPU, **25.3x** trên GPU
- MediaTek Dimensity 9400+ tích hợp Speculative Decoding+ ở hardware level, cải thiện thêm 20% cho agentic AI workloads

Điều quan trọng hơn benchmark: speculative decoding thay đổi cách chúng ta thiết kế hệ thống on-device LLM. Thay vì một model duy nhất, bạn cần quản lý cặp draft-target, KV cache của cả hai, và logic verification. AI Engineer nào chỉ quen deploy single model sẽ cần học lại architecture của toàn bộ inference pipeline.

---

## 4. Multimodal On-Device — LLM Không Còn Mù

Năm 2024, đưa một Vision-Language Model (VLM) lên mobile là bài toán nghiên cứu. Năm 2026, nó đang trở thành production reality.

Động lực kỹ thuật: SigLIP 2 (tháng 2/2025) thay thế CLIP làm vision encoder chuẩn, với multilingual support tốt hơn và dynamic resolution processing — xử lý ảnh 4K mà không cần resize hay token explosion. Kiến trúc multimodal native (thay vì "gắn vision vào LLM có sẵn") bắt đầu chiếm ưu thế.

Các model đáng chú ý:
- **BlueLM-V-3B** (Vivo): co-design giữa algorithm và system cho mobile, optimize cho Qualcomm NPU
- **MagicVL-2B**: lightweight visual encoder qua curriculum learning, chạy được trên mid-range device
- **Qwen3-VL** (Alibaba): flagship VLM với agentic capabilities và long-context comprehension

**Use cases đang trở thành thực tế:** scan tài liệu và hỏi trực tiếp về nội dung, camera nhận diện sản phẩm và gợi ý mà không cần internet, AR overlay giải thích môi trường xung quanh theo ngữ cảnh.

Thách thức kỹ thuật chưa giải quyết: vision encoder thường ăn nhiều memory hơn language model. Quản lý KV cache của VLM phức tạp hơn LLM thuần túy. Và latency của prefill phase (xử lý ảnh trước khi generate text) vẫn là bottleneck chính trên low-end device.

---

## 5. Agentic AI On-Device — Điện Thoại Từ Tool Thành Agent

Đây là sự thay đổi lớn nhất về paradigm trong 5 năm tới.

Gartner dự báo **40% ứng dụng enterprise sẽ có AI agent** vào cuối 2026, tăng từ dưới 5% năm 2025. Thị trường agentic AI toàn cầu: 28 tỷ USD (2024) → 127 tỷ USD (2029). Và một phần lớn trong số đó sẽ chạy trực tiếp trên device.

**Tại sao on-device lại quan trọng cho agentic AI?**

Agent không phải là chatbot. Nó thực hiện nhiều bước liên tiếp: Perception → Planning → Action → Self-Correction. Mỗi bước round-trip lên cloud thêm 200-800ms latency. Với một agent thực hiện 10-20 bước để hoàn thành một task, latency tích lũy đó là không thể chấp nhận trong real-time interaction. NPU on-device giải quyết bài toán này.

Samsung Galaxy S26 Ultra và Google Pixel 10 Pro (Tensor G5 + Gemini Live 2.0) là những flagship đầu tiên marketing rõ ràng với agentic capability — cross-app actions, autonomous task completion mà không cần user confirm từng bước.

Infrastructure layer cũng đang hình thành: **MCP (Model Context Protocol)** của Anthropic (tháng 11/2024) đã trở thành standard cho agent-to-tool communication, và **A2A protocol** cho agent-to-agent. Đây là những protocol mà AI Engineer on-device sẽ cần hiểu để build pipeline agentic trong vài năm tới.

**Implication:** AI Engineer on-device sẽ không chỉ deploy model — mà sẽ thiết kế hệ thống multi-model phối hợp với nhau, với state management, tool access, và error recovery chạy hoàn toàn local.

---

## 6. Privacy-First AI — Federated Learning Từ Optional Thành Regulatory

Có một câu hỏi mà nhiều người trong ngành đang dần phải đối mặt: *nếu tất cả user data đều phải lên cloud để model học, thì privacy có thực sự tồn tại không?*

**Federated learning** — train model trực tiếp trên device, chỉ gửi gradient updates lên server — đang chuyển từ academic concept sang regulatory requirement.

Tháng 4/2025, NVIDIA và Meta công bố tích hợp **NVIDIA FLARE với ExecuTorch**, đặt nền móng cho federated learning trên mobile devices ở quy mô production. EU đang thảo luận khả năng mandate federated training architecture cho AI trong critical infrastructure — một precedent mà, nếu được thông qua, sẽ lan rộng sang các ngành khác.

Về mặt kỹ thuật, federated learning giải quyết đồng thời ba bài toán: privacy (data không rời device), latency (model học từ local context), và personalization (mỗi device có model được fine-tune cho behavior của riêng user đó). Đây là trifecta mà cloud-only approach không thể đạt được.

Thị trường federated learning: 155 triệu USD (2025) → 315 triệu USD (2032). Nhỏ theo absolute terms — nhưng đây là con số của infrastructure layer, không phải end product. Impact thực tế sẽ lớn hơn nhiều.

**Kỹ năng mới AI Engineer cần chuẩn bị:** differential privacy, secure aggregation, gradient compression — những khái niệm nằm ở giao điểm giữa ML và cryptography.

---

## 7. Beyond Phones — AI Đang Lan Ra Toàn Bộ Vật Lý

Mobile phone là beachhead. Các form factor tiếp theo đang hình thành.

**Wearables và smart glasses** đang chuyển từ accessory sang computing platform. Everysight Maverick tích hợp Alif Ensemble E7 — fusion processor chạy AI inference trực tiếp trên kính, không cần phone làm proxy. Snap đang tích hợp on-device AI để giảm round-trip lên cloud trong AR experiences. OpenGlass (arxiv 2025) demonstrate ultra-low-power on-device AI cho eyewear với event-based vision sensor, kéo dài battery life đáng kể so với conventional camera approach.

**Điều kiện để smart glasses trở thành primary computing device:** MicroLED displays (chi phí vẫn đang cao), edge cloud rendering (để offload heavy compute khi cần mà không có latency), và contextual AI đủ tốt để hiểu môi trường xung quanh. Hầu hết analyst đặt convergence đầy đủ vào cửa sổ **2030–2035**.

**Embedded và IoT:** MediaTek APU, Qualcomm Hexagon, và các chip ARM Cortex-M với ML extensions đang đưa inference xuống các device không có OS đầy đủ — camera công nghiệp, sensor nông nghiệp, thiết bị y tế. Đây là segment mà memory constraint cực đoan (< 1MB RAM trong một số case), yêu cầu compression technique cực đoan, và BitNet 1.58 lại một lần nữa trở thành hướng đi đáng chú ý.

---

## Đọc Nhanh: Timeline Dự Báo

```
2025-2026 (Đang xảy ra)
├── Sub-20ms inference trên mid-range Android (đã đạt)
├── BitNet 2B production-ready, 7B đang nghiên cứu
├── Speculative decoding trở thành standard practice
├── VLM trên mobile: early production deployment
└── Agentic AI trên flagship phones: marketing → thực tế

2027-2028 (Có xác suất cao)
├── BitNet 7B+ available, ecosystem rộng hơn GGUF hiện tại
├── Federated learning regulatory requirements xuất hiện (EU first)
├── Multimodal on-device: mainstream mid-range
├── 94% new PCs có NPU (IDC forecast)
└── Agentic AI: multi-device coordination (phone + laptop + wearable)

2029-2030 (Đang hình thành)
├── Smart glasses với on-device AI: daily driver cho early adopters
├── On-device LLM: tất cả flagship phones chạy 7B+ natively
├── Personalized federated model: model của bạn ≠ model của người khác
└── AI inference trên MCU-class device: embedded everywhere

2030-2035 (Chưa chắc, nhưng đang hướng tới)
└── Smart glasses → primary computing device cho một số use cases
```

---

## Điều Này Có Nghĩa Gì Với AI Engineer On-Device

Ngành đang ở một inflection point thực sự. Không phải hype — mà là hardware đang bắt kịp với những gì ML research đã có thể làm từ vài năm trước.

**Ba điều sẽ định hình career path trong 5 năm tới:**

Thứ nhất, **hardware literacy sẽ phân tách engineer giỏi và engineer xuất sắc**. Biết viết code chạy đúng là baseline. Biết tại sao code đó chạy nhanh trên Hexagon NPU nhưng chậm trên Dimensity APU — và biết cách fix — là thứ tạo ra sự khác biệt thực sự.

Thứ hai, **LLM on-device sẽ đòi hỏi system thinking nhiều hơn model thinking**. KV cache management, speculative decoding pipeline, federated update mechanism — đây là những bài toán systems engineering, không phải ML research. AI Engineer nào hiểu cả hai sẽ cực kỳ khan hiếm.

Thứ ba, **form factor mới = opportunity mới**. Người build AI cho smart glasses, embedded medical devices, hay ultra-low-power IoT sensor ngay bây giờ đang đặt nền móng cho một platform sẽ có hàng tỷ devices trong vòng 5-7 năm. Đây là window mà những người đến sớm sẽ có lợi thế không thể bù đắp.

On-device AI không cạnh tranh với cloud AI — chúng giải quyết những bài toán khác nhau. Nhưng tỷ lệ bài toán phù hợp với on-device đang tăng nhanh. Và con người hiểu ngành này từ cả hai phía — ML lẫn hardware — vẫn còn rất hiếm.

---

*Số liệu trong bài được tổng hợp từ nghiên cứu thực tế (tháng 6/2026): Qualcomm, MediaTek, Microsoft BitNet, IEEE Transactions on Mobile Computing (EdgeLLM), IDC, Gartner, và các báo cáo phân tích thị trường của Grand View Research và Coherent Market Insights.*

</div>