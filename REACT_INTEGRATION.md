# Hướng dẫn tích hợp hiển thị Từ vựng phương ngữ Nam Bộ vào ReactJS

Tài liệu này hướng dẫn cách tích hợp tính năng **Hiển thị thẻ từ vựng tham khảo (Vocabulary Cards)** vào dự án ReactJS (`namky-cultural-project`) sau khi backend đã nâng cấp lên bản v4 (trả về danh sách `relevant_words`).

---

## 1. Khai báo State lưu dữ liệu
Mở component chính xử lý tính năng dịch (thường là `App.js`), thêm state `relevantWords` để lưu trữ dữ liệu từ vựng nhận từ API:

```javascript
import React, { useState } from 'react';

// Trong component của bạn:
const [relevantWords, setRelevantWords] = useState([]);
```

---

## 2. Cập nhật Hàm gọi API (`fetch`)
Cập nhật phần xử lý phản hồi API để lưu danh sách từ vựng được gợi ý từ hệ thống RAG:

```javascript
const handleConvert = async () => {
  setLoading(true);
  try {
    const response = await fetch("https://fivec-bau-rag-backend.hf.space/api/convert", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text: inputText }),
    });
    
    const data = await response.json();
    
    // Lưu kết quả dịch
    setConvertedText(data.converted);
    
    // Lưu danh sách từ vựng tham khảo vào state
    setRelevantWords(data.relevant_words || []);
  } catch (error) {
    console.error("Lỗi kết nối API:", error);
  } finally {
    setLoading(false);
  }
};
```

*(Lưu ý: Nếu có hàm reset/xóa nội dung dịch, hãy gọi thêm `setRelevantWords([])` để dọn dẹp giao diện).*

---

## 3. Render giao diện thẻ từ vựng (JSX)
Đặt đoạn mã JSX này ở phía dưới phần hiển thị kết quả dịch câu để vẽ ra các Card từ vựng:

```jsx
{/* Hiển thị danh sách từ vựng tham chiếu */}
{relevantWords.length > 0 && (
  <div className="vocab-section">
    <h3 className="vocab-section-title">
      📖 Từ vựng Nam Bộ tham chiếu ({relevantWords.length})
    </h3>
    <div className="vocab-grid">
      {relevantWords.map((wordObj, idx) => {
        let posList = [];
        let viDuList = [];
        
        // Parse dữ liệu từ loại (pos) và ví dụ (vi_du) được lưu dạng JSON string
        try {
          posList = JSON.parse(wordObj.pos || "[]");
        } catch (e) {
          posList = Array.isArray(wordObj.pos) ? wordObj.pos : [];
        }
        
        try {
          viDuList = JSON.parse(wordObj.vi_du || "[]");
        } catch (e) {
          viDuList = Array.isArray(wordObj.vi_du) ? wordObj.vi_du : [];
        }

        return (
          <div key={idx} className="vocab-card">
            <div className="vocab-card-header">
              <span className="vocab-word">{wordObj.tu}</span>
              <div className="vocab-pos-container">
                {posList.map((pos, pIdx) => (
                  <span key={pIdx} className="vocab-pos-badge">
                    {pos}
                  </span>
                ))}
              </div>
            </div>
            
            {wordObj.tu_hien_nay && (
              <div className="vocab-modern-eq">
                Nói cách khác: <strong>{wordObj.tu_hien_nay}</strong>
              </div>
            )}
            
            <p className="vocab-meaning">{wordObj.nghia}</p>
            
            {viDuList.length > 0 && (
              <div className="vocab-examples">
                <span className="vocab-example-label">Ví dụ ngữ cảnh xưa:</span>
                {viDuList.map((vidu, vIdx) => (
                  <p key={vIdx} className="vocab-example-item">
                    💡 <em>"{vidu}"</em>
                  </p>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  </div>
)}
```

---

## 4. Thiết lập CSS Modern Glassmorphism
Thêm đoạn mã CSS này vào tệp CSS chính của bạn (`App.css` hoặc `index.css`):

```css
/* Container khu vực từ vựng */
.vocab-section {
  margin-top: 2.5rem;
  padding-top: 2rem;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
  text-align: left;
}

.vocab-section-title {
  font-size: 1.3rem;
  font-weight: 600;
  color: #f3f4f6;
  margin-bottom: 1.25rem;
  display: flex;
  align-items: center;
  gap: 0.6rem;
}

/* Grid xếp các thẻ từ vựng */
.vocab-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 1.25rem;
}

/* Thẻ từ vựng (Card) phong cách Glassmorphism */
.vocab-card {
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid rgba(255, 255, 255, 0.07);
  border-radius: 14px;
  padding: 1.25rem;
  transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
  backdrop-filter: blur(10px);
  display: flex;
  flex-direction: column;
}

.vocab-card:hover {
  transform: translateY(-3px);
  background: rgba(255, 255, 255, 0.06);
  border-color: rgba(251, 191, 36, 0.4); /* Màu viền hổ phách khi hover */
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
}

/* Tiêu đề từ và Badge Từ loại */
.vocab-card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.75rem;
}

.vocab-word {
  font-size: 1.2rem;
  font-weight: 700;
  color: #fbbf24; /* Màu vàng hổ phách nổi bật */
  letter-spacing: -0.01em;
}

.vocab-pos-container {
  display: flex;
  gap: 0.35rem;
}

.vocab-pos-badge {
  font-size: 0.7rem;
  padding: 0.15rem 0.5rem;
  border-radius: 9999px;
  background: rgba(251, 191, 36, 0.1);
  color: #fcd34d;
  border: 1px solid rgba(251, 191, 36, 0.25);
  font-weight: 600;
  text-transform: capitalize;
}

/* Phần từ tương đương hiện nay */
.vocab-modern-eq {
  font-size: 0.85rem;
  color: #9ca3af;
  margin-bottom: 0.75rem;
  padding: 0.35rem 0.6rem;
  background: rgba(255, 255, 255, 0.02);
  border-radius: 6px;
  width: fit-content;
}

/* Định nghĩa */
.vocab-meaning {
  font-size: 0.95rem;
  color: #e5e7eb;
  line-height: 1.55;
  margin-bottom: 1rem;
  flex-grow: 1;
}

/* Ví dụ ngữ cảnh xưa */
.vocab-examples {
  border-left: 2px solid rgba(251, 191, 36, 0.4);
  padding-left: 0.85rem;
  margin-top: auto;
}

.vocab-example-label {
  display: block;
  font-size: 0.75rem;
  color: #6b7280;
  margin-bottom: 0.25rem;
  font-weight: 600;
  text-transform: uppercase;
}

.vocab-example-item {
  font-size: 0.85rem;
  color: #9ca3af;
  line-height: 1.45;
  margin-top: 0.15rem;
}
```
