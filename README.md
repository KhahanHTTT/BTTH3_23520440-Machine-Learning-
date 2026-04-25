# BTTH3 - Support Vector Machine

## Thông tin sinh viên
| | |
|---|---|
| Họ và tên | Võ Hà Khả Hân |
| MSSV | 23520440 |
| Môn học | Học máy thống kê |

## Mô tả
- **Assignment 1**: Cài đặt Soft-Margin SVM từ đầu bằng NumPy + SGD
- **Assignment 2**: Sử dụng scikit-learn SVM, so sánh kết quả với Assignment 1

## Dataset
Download tại: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Giải nén vào thư mục `chest_xray/` cùng cấp với `main.py`

## Chạy chương trình
```bash
python main.py
```

## Kết quả

| Metric    | NumPy SVM | scikit-learn SVM |
|-----------|:---------:|:----------------:|
| Accuracy  | 82.53%    | 75.48%           |
| Precision | 88.08%    | 72.07%           |
| Recall    | 83.33%    | 99.23%           |
| F1 Score  | 85.64%    | 83.50%           |

## Nhận xét
- **NumPy SVM** cho Accuracy và Precision cao hơn, model cân bằng hơn giữa các metric
- **scikit-learn SVM** có Recall rất cao (99.23%) — gần như phát hiện được toàn bộ ca Pneumonia, nhưng Precision thấp hơn (nhiều dự đoán sai hơn)
- Sự khác biệt do 2 model dùng thuật toán tối ưu khác nhau: SGD (NumPy) vs SMO (sklearn)
