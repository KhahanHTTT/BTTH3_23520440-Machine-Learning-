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
| Accuracy  | 79.17%    | 78.69%           |
| Precision | 76.97%    | 75.35%           |
| Recall    | 95.13%    | 97.95%           |
| F1 Score  | 85.09%    | 85.17%           |
