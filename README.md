# 🧩 Code Comparison Tool (WIP)

> **Status:** 🚧 _In Progress_  
> Streamlit 기반의 코드 비교 및 시각화 툴입니다.  
> 동일한 기능을 하는 두 Python 파일을 선택하여 **차이점을 시각적으로 비교**하고,  
> 코드 하이라이팅 및 스크롤 동기화 기능을 제공합니다.

---

## 📁 Project Structure

```
project_root/
├── app.py                     # Streamlit 메인 앱
├── pages/
│   └── compare_code.py        # 코드 비교 페이지
├── api/
│   └── main.py                # FastAPI 서버 (파일 리스트/내용 반환)
├── data/
│   ├── code/                  # 비교할 .py 파일 저장 폴더
│   └── results/               # 결과 파일 (선택사항)
└── requirements.txt
```

---

## ⚙️ Features

- ✅ 파일 목록 자동 로드 (FastAPI 연동)
- ✅ 코드 A/B 선택 후 HTML Diff 비교
- ✅ 하이라이트 및 줄 단위 차이 시각화
- 🔄 스크롤 동기화 지원 (개선 중)
- 🚧 향후 추가 예정:
  - 코드 실행 결과 비교
  - 변경점 자동 요약
  - SQLite 연동을 통한 이력 관리

---

## 🧠 How It Works

1. FastAPI 서버(`api/main.py`)가 `/list_files`, `/get_file` 엔드포인트를 제공  
2. Streamlit 앱(`app.py`)에서 두 파일을 선택  
3. `difflib.HtmlDiff`를 활용해 차이점 비교  
4. HTML 결과를 렌더링하여 Streamlit 페이지에 표시

---

## 💻 Run Locally

### 1️⃣ 환경 세팅
```bash
# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)

# 패키지 설치
pip install -r requirements.txt
```

### 2️⃣ API 서버 실행
```bash
cd api
uvicorn main:app --reload --port 8000
```

### 3️⃣ Streamlit 앱 실행
```bash
streamlit run app.py
```

---

## 🧩 Example

| Before | After |
|:--:|:--:|
| <img src="assets/example_before.png" width="400"> | <img src="assets/example_after.png" width="400"> |

---

## 📅 Development Roadmap

| 단계 | 내용 | 상태 |
|------|------|------|
| 1 | FastAPI + Streamlit 연동 | ✅ 완료 |
| 2 | 코드 비교 시각화 (HTMLDiff) | ✅ 완료 |
| 3 | 스크롤 동기화 | 🚧 진행 중 |
| 4 | 파일 삭제/추가 기능 | ⏳ 예정 |
| 5 | SQLite 기반 파일 관리 | ⏳ 예정 |

---

## 🧾 Requirements

- Python 3.9+
- Streamlit 1.37+
- FastAPI 0.110+
- Requests, Pandas
