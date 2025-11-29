import os
# 최신 SDK에서는 'google' 패키지에서 'genai'를 가져옵니다.
from google import genai
from google.genai import types 

# API 키는 환경 변수에서 자동으로 로드됩니다. 
# 'GEMINI_API_KEY' 또는 'GOOGLE_API_KEY'로 설정되어 있다면 Client()에서 별도로 전달할 필요가 없습니다.
API_KEY = os.getenv("GOOGLE_API_KEY") 
if not API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is not set")

# 1. Client 객체 생성
# API 키가 환경 변수에 설정되어 있으므로 인자 없이 호출
client = genai.Client()

def gemini_free(prompt: str) -> str:
    with open("/home/goatyeon/ml_code_insight/backend/llm/gemini_debug.txt", "a", encoding="utf-8") as f:
        f.write("\n[CALL] gemini_free()\n")

    # 2. generate_content 메서드 사용
    response = client.models.generate_content(
        model="gemini-2.0-flash-001", # 모델 이름 변경 (최신 버전은 'models/' 접두사 불필요)
        contents=prompt,
        config=types.GenerateContentConfig( # 추가 설정은 config 객체로 전달
            temperature=0.2,
            max_output_tokens=2048,
        )
    )

    with open("/home/goatyeon/ml_code_insight/backend/llm/gemini_debug.txt", "a", encoding="utf-8") as f:
        f.write("PROMPT_HEAD: " + prompt[:200] + "\n")

    with open("/home/goatyeon/ml_code_insight/backend/llm/gemini_debug.txt", "a", encoding="utf-8") as f:
        f.write("[RESPONSE] Gemini returned successfully\n")

    # 3. 응답 텍스트 반환
    return response.text

# 예시 사용:
# result = gemini_free("코드 파서가 무엇인지 설명해 줘.")
# print(result)