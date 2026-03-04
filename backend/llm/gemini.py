import os
import google.generativeai as genai
from google.genai import types 

# API 키는 환경 변수에서
API_KEY = os.getenv("GOOGLE_API_KEY") 
if not API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is not set")

# 1. Client 객체 생성
# API 키가 환경 변수에 설정되어 있으므로 인자 없이 호출
client = genai.Client()

def gemini_free(prompt: str) -> str:

    # 2. generate_content 메서드 사용
    response = client.models.generate_content(
        model="gemini-2.0-flash-001", # 모델 이름
        contents=prompt,
        config=types.GenerateContentConfig( # 추가 설정은 config 객체로 전달
            temperature=0.5,
            max_output_tokens=2048,
        )
    )

    # 3. 응답 텍스트 반환
    return response.text
