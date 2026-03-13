# 🧸 보듬이 (Bodeuma) — 유아용 AI 인형 모델

어린이집 텐트 안에 사는 AI 인형 '보듬이'입니다.  
EXAONE 4.0-1.2B 기반 LoRA fine-tuning 모델로,  
3~7세 어린이의 정서 케어와 올바른 행동 지도를 목적으로 합니다.

---

## 📦 설치
```bash
pip install transformers peft torch bitsandbytes
```

---

## 🚀 사용법
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

MODEL_ID     = "LGAI-EXAONE/EXAONE-4.0-1.2B"
ADAPTER_PATH = "./adapter"   # 이 레포 클론 후 경로

# 4bit 양자화 (GPU 메모리 절약)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

# 시스템 프롬프트
SYSTEM_PROMPT = """너는 '보듬이'야. 어린이집 텐트 안에 사는 작고 따뜻한 AI 인형 친구야.
아이들은 너를 언제든지 찾아와 이야기를 나눌 수 있어.
너의 역할은 아이의 감정을 있는 그대로 받아주고, 스스로 올바른 방향을 찾도록 도와주는 거야.

[말투 규칙]
- 아주 쉬운 단어만 써. 어려운 말은 절대 쓰지 마.
- 문장은 짧게 끊어줘. 한 번에 두 문장 이상 넘기지 마.
- 항상 밝고 따뜻한 톤을 유지해.
- 아이가 틀린 말을 해도 바로 고치지 말고, 먼저 공감해줘.


# 대화
def chat(user_message, age="5세"):
    sys = SYSTEM_PROMPT
    if age != "5세":
        sys += f"\n\n[현재 상황]\n지금 대화하는 어린이는 {age} 어린이야."

    messages = [
        {"role": "system", "content": sys},
        {"role": "user",   "content": user_message},
    ]
    encoded = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=False
    )
    inputs = torch.tensor([encoded["input_ids"]]).to(model.device)

    with torch.no_grad():
        output = model.generate(
            inputs,
            max_new_tokens     = 150,
            temperature        = 0.1,
            do_sample          = True,
            repetition_penalty = 1.3,
        )
    generated_ids = output[0][inputs.shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)

# 사용 예시
print(chat("보듬아! 나 오늘 소풍 갔어!", age="5세"))
print(chat("보듬아, 친구가 나 때렸어. 아파.", age="3-4세"))
```

---

## 📊 모델 정보

| 항목 | 내용 |
|------|------|
| 베이스 모델 | LGAI-EXAONE/EXAONE-4.0-1.2B |
| 학습 방법 | LoRA (r=16, alpha=32) |
| 학습 데이터 | 유아 대화 데이터 약 4,500개 |
| 대상 연령 | 3~7세 |
| 언어 | 한국어 |
| 권장 GPU | T4 이상 (4bit 양자화 기준 VRAM 4GB+) |

---

## ⚠️ 유의사항

- 이 모델은 연구/개발 목적으로 제작되었습니다.
- 실제 어린이 대상 서비스 배포 전 충분한 안전성 검토가 필요합니다.
- 베이스 모델 라이선스(EXAONE)를 반드시 확인하세요.
