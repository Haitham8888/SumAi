from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os
from typing import List, Optional

app = Flask(__name__)

# =========================
# إعدادات المسارات
# =========================
MODEL_DIR = "./models_cache"
ALLAM_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "models--humain-ai--ALLaM-7B-Instruct-preview/snapshots/a28dd1e67420cde72d3629c8633a974cf7d9c366"
)

# =========================
# اختيار الجهاز
# =========================
USE_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if USE_CUDA else "cpu"

# bf16 ممتاز على بعض كروت NVIDIA الحديثة، لو ما يدعمه خله fp16
if USE_CUDA:
    try:
        _ = torch.tensor([1.0], device="cuda", dtype=torch.bfloat16)
        TORCH_DTYPE = torch.bfloat16
    except Exception:
        TORCH_DTYPE = torch.float16
else:
    TORCH_DTYPE = torch.float32

print(f"استخدام Device: {DEVICE} | dtype: {TORCH_DTYPE}")

# =========================
# تحميل النموذج
# =========================
summarization_pipeline = None
tokenizer: Optional[AutoTokenizer] = None

print("جاري تحميل نموذج ALLaM...")
try:
    tokenizer = AutoTokenizer.from_pretrained(
        ALLAM_MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True
    )

    # مهم جدًا لبعض نماذج LLaMA: pad = eos
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # device_map لازم يكون "auto" أو None (مو "cuda"/"cpu")
    if USE_CUDA:
        model = AutoModelForCausalLM.from_pretrained(
            ALLAM_MODEL_PATH,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=TORCH_DTYPE,
            device_map="auto",
            low_cpu_mem_usage=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            ALLAM_MODEL_PATH,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=TORCH_DTYPE,
            device_map=None,
            low_cpu_mem_usage=True
        )
        model.to("cpu")

    summarization_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer
    )

    print("✓ تم تحميل نموذج ALLaM بنجاح")

except Exception as e:
    print(f"✗ خطأ في تحميل نموذج ALLaM: {str(e)}")
    summarization_pipeline = None


# =========================
# مساعدات التوكن/البرومبت
# =========================
def _model_context_limit() -> int:
    """
    محاولة معرفة حد السياق (context length) من model/tokenizer.
    """
    try:
        cfg = summarization_pipeline.model.config
        if hasattr(cfg, "max_position_embeddings") and cfg.max_position_embeddings:
            return int(cfg.max_position_embeddings)
    except Exception:
        pass

    try:
        if tokenizer is not None and tokenizer.model_max_length and tokenizer.model_max_length < 10**9:
            return int(tokenizer.model_max_length)
    except Exception:
        pass

    # fallback آمن
    return 4096


def build_prompt(user_note: str, text: str) -> str:
    """
    يبني prompt بشكل Chat Template إن كان متوفر، وإلا نص عادي.
    """
    system_prompt = (
        "أنت مساعد متخصص في تلخيص النصوص بطريقة احترافية.\n"
        "قواعد التلخيص:\n"
        "- أعطني الملخص مباشرة فقط بدون مقدمات أو إضافات\n"
        "- لا تكتب \"أرجو\"، \"يرجى\"، \"ملاحظة\"، أو أي جمل إضافية\n"
        "- الملخص يجب أن يكون واضحاً ومباشراً\n"
        "- إذا كان المستخدم طلب صيغة معينة، التزم بها تماماً"
    )

    user_msg = (
        f"طلب المستخدم:\n{user_note}\n\n"
        f"النص المراد تلخيصه:\n{text}\n\n"
        "الملخص:"
    )

    # لو tokenizer تدعم chat template: هذا أفضل بكثير لنماذج Instruct
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            return prompt
        except Exception:
            pass  # نرجع للفallback

    # fallback نصّي
    return f"{system_prompt}\n\n{user_msg}"


def split_text_by_tokens(text: str, token_budget: int) -> List[str]:
    """
    تقسيم النص حسب عدد التوكنات (بدون special tokens).
    """
    if tokenizer is None:
        # fallback بدائي (نادرًا نحتاجه)
        words = text.split()
        step = max(1, token_budget // 2)
        return [" ".join(words[i:i + step]) for i in range(0, len(words), step)]

    ids = tokenizer(text, add_special_tokens=False).input_ids
    chunks = []
    for i in range(0, len(ids), token_budget):
        chunk_ids = ids[i:i + token_budget]
        chunks.append(tokenizer.decode(chunk_ids, skip_special_tokens=True))
    return chunks


def generate_summary_once(text: str, note: str, max_new_tokens: int) -> str:
    """
    توليد ملخص لقطعة واحدة.
    """
    if summarization_pipeline is None or tokenizer is None:
        return "خطأ: لم يتم تحميل النموذج"

    prompt = build_prompt(note, text)

    # عشان نستخرج الناتج بدون لعب "split('الملخص:')"
    prompt_len = len(prompt)

    with torch.inference_mode():
        out = summarization_pipeline(
            prompt,
            max_new_tokens=int(max_new_tokens),
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated = out[0]["generated_text"]

    # لو رجّع النص كامل مع البرومبت، نقصّه
    if isinstance(generated, str) and len(generated) >= prompt_len and generated[:prompt_len] == prompt:
        return generated[prompt_len:].strip()

    # fallback
    return generated.strip() if isinstance(generated, str) else str(generated)


def summarize_text(text: str, note: str, max_new_tokens: int = 150) -> str:
    """
    تلخيص مع دعم التقطيع إذا تجاوز السياق.
    """
    if summarization_pipeline is None or tokenizer is None:
        return "خطأ: لم يتم تحميل النموذج"

    context_limit = _model_context_limit()

    # نترك هامش لأوامر النظام/اليوزر + توكنات التوليد
    safety_margin = 512
    token_budget_for_text = max(256, context_limit - safety_margin - int(max_new_tokens))

    # قياس توكنات النص
    try:
        text_tokens = len(tokenizer(text, add_special_tokens=False).input_ids)
    except Exception:
        text_tokens = len(text.split())  # fallback

    if text_tokens <= token_budget_for_text:
        return generate_summary_once(text, note, max_new_tokens)

    # إذا طويل: قطّع وَلخّص أجزاء ثم لخص ملخصات
    chunks = split_text_by_tokens(text, token_budget_for_text)
    summaries = []

    for i, chunk in enumerate(chunks, start=1):
        print(f"معالجة الجزء {i}/{len(chunks)}...")
        try:
            part = generate_summary_once(chunk, note, max_new_tokens=max_new_tokens)
            if part:
                summaries.append(part)
        except Exception as e:
            print(f"خطأ في الجزء {i}: {e}")

    if not summaries:
        return "خطأ: لم يتمكن من معالجة الأجزاء"

    merged = " ".join(summaries).strip()

    # لو طلع طويل، رجّع تلخيص نهائي أقصر
    final_max_new_tokens = max(80, int(max_new_tokens))
    try:
        return generate_summary_once(merged, "لخص التالي كملخص نهائي موحد ومختصر جداً", final_max_new_tokens)
    except Exception:
        return merged


# =========================
# API
# =========================
@app.route("/api/summarize", methods=["POST"])
def summarize_api():
    """
    Expected JSON:
    {
        "text": "...",
        "note": "...",
        "max_length": 150   # (هنا نستخدمها كـ max_new_tokens)
    }
    """
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"status": "error", "message": "لم يتم إرسال بيانات JSON"}), 400

        text = (data.get("text") or "").strip()
        note = (data.get("note") or "قم بتلخيص النص التالي بطريقة مختصرة ومفيدة").strip()

        # max_length عندك هو فعليًا max_new_tokens (عدد توكنات التوليد)
        max_length = data.get("max_length", 150)
        try:
            max_length = int(max_length)
        except Exception:
            max_length = 150

        if not text:
            return jsonify({"status": "error", "message": "النص مفقود أو فارغ"}), 400

        summary = summarize_text(text, note, max_new_tokens=max_length)

        return jsonify({
            "status": "success",
            "original_text": text,
            "note": note,
            "summary": summary,
            "text_length_chars": len(text),
            "summary_length_chars": len(summary),
            "device": DEVICE
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "message": f"خطأ في معالجة الطلب: {str(e)}"}), 500


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": summarization_pipeline is not None,
        "device": DEVICE
    }), 200


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "مرحباً بك في API التلخيص",
        "endpoints": {
            "/api/summarize": "POST - لتلخيص النصوص",
            "/health": "GET - فحص صحة الخادم"
        },
        "example": {
            "endpoint": "/api/summarize",
            "method": "POST",
            "body": {
                "text": "النص المراد تلخيصه",
                "note": "قم بتلخيص النص بشكل مختصر",
                "max_length": 150
            }
        }
    }), 200


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("🚀 تشغيل API التلخيص")
    print("=" * 50)
    print("📍 الرابط: http://localhost:5001")
    print("📝 لتلخيص النص: POST http://localhost:5001/api/summarize")
    print("💚 فحص الصحة: GET http://localhost:5001/health")
    print("=" * 50 + "\n")

    app.run(
        host="0.0.0.0",
        port=5001,
        debug=True,
        use_reloader=False
    )
