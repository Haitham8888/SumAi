from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, pipeline
import torch
import os

app = Flask(__name__)

# المسارات الخاصة بالنماذج المحلية
MODEL_DIR = "./models_cache"
ALLAM_MODEL_PATH = os.path.join(MODEL_DIR, "models--humain-ai--ALLaM-7B-Instruct-preview/snapshots/a28dd1e67420cde72d3629c8633a974cf7d9c366")
ARABERT_MODEL_PATH = os.path.join(MODEL_DIR, "models--MostafaAhmed98--AraBert-Arabic-NER-CoNLLpp")

# اختيار device (GPU إذا توفر، وإلا CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"استخدام Device: {device}")

# تحميل نموذج ALLaM للتلخيص
print("جاري تحميل نموذج ALLaM...")
try:
    tokenizer = AutoTokenizer.from_pretrained(
        ALLAM_MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True
    )
    # تحميل نموذج Llama مباشرة
    model = LlamaForCausalLM.from_pretrained(
        ALLAM_MODEL_PATH,
        local_files_only=True,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=device,
        low_cpu_mem_usage=True
    )
    summarization_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )
    print("✓ تم تحميل نموذج ALLaM بنجاح")
except Exception as e:
    print(f"✗ خطأ في تحميل نموذج ALLaM: {str(e)}")
    summarization_pipeline = None


def create_summarization_prompt(user_prompt: str, text: str) -> str:
    """
    إنشاء prompt محسّن للتلخيص
    """
    # Prompt محسّن يمنع الرسائل الإضافية
    system_prompt = """أنت مساعد متخصص في تلخيص النصوص بطريقة احترافية.
قواعد التلخيص:
- أعطني الملخص مباشرة فقط بدون مقدمات أو إضافات
- لا تكتب "أرجو"، "يرجى"، "ملاحظة"، أو أي جمل إضافية
- الملخص يجب أن يكون واضحاً ومباشراً
- إذا كان المستخدم طلب صيغة معينة، التزم بها تماماً"""
    
    full_prompt = f"""{system_prompt}

طلب المستخدم: {user_prompt}

النص المراد تلخيصه:
{text}

الملخص:"""
    
    return full_prompt


def summarize_text(text: str, prompt: str, max_length: int = 150) -> str:
    """
    تلخيص النص باستخدام نموذج ALLaM
    
    Args:
        text: النص المراد تلخيصه
        prompt: أوامر/تعليمات التلخيص
        max_length: الحد الأقصى لطول التلخيص
    
    Returns:
        النص الملخص
    """
    if not summarization_pipeline:
        return "خطأ: لم يتم تحميل النموذج"
    
    # الحد الأقصى للتوكنات المدخل (بناءً على config.json)
    max_input_tokens = 3000  # نترك بعض المجال الآمن
    
    # تقدير عدد الكلمات (توكن تقريباً)
    estimated_tokens = len(text.split())
    
    if estimated_tokens > max_input_tokens:
        # تقسيم النص إلى أجزاء
        words = text.split()
        chunk_size = max_input_tokens - 100  # حجم الجزء
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        summaries = []
        for i, chunk in enumerate(chunks):
            print(f"معالجة الجزء {i+1} من {len(chunks)}...")
            
            full_prompt = create_summarization_prompt(prompt, chunk)
            
            try:
                result = summarization_pipeline(
                    full_prompt,
                    max_new_tokens=max_length,
                    num_return_sequences=1,
                    temperature=0.3,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                generated_text = result[0]['generated_text']
                summary = generated_text.split("الملخص:")[-1].strip()
                summaries.append(summary)
            except Exception as e:
                print(f"خطأ في معالجة الجزء {i+1}: {str(e)}")
                continue
        
        # دمج ملخصات الأجزاء
        if summaries:
            final_summary = " ".join(summaries)
            
            # إذا كان ملخص الأجزاء طويل جداً، ملخصه مرة أخرى
            if len(final_summary.split()) > max_length:
                full_prompt = create_summarization_prompt(prompt, final_summary)
                try:
                    result = summarization_pipeline(
                        full_prompt,
                        max_new_tokens=max_length,
                        num_return_sequences=1,
                        temperature=0.3,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    final_summary = result[0]['generated_text'].split("الملخص:")[-1].strip()
                except:
                    pass
            
            return final_summary
        else:
            return "خطأ: لم يتمكن من معالجة الأجزاء"
    
    else:
        # النص قصير - معالجة عادية
        full_prompt = create_summarization_prompt(prompt, text)
        
        try:
            result = summarization_pipeline(
                full_prompt,
                max_new_tokens=max_length,
                num_return_sequences=1,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            generated_text = result[0]['generated_text']
            # استخراج الملخص من النص المُولد
            summary = generated_text.split("الملخص:")[-1].strip()
            return summary
        except Exception as e:
            return f"خطأ في التلخيص: {str(e)}"


@app.route('/api/summarize', methods=['POST'])
def summarize_api():
    """
    API Endpoint لتلخيص النصوص
    
    Expected JSON input:
    {
        "text": "النص المراد تلخيصه",
        "note": "أوامر التلخيص - يجب أن تكون التلخيص مختصر ومفيد",
        "max_length": 150  (اختياري)
    }
    """
    try:
        # التحقق من البيانات المرسلة
        data = request.get_json()
        
        if not data:
            return jsonify({
                "status": "error",
                "message": "لم يتم إرسال بيانات JSON"
            }), 400
        
        text = data.get('text', '').strip()
        note = data.get('note', 'قم بتلخيص النص التالي بطريقة مختصرة ومفيدة').strip()
        max_length = data.get('max_length', 150)
        
        # التحقق من وجود النص
        if not text:
            return jsonify({
                "status": "error",
                "message": "النص مفقود أو فارغ"
            }), 400
        
        if not note:
            note = "قم بتلخيص النص التالي بطريقة مختصرة ومفيدة"
        
        # تلخيص النص
        summary = summarize_text(text, note, max_length)
        
        return jsonify({
            "status": "success",
            "original_text": text,
            "note": note,
            "summary": summary,
            "text_length": len(text),
            "summary_length": len(summary)
        }), 200
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"خطأ في معالجة الطلب: {str(e)}"
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """فحص صحة الخادم"""
    return jsonify({
        "status": "healthy",
        "model_loaded": summarization_pipeline is not None,
        "device": device
    }), 200


@app.route('/', methods=['GET'])
def home():
    """الصفحة الرئيسية"""
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


if __name__ == '__main__':
    # تشغيل الخادم
    print("\n" + "="*50)
    print("🚀 تشغيل API التلخيص")
    print("="*50)
    print("📍 الرابط: http://localhost:5001")
    print("📝 لتلخيص النص: POST http://localhost:5001/api/summarize")
    print("💚 فحص الصحة: GET http://localhost:5001/health")
    print("="*50 + "\n")
    
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=True,
        use_reloader=False
    )
