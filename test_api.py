import requests
import json

# رابط الخادم
BASE_URL = "http://localhost:5001"

def test_health():
    """اختبار فحص صحة الخادم"""
    print("\n" + "="*50)
    print("اختبار فحص الصحة")
    print("="*50)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")
    return response.status_code == 200


def test_summarize():
    """اختبار تلخيص النص"""
    print("\n" + "="*50)
    print("اختبار تلخيص النص")
    print("="*50)
    
    # نص العينة (يمكنك تغييره)
    sample_text = """
    الذكاء الاصطناعي هو فرع من فروع علوم الحاسوب يهتم بإنشاء آلات وأنظمة قادرة على أداء مهام تتطلب عادة ذكاء بشري. 
    تشمل هذه المهام التعلم من التجارب والتعرف على الأنماط والفهم اللغوي والقدرة على اتخاذ القرارات. 
    لقد أحرز الذكاء الاصطناعي تقدماً كبيراً في السنوات الأخيرة في تطبيقات متعددة مثل معالجة اللغة الطبيعية 
    والرؤية الحاسوبية والألعاب والروبوتات.
    """
    
    payload = {
        "text": sample_text,
        "note": "قم بتلخيص النص التالي في جملة أو جملتين مختصرة تحافظ على المعنى الأساسي",
        "max_length": 150
    }
    
    print(f"Payload:\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n")
    
    response = requests.post(
        f"{BASE_URL}/api/summarize",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    result = response.json()
    print(f"Response:\n{json.dumps(result, ensure_ascii=False, indent=2)}")
    
    return response.status_code == 200


def test_summarize_with_custom_prompt():
    """اختبار التلخيص مع برومت مخصص"""
    print("\n" + "="*50)
    print("اختبار التلخيص مع برومت مخصص")
    print("="*50)
    
    sample_text = """
    التعليم الإلكتروني أصبح ضرورة حتمية في عالمنا الحديث. مع انتشار الإنترنت والتكنولوجيا، 
    تطورت طرق التعليم بشكل كبير. المنصات التعليمية الرقمية توفر فرصة الوصول إلى المعرفة من أي مكان وفي أي وقت.
    """
    
    payload = {
        "text": sample_text,
        "note": "قم بتلخيص النص بشكل نقاطي يركز على الفوائد الرئيسية",
        "max_length": 100
    }
    
    print(f"Payload:\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n")
    
    response = requests.post(
        f"{BASE_URL}/api/summarize",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    result = response.json()
    print(f"Response:\n{json.dumps(result, ensure_ascii=False, indent=2)}")
    
    return response.status_code == 200


def test_empty_text():
    """اختبار معالجة النص الفارغ"""
    print("\n" + "="*50)
    print("اختبار معالجة النص الفارغ")
    print("="*50)
    
    payload = {
        "text": "",
        "note": "قم بالتلخيص"
    }
    
    response = requests.post(
        f"{BASE_URL}/api/summarize",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    result = response.json()
    print(f"Response:\n{json.dumps(result, ensure_ascii=False, indent=2)}")
    
    return response.status_code == 400


if __name__ == "__main__":
    print("\n" + "🧪 بدء اختبار API التلخيص" + "\n")
    
    try:
        # اختبار فحص الصحة
        if test_health():
            print("✓ فحص الصحة نجح")
        else:
            print("✗ فحص الصحة فشل")
        
        # اختبار التلخيص الأساسي
        if test_summarize():
            print("✓ اختبار التلخيص الأساسي نجح")
        else:
            print("✗ اختبار التلخيص الأساسي فشل")
        
        # اختبار التلخيص مع برومت مخصص
        if test_summarize_with_custom_prompt():
            print("✓ اختبار البرومت المخصص نجح")
        else:
            print("✗ اختبار البرومت المخصص فشل")
        
        # اختبار معالجة الأخطاء
        if test_empty_text():
            print("✓ اختبار معالجة الأخطاء نجح")
        else:
            print("✗ اختبار معالجة الأخطاء فشل")
        
        print("\n" + "="*50)
        print("✓ اكتملت جميع الاختبارات")
        print("="*50 + "\n")
    
    except ConnectionError:
        print("\n✗ خطأ: لا يمكن الاتصال بالخادم")
        print("تأكد من تشغيل app.py أولاً\n")
    except Exception as e:
        print(f"\n✗ خطأ: {str(e)}\n")
