# توافق الأصول والمكتبات والنسب العلمي

## الأصول المستخدمة

يوثق الجدول فحص بيانات وصفية عامة بتاريخ 2026-11-06، ولا يمثل تنزيلًا أو قياسًا جديدًا للأوزان في إعداد التسليم.

| الأصل | المصدر | الاستنتاج وحدوده |
|---|---|---|
| AraT5v2 | [بطاقة النموذج](https://huggingface.co/UBC-NLP/AraT5v2-base-1024) و[الإعدادات](https://huggingface.co/UBC-NLP/AraT5v2-base-1024/blob/main/config.json) | بنية T5 Encoder–Decoder وT5Tokenizer، بسياق 1024 token؛ توافق مع AutoModelForSeq2SeqLM |
| صيغة النموذج الأساسي | [قائمة الملفات](https://huggingface.co/UBC-NLP/AraT5v2-base-1024/tree/main) | سُجل ملف pytorch_model.bin بحجم يقارب 1.47 GB دون model.safetensors في الفحص الأول؛ تحميل الأساس في تدريب المشروع يستخدم PyTorch ≥2.6 وweights_only=True، بينما يُصدر النموذج المضبوط بصيغة safetensors |
| حقوق الأوزان | [بطاقة المصدر](https://huggingface.co/UBC-NLP/AraT5v2-base-1024/raw/main/README.md) و[مستودع AraT5](https://github.com/UBC-NLP/araT5) | الإتاحة البحثية أو التنزيل العام لا يثبتان ترخيص إعادة توزيع أوزان التصدير؛ لم يُفترض ترخيص غير موثق |
| Arabic XL-Sum | [بطاقة البيانات](https://huggingface.co/datasets/csebuetnlp/xlsum) و[الأرشيف](https://huggingface.co/datasets/csebuetnlp/xlsum/tree/main/data) | الحقول id/url/title/summary/text؛ train/validation/test بأعداد 37,519/4,689/4,689 قبل التنظيف؛ تُقرأ نسخة الأرشيف الرسمية دون تنفيذ dataset script بعيد |
| ترخيص البيانات | [معلومات الترخيص](https://huggingface.co/datasets/csebuetnlp/xlsum#licensing-information) | CC BY-NC-SA 4.0؛ نسب إلى مؤلفي XL-Sum ومصادر BBC، مع الفصل بين حقوق البيانات والأوزان |
| NLI الاختياري | [بطاقة النموذج](https://huggingface.co/MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli) | وُثق خارجيًا MIT وsafetensors وتحميل CPU واختبار عربي محدود؛ غير مفعّل افتراضيًا وليس مصنف تحقق مقوّمًا إنتاجيًا |

## بيئة البرمجيات

الإصدارات المثبتة الحساسة للواجهات: Transformers 4.57.1 وAccelerate 1.11.0 وDatasets 4.3.0 وGradio 5.49.1 وSentencePiece 0.2.1 وTrafilatura 2.0.0. يلزم PyTorch `>=2.6,<3`؛ تستخدم اختبارات CPU المحلية 2.8.0+cpu. تُحفظ نسخة CUDA الموجودة على Colab عبر قيد تثبيت في الدفتر النهائي. قد تتغير الاعتمادات العابرة ضمن الحدود المتوافقة، ولا يدعي المستودع بيئة مغلقة متطابقة على جميع الأجهزة.

بصمات المصدر في دفاتر التجارب تحفظ تاريخ المنهجية، وليست بالضرورة مصدر الواجهة الحالية. يُحمّل الدفتر النهائي `app.py` من فرع التسليم ويسجل البصمة الفعلية. لا يُستبدل النموذج المضبوط تلقائيًا بالنموذج الأساسي.

## المراجع

- Nagoudi وآخرون، [AraT5](https://aclanthology.org/2022.acl-long.47/)، ACL 2022.
- Elmadany وآخرون، [Octopus](https://aclanthology.org/2023.arabicnlp-1.20/)، ArabicNLP 2023.
- Hasan وآخرون، [XL-Sum](https://aclanthology.org/2021.findings-acl.413/)، Findings of ACL 2021.
