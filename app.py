"""Arabic-first Gradio application. Importing/building never loads model weights."""

import argparse

from arabic_summarizer.articles import ArticleError, extract_article
from arabic_summarizer.inference import Summarizer

CSS = """
.gradio-container {direction: rtl; max-width: 1050px !important; margin: auto;}
textarea, input {direction: rtl; text-align: right;}
#summary textarea {font-size: 19px; line-height: 1.9;}
footer {display: none !important;}
"""
EXAMPLES = [
    "أعلنت الجامعة افتتاح مكتبة جديدة تضم قاعات للمطالعة ومختبرًا رقميًا. "
    "وقالت إدارة الجامعة إن المكتبة ستفتح أبوابها للطلاب الأسبوع المقبل، "
    "وستوفر مصادر إلكترونية وبرامج تدريبية لمساعدة الباحثين في الوصول إلى المعلومات.",
    "أطلقت بلدية المدينة مبادرة لزراعة ألف شجرة في الحدائق والشوارع العامة. "
    "ويشارك في المبادرة طلاب ومتطوعون من الأحياء المختلفة. "
    "وتهدف الحملة إلى زيادة المساحات الخضراء وتحسين جودة الهواء خلال السنوات المقبلة.",
]


def build_app(summarizer=None):
    import gradio as gr

    summarizer = summarizer or Summarizer()

    def submit(text, url, preference):
        try:
            if text and text.strip() and url and url.strip():
                raise ValueError("أدخل النص أو الرابط فقط لتحديد مصدر المقال.")
            source = extract_article(url.strip()) if url and url.strip() else text or ""
            result = summarizer.summarize(source, preference)
            stats = (
                f"الكلمات الأصلية: {result['original_words']} | كلمات الملخص: "
                f"{result['summary_words']} | تقليل الكلمات: {result['compression_percent']}٪ "
                f"| زمن التوليد: {result['seconds']} ثانية"
            )
            return result["summary"], stats, result["warning"]
        except (ValueError, ArticleError) as exc:
            raise gr.Error(str(exc)) from exc
        except Exception as exc:
            raise gr.Error("تعذر التلخيص. تحقق من مسار النموذج والذاكرة ثم أعد المحاولة.") from exc

    with gr.Blocks(css=CSS, title="ملخّص الأخبار العربية", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# ملخّص الأخبار العربية\nتلخيص آلي باستخدام AraT5v2 — مشروع تعلّم عميق")
        gr.Markdown(
            "أدخل مقالًا عربيًا أو رابطه، ثم اختر طول الملخص. راجع الحقائق في المصدر قبل استخدامها."
        )
        with gr.Row():
            with gr.Column(scale=1):
                text = gr.Textbox(label="نص المقال", lines=12, placeholder="الصق المقال هنا…")
                url = gr.Textbox(label="أو رابط المقال", placeholder="https://…")
                preference = gr.Radio(["قصير", "متوسط", "طويل"], value="متوسط", label="طول الملخص")
                button = gr.Button("تلخيص المقال", variant="primary")
            with gr.Column(scale=1):
                summary = gr.Textbox(
                    label="الملخص",
                    lines=12,
                    interactive=False,
                    show_copy_button=True,
                    elem_id="summary",
                )
                stats = gr.Markdown()
                warning = gr.Markdown()
        gr.Examples([[e] for e in EXAMPLES], inputs=[text], label="أمثلة تجريبية مكتوبة للمشروع")
        button.click(
            submit,
            [text, url, preference],
            [summary, stats, warning],
            concurrency_limit=1,
            api_name=False,
        )
        gr.Markdown("قد يخطئ النموذج أو يضيف معلومات غير دقيقة. المقالات الطويلة تُعالج على مقاطع.")
    return demo.queue(max_size=8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Temporary public Colab link")
    args = parser.parse_args()
    build_app().launch(share=args.share)
