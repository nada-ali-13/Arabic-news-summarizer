"""خُلاصة: Arabic Gradio UI adapted from the final handoff; weights load on demand."""

import argparse
import os
from html import escape

from arabic_summarizer.articles import ArticleError, extract_article
from arabic_summarizer.inference import Summarizer

# Configure before the lazy loader imports Transformers.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

CUSTOM_CSS = """
.gradio-container {direction: rtl; font-family:'Segoe UI',Tahoma,Arial,sans-serif!important;
  max-width:1200px!important; margin:auto!important; background:#f8f9fe;}
body {background:#f8f9fe!important;}
#hero {background:linear-gradient(125deg,#172554,#4338ca,#6d5dfc);
  border-radius:22px; padding:42px 35px; color:white; text-align:center;
  margin-bottom:24px; box-shadow:0 12px 30px rgba(67,56,202,.15);}
#hero h1 {font-size:40px; font-weight:800; color:white; margin:4px 0 8px;}
#hero h2 {font-size:25px; font-weight:650; color:white; margin:10px 0 18px;}
#hero p {max-width:700px;margin:0 auto;color:#e0e7ff;font-size:17px;line-height:2;}
#hero .chip {display:inline-block;margin-bottom:12px;padding:7px 16px;
 background:rgba(255,255,255,.14);border:1px solid rgba(255,255,255,.25);
 border-radius:50px;font-size:13px;color:white;}
#input-card,#output-card {background:white;border:1px solid #e9eaf4;
 border-radius:20px;padding:23px;box-shadow:0 5px 22px rgba(26,35,88,.045);}
.section-title {font-size:19px;font-weight:750;color:#18244a;margin-bottom:7px;}
.section-subtitle {color:#7b849d;font-size:13px;margin-bottom:20px;}
#article-box textarea,#url-box input,#summary-box textarea {direction:rtl!important;
 text-align:right!important;line-height:1.9!important;font-size:15px!important;
 border-radius:12px!important;}
#summary-box textarea {font-size:17px!important;color:#192653!important;background:#fafbff!important;}
#submit-btn {background:linear-gradient(100deg,#4338ca,#7064fa)!important;
 color:white!important;border:none!important;border-radius:12px!important;
 font-weight:750!important;min-height:52px!important;font-size:17px!important;}
#clear-btn {border-radius:12px!important;min-height:45px!important;}
.metric-grid {display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin-top:12px;}
.metric-item {text-align:center;background:#f4f5ff;padding:17px 8px;border-radius:12px;border:1px solid #e9eaff;}
.metric-number {font-weight:800;color:#4338ca;font-size:21px;}
.metric-label {color:#77809b;font-size:12px;margin-top:5px;}
.footer-note {text-align:center;color:#838ba3;font-size:13px;padding:20px 10px 4px;line-height:1.9;}
.author-note {text-align:center;padding:0 10px 20px;font-size:13px;color:#64748b;}
@media (max-width:640px) {
  #hero {padding:28px 16px;}
  #hero h1 {font-size:32px;}
  #hero h2 {font-size:21px;}
  #input-card,#output-card {padding:16px;min-width:0!important;flex-basis:100%!important;}
}
"""


def build_app(summarizer=None):
    """Build without loading weights; accept an already loaded notebook summarizer."""
    import gradio as gr

    if summarizer is None:
        summarizer = Summarizer()

    def submit(text, url, length):
        try:
            text = (text or "").strip()
            url = (url or "").strip()
            if text and url:
                raise ValueError("أدخلي نص المقال أو رابطه، وليس الاثنين معًا.")
            if not text and not url:
                raise ValueError("يرجى إدخال نص مقال أو رابط أولًا.")
            article = extract_article(url) if url else text
            result = summarizer.summarize(article, preference=length)
            stats = f"""<div class="metric-grid">
              <div class="metric-item"><div class="metric-number">{int(result["original_words"])}</div><div class="metric-label">كلمات المقال</div></div>
              <div class="metric-item"><div class="metric-number">{int(result["summary_words"])}</div><div class="metric-label">كلمات الملخص</div></div>
              <div class="metric-item"><div class="metric-number">{float(result["compression_percent"]):.1f}%</div><div class="metric-label">نسبة الاختصار</div></div>
            </div><p style="color:#8b91a6;font-size:12px;margin-top:15px">زمن التوليد: {float(result["seconds"]):.2f} ثانية</p>"""
            return result["summary"], stats, escape(result.get("warning", ""))
        except (ValueError, ArticleError) as exc:
            raise gr.Error(str(exc)) from exc
        except Exception as exc:
            print("Summarization error:", repr(exc))
            raise gr.Error("تعذّر التلخيص. تحققي من المقال أو الرابط وحاولي مجددًا.") from exc

    # Import and initialize the optional subsystem only after the user requests it.
    checker = None

    def verify(text, url, demo_mode):
        nonlocal checker
        try:
            from arabic_summarizer.factcheck.ui import demo_html, render

            if demo_mode:
                return demo_html()
            if os.environ.get("KHULASA_FACTCHECK_ENABLED", "1") == "0":
                return "<p>التحقق معطّل؛ يمكنك استخدام التلخيص.</p>"
            text, url = (text or "").strip(), (url or "").strip()
            if bool(text) == bool(url):
                return "<p>أدخل نص المقال أو رابطه فقط.</p>"
            article = extract_article(url) if url else text
            if checker is None:
                from arabic_summarizer.factcheck.pipeline import Pipeline

                checker = Pipeline()
            return render(*checker.verify(article))
        except Exception:
            return "<p>تعذر التحقق؛ يمكنك استخدام التلخيص بصورة مستقلة.</p>"

    with gr.Blocks(
        theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"),
        css=CUSTOM_CSS,
        title="خُلاصة | ملخص الأخبار الذكي",
    ) as demo:
        gr.HTML("""<div id="hero"><div class="chip">✦ اقرأ أقل، واعرف أكثر</div>
          <h1>خُلاصة</h1><h2>الخبر كاملًا؟ خذ خُلاصته.</h2>
          <p>أخبار كثيرة، ووقت قليل؟ الصق المقال أو أضف رابطه، ودع خُلاصة يحوّله إلى ملخص عربي سريع وواضح، بالطول الذي يناسبك.</p></div>""")
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, elem_id="input-card"):
                gr.HTML(
                    '<div class="section-title">إدخال المقال</div><div class="section-subtitle">الصق نص الخبر أو أدخل رابطه لتلخيصه.</div>'
                )
                article_input = gr.Textbox(
                    label="نص المقال",
                    placeholder="اكتب أو الصق المقال الإخباري هنا...",
                    lines=12,
                    max_lines=18,
                    elem_id="article-box",
                )
                url_input = gr.Textbox(
                    label="رابط المقال", placeholder="https://example.com/news", elem_id="url-box"
                )
                length_input = gr.Radio(
                    ["قصير", "متوسط", "طويل"], value="متوسط", label="طول الملخص"
                )
                submit_btn = gr.Button("تلخيص المقال", variant="primary", elem_id="submit-btn")
                clear_btn = gr.Button("مسح الحقول", variant="secondary", elem_id="clear-btn")
            with gr.Column(scale=1, elem_id="output-card"):
                gr.HTML(
                    '<div class="section-title">نتيجة التلخيص</div><div class="section-subtitle">سيظهر الملخص الناتج هنا بعد المعالجة.</div>'
                )
                summary_output = gr.Textbox(
                    label="الملخص",
                    placeholder="الملخص سيظهر هنا...",
                    lines=16,
                    interactive=False,
                    show_copy_button=True,
                    elem_id="summary-box",
                )
                stats_output = gr.HTML()
                warning_output = gr.Markdown()
        with gr.Accordion("التحقق من الخبر — اختياري", open=False):
            gr.Markdown("استخراج الادعاءات والبحث عن أدلة خارجية؛ قد تستغرق العملية نحو دقيقة.")
            demo_mode = gr.Checkbox(label="وضع العرض — نتائج محفوظة وليست تحققًا مباشرًا", value=False)
            verify_btn = gr.Button("التحقق من الخبر", variant="secondary")
            factcheck_output = gr.HTML()
        verify_btn.click(
            fn=verify, inputs=[article_input, url_input, demo_mode],
            outputs=[factcheck_output], concurrency_limit=1, api_name=False,
            show_progress="full",
        )
        submit_btn.click(
            fn=submit,
            inputs=[article_input, url_input, length_input],
            outputs=[summary_output, stats_output, warning_output],
            concurrency_limit=1,
            api_name=False,
        )
        clear_btn.click(
            fn=lambda: ("", "", "", "", ""),
            inputs=[],
            outputs=[article_input, url_input, summary_output, stats_output, warning_output],
            queue=False,
        )
        gr.HTML(
            '<div class="footer-note">يعتمد التطبيق على نموذج AraT5v2 بعد إعادة تدريبه على الأخبار العربية.<p>قد تحتوي الملخصات الآلية على أخطاء واقعية؛ يُنصح بمراجعة المصدر.</p></div>'
        )
        gr.HTML(
            '<div class="author-note">تصميم وتطوير: <strong style="color:#6366f1;">ندى علي</strong> ✦ © 2026</div>'
        )
    return demo.queue(max_size=8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="خُلاصة — ملخص الأخبار العربية")
    parser.add_argument(
        "--share", action="store_true", help="Create a temporary public Gradio link"
    )
    args = parser.parse_args()
    build_app().launch(share=args.share)
