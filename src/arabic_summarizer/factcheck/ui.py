"""Escape every external field. Ratings stay separate from model assessments."""
from html import escape

from .demo import LABEL, load_records
from .evidence import safe_url


def render(results, notes):
    out = ['<section dir="rtl"><h3>التحقق من الخبر</h3>',
           '<p>هذه علاقة الأدلة بكل ادعاء، وليست حكمًا على صحة المقال كاملًا. '
           'التطابق آلي ومحافظ؛ راجع هوية الأشخاص والتاريخ والسياق. '
           'تكرار التغطية لا يثبت استقلال المصادر.</p>']
    out.extend(f"<p>{escape(note)}</p>" for note in notes)
    for result in results:
        out.append(f"<h4>الادعاء المستخرج: {escape(result.claim.text)}</h4>")
        reviews = [e for e in result.evidence if e.kind == "factcheck"]
        if reviews:
            out.append("<p><strong>عُثر على تقرير تدقيق مطابق للادعاء</strong> — "
                       "راجع تقييم الجهة الأصلية ومصدره أدناه.</p>")
        else:
            out.append("<p>لم يُعثر على تقرير تدقيق مطابق ضمن النتائج المسترجعة.</p>")
        out.append(f"<p>التقييم الآلي المستقل للأدلة (NLI): <b>{result.verdict.value}</b></p>")
        if reviews and not result.relations:
            out.append("<p>لم يُجرَ تقييم NLI مستقل لهذا التقرير. «أدلة غير كافية» "
                       "هنا تخص تقييمنا الآلي، ولا تلغي وجود تقرير التدقيق المنشور.</p>")
        out.append(f"<p>موضع الجملة الأصلية: {result.claim.start}–{result.claim.end}</p>")
        for evidence in result.evidence:
            url = safe_url(evidence.url)
            if not url:
                continue
            date_label = "تاريخ المراجعة" if evidence.kind == "factcheck" else "رصد GDELT (ليس تاريخ النشر المؤكد)"
            out.append(f'<p><a href="{escape(url, quote=True)}" target="_blank" '
                       f'rel="noopener noreferrer">{escape(evidence.publisher)}</a> — '
                       f'{date_label}: {escape(evidence.date or "تاريخ غير متاح")}</p>')
            if evidence.rating:
                out.append(f"<p>التقييم المنشور لدى جهة التدقيق (ليس حكم نموذجنا): {escape(evidence.rating)}</p>")
            if evidence.claim_date:
                out.append(f"<p>تاريخ الادعاء الأصلي: {escape(evidence.claim_date)}</p>")
            out.append(f"<blockquote>{escape(evidence.text or evidence.title)}</blockquote>")
            if not evidence.text:
                out.append("<p>عنوان تغطية فقط؛ لم يُستخدم دليلًا للحكم.</p>")
        for url, relation in result.relations:
            out.append(f"<p>علاقة NLI: {escape(relation)} — {escape(url)}</p>")
        out.extend(f"<p>{escape(note)}</p>" for note in dict.fromkeys(result.notes))
    return "".join(out) + "</section>"


def demo_html():
    records = load_records()
    out = f"<section dir='rtl'><h3>{LABEL}</h3>"
    out += ("<p>سجل عام استُرجع سابقًا في اختبار المستخدم؛ ليس نتيجة API مباشرة، "
            "ولا تحققًا من النص المُدخل. لم يُجرَ تقييم NLI مستقل لهذا السجل.</p>")
    if not records:
        out += "<p>لا توجد سجلات عامة موثّقة محفوظة حاليًا. لم يُنفذ بحث مباشر.</p>"
    for record in records:
        out += (f'<p>الادعاء المحفوظ: {escape(record["text"])}<br>'
                f'التقييم المنشور لدى جهة التدقيق: <strong>{escape(record["rating"])}</strong><br>'
                f'<a href="{escape(record["url"], quote=True)}" target="_blank" rel="noopener noreferrer">'
                f'{escape(record["publisher"])}</a> — استرجاع: '
                f'{escape(record["retrieved_at"])}</p>')
    return out + "</section>"
