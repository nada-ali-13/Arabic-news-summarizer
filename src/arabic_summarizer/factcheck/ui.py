"""Concise Arabic cards for published reviews; escape all external content."""
from html import escape

from .demo import LABEL, load_records
from .evidence import relevance, safe_url

_STYLES = """<style>
.kh-check {direction:rtl;text-align:right;color:#18244a;font-family:inherit;}
.kh-check * {box-sizing:border-box;}
.kh-check h3 {font-size:23px;font-weight:750;color:#18244a;margin:0 0 18px;}
.kh-check .kh-card {background:#fff;border:1px solid #dddff0;border-radius:18px;
 padding:24px;margin:0 0 16px;box-shadow:0 4px 18px rgba(26,35,88,.05);}
.kh-check .kh-badge {display:inline-block;background:#eeecff;color:#37308c;
 border:1px solid #c8c3f4;border-radius:14px;padding:10px 24px;font-size:30px;
 font-weight:800;line-height:1.5;max-width:100%;overflow-wrap:anywhere;}
.kh-check .kh-neutral {background:#f1f5f9;color:#334155;border-color:#cbd5e1;
 font-size:21px;padding:12px 18px;}
.kh-check .kh-claim {font-size:18px;line-height:1.9;margin:20px 0 10px;
 color:#18244a;overflow-wrap:anywhere;}
.kh-check .kh-message {color:#475569;font-size:15px;line-height:1.8;margin:8px 0;}
.kh-check .kh-publisher {color:#334155;font-size:15px;margin:16px 0 12px;}
.kh-check .kh-link {display:inline-block;background:#4338ca;color:#fff!important;
 text-decoration:underline;border-radius:10px;padding:11px 18px;font-size:15px;
 font-weight:650;line-height:1.7;}
.kh-check .kh-link:hover {background:#3730a3;}
.kh-check .kh-link:focus-visible,.kh-check summary:focus-visible {
 outline:3px solid #4338ca;outline-offset:4px;}
.kh-check .kh-disclaimer {color:#475569;font-size:13px;line-height:1.9;
 margin:18px 0 0;padding-top:14px;border-top:1px solid #e2e8f0;}
.kh-check .kh-demo {background:#f1f5f9;border-radius:12px;padding:14px 18px;
 color:#334155;font-size:14px;line-height:1.9;margin-bottom:16px;}
.kh-check details {color:#475569;font-size:13px;line-height:1.9;margin-top:14px;}
.kh-check summary {cursor:pointer;color:#37308c;}
.kh-check .kh-counts {display:flex;flex-wrap:wrap;gap:10px;margin:0 0 16px;}
.kh-check .kh-counts span {background:#f1f5f9;border-radius:8px;padding:6px 12px;
 color:#334155;font-size:14px;}
.kh-check .kh-row {border:1px solid #dddff0;border-radius:12px;padding:14px;
 margin:10px 0;background:#fff;overflow-wrap:anywhere;}
.kh-check .kh-preview {font-size:15px;line-height:1.8;margin:0 0 8px;}
.kh-check .kh-row .kh-badge {font-size:21px;padding:4px 14px;border-radius:9px;}
.kh-check .kh-row .kh-publisher {margin:8px 0;font-size:14px;}
.kh-check .kh-row .kh-link {padding:6px 12px;font-size:14px;}
.kh-check .kh-unmatched {color:#475569;font-size:14px;}
.kh-check .kh-claim-list > summary {font-size:16px;font-weight:650;padding:8px 0;}
.kh-check .kh-full {margin:6px 0;font-size:14px;}
.kh-check .kh-full p {white-space:pre-wrap;overflow-wrap:anywhere;}
@media(max-width:640px) {
 .kh-check .kh-card {padding:18px;}
 .kh-check h3 {font-size:21px;}
 .kh-check .kh-badge {font-size:27px;}
 .kh-check .kh-neutral {font-size:19px;}
 .kh-check .kh-link {display:block;text-align:center;}
}
</style>"""


def _card(claim, review=None, saved=False):
    out = ['<article class="kh-card">']
    if review:
        # One consistent badge color: never infer true/false from a rating string.
        out.append(f'<div class="kh-badge">{escape(review["rating"])}</div>')
    else:
        out.append('<div class="kh-badge kh-neutral">لم تتوفر معلومات كافية للتحقق</div>')
    if claim:
        out.append(f'<p class="kh-claim">{escape(claim)}</p>')
    if review:
        message = ("تقرير تدقيق محفوظ لهذا الادعاء." if saved
                   else "عُثر على تقرير تدقيق مطابق لهذا الادعاء.")
        out.append(f'<p class="kh-message">{message}</p>')
        out.append(f'<p class="kh-publisher">جهة التدقيق: <strong>{escape(review["publisher"])}</strong></p>')
        out.append(f'<a class="kh-link" href="{escape(review["url"], quote=True)}" '
                   'target="_blank" rel="noopener noreferrer">عرض تقرير التحقق الأصلي</a>')
        out.append('<p class="kh-disclaimer">هذا تقييم جهة التدقيق المنشور، '
                   'وليس حكمًا مستقلًا من نموذج الذكاء الاصطناعي.</p>')
    else:
        out.append('<p class="kh-message">لم نعثر على تقرير تدقيق مناسب لهذا الادعاء. '
                   'هذا لا يعني أنه صحيح أو خاطئ.</p>')
    return "".join(out) + "</article>"


def _matching_reviews(result):
    # Defensive presentation check; upstream search already ranks relevant reviews.
    return [{"rating": e.rating, "publisher": e.publisher, "url": e.url}
            for e in result.evidence if e.kind == "factcheck"
            and e.rating.strip() and safe_url(e.url)
            and relevance(result.claim.text, e.text)]


def _preview(claim):
    preview = claim if len(claim) <= 140 else claim[:137].rsplit(" ", 1)[0] + "…"
    return f'<p class="kh-preview">{escape(preview)}</p>'


def _review_line(review):
    return (f'<div class="kh-badge">{escape(review["rating"])}</div>'
            f'<p class="kh-publisher">جهة التدقيق: <strong>{escape(review["publisher"])}</strong></p>'
            f'<a class="kh-link" href="{escape(review["url"], quote=True)}" '
            'target="_blank" rel="noopener noreferrer">عرض تقرير التحقق الأصلي</a>')


def render(results, notes):
    out = [_STYLES, '<section class="kh-check" dir="rtl" aria-label="نتيجة التحقق من الخبر">',
           '<h3 style="font-size:23px;font-weight:750;line-height:1.6;margin:0 0 18px">نتيجة التحقق من الخبر</h3>']
    if len(results) <= 1:
        if results:
            reviews = _matching_reviews(results[0])
            out.append(_card(results[0].claim.text, reviews[0] if reviews else None))
            for review in reviews[1:]:
                out.append('<div class="kh-row">' + _review_line(review) + '</div>')
        else:
            out.append(_card(""))
        return "".join(out) + "</section>"

    reviews_by_claim = [_matching_reviews(result) for result in results]
    matched = sum(bool(reviews) for reviews in reviews_by_claim)
    count = len(results)
    out.append('<article class="kh-card kh-article-summary">'
               '<div class="kh-counts">'
               f'<span>الادعاءات المفحوصة: <strong>{count}</strong></span>'
               f'<span>ادعاءات لها تقارير مطابقة: <strong>{matched}</strong></span></div>')
    if not matched:
        out.append('<div class="kh-badge kh-neutral">لم تتوفر تقارير تدقيق كافية</div>'
                   f'<p class="kh-message">فُحصت {count} ادعاءات من المقال، ولم يُعثر على '
                   'تقارير تدقيق مطابقة لها. هذا لا يعني أن الخبر صحيح أو خاطئ.</p>')
    else:
        out.append('<p class="kh-message">عُثر على تقارير تدقيق لبعض الادعاءات المفحوصة أدناه.</p>')
    out.append('<p class="kh-message">هذه النتيجة تخص الادعاءات المستخرجة التي فُحصت فقط، '
               'وليست حكمًا على المقال كاملًا.</p></article>')
    # Published matches remain visible even when other claims have no reviews.
    for result, reviews in zip(results, reviews_by_claim):
        for review in reviews:
            out.append('<article class="kh-row kh-match">' + _preview(result.claim.text)
                       + _review_line(review) + '</article>')
    if matched:
        out.append('<p class="kh-disclaimer">هذا تقييم جهة التدقيق المنشور، '
                   'وليس حكمًا مستقلًا من نموذج الذكاء الاصطناعي.</p>')
    out.append('<details class="kh-claim-list"><summary>عرض تفاصيل الادعاءات</summary>')
    for index, (result, reviews) in enumerate(zip(results, reviews_by_claim), 1):
        claim = result.claim.text
        out.append(f'<article class="kh-row"><strong>الادعاء {index}</strong>' + _preview(claim))
        if len(claim) > 140:
            out.append('<details class="kh-full"><summary>عرض الادعاء كاملًا</summary>'
                       f'<p>{escape(claim)}</p></details>')
        if reviews:
            out.extend(_review_line(review) for review in reviews)
        else:
            out.append('<span class="kh-unmatched">لم يُعثر على تقرير تدقيق مطابق.</span>')
        out.append('</article>')
    out.append('</details></section>')
    return "".join(out)


def demo_html():
    records = load_records()
    out = [_STYLES, '<section class="kh-check" dir="rtl">',
           '<h3 style="font-size:23px;font-weight:750;line-height:1.6;margin:0 0 18px">نتيجة التحقق من الخبر</h3>',
           f'<div class="kh-demo"><strong>{LABEL}</strong><br>'
           'سجل عام محفوظ سابقًا، وليس بحثًا مباشرًا ولا تحققًا من النص المُدخل.</div>']
    if not records:
        out.append(_card(""))
    for record in records:
        out.append(_card(record["text"], record, saved=True))
        out.append('<details><summary>تفاصيل إضافية</summary>'
                   f'<p>تاريخ استرجاع السجل في اختبار المستخدم: {escape(record["retrieved_at"])}</p>'
                   '</details>')
    return "".join(out) + "</section>"
