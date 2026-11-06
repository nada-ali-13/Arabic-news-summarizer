"""SSRF-resistant bounded article fetch with DNS-pinned direct connections.

Every redirect is revalidated. No cookies, environment proxies, retries, scripts,
subresources or automatic redirects. TLS uses the original hostname for SNI and
certificate verification while connecting only to an already validated numeric IP.
"""

import ipaddress
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin, urlsplit, urlunsplit

from .text import normalize


class ArticleError(ValueError):
    pass


MAX_BYTES = 2_000_000
_DNS_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="article-dns")


def resolve_url(url, resolver=None, dns_timeout=4):
    try:
        if not isinstance(url, str) or len(url) > 2048 or any(ord(c) <= 32 for c in url):
            raise ValueError()
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError()
        if parsed.username is not None or parsed.password is not None:
            raise ValueError()
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if port != (443 if parsed.scheme == "https" else 80):
            raise ValueError()
        host = parsed.hostname.rstrip(".").encode("idna").decode("ascii")
        if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
            raise ValueError()
        if resolver is None:
            lookup = _DNS_POOL.submit(socket.getaddrinfo, host, port, type=socket.SOCK_STREAM)
            try:
                answers = lookup.result(timeout=dns_timeout)
            finally:
                lookup.cancel()
        else:
            answers = resolver(host, port, type=socket.SOCK_STREAM)
        addresses = sorted({info[4][0] for info in answers})
        if not addresses:
            raise ValueError()
        for address in addresses:
            ip = ipaddress.ip_address(address)
            if (
                not ip.is_global
                or ip.is_multicast
                or ip.is_reserved
                or getattr(ip, "ipv4_mapped", None)
            ):
                raise ValueError()
            if ip.version == 6 and (
                ip.sixtofour
                or ip.teredo
                or ip in ipaddress.ip_network("64:ff9b::/96")
                or ip in ipaddress.ip_network("64:ff9b:1::/48")
            ):
                raise ValueError()
        authority = f"[{host}]" if ":" in host else host
        clean = urlunsplit((parsed.scheme, authority, parsed.path or "/", parsed.query, ""))
        return clean, parsed.scheme, host, port, addresses[0]
    except (ValueError, UnicodeError, OSError) as exc:
        raise ArticleError(
            "الرابط غير صالح أو يشير إلى وجهة محظورة. الصق نص المقال مباشرة."
        ) from exc


def pinned_pool(scheme, host, port, address):
    import ssl

    import urllib3

    if scheme == "https":
        return urllib3.HTTPSConnectionPool(
            address,
            port=port,
            maxsize=1,
            cert_reqs=ssl.CERT_REQUIRED,
            assert_hostname=host,
            server_hostname=host,
        )
    return urllib3.HTTPConnectionPool(address, port=port, maxsize=1)


def fetch_html(url, resolver=None, pool_factory=None):
    import urllib3

    pool_factory = pool_factory or pinned_pool
    started = time.monotonic()
    for _ in range(4):
        remaining = 20 - (time.monotonic() - started)
        if remaining <= 0:
            raise ArticleError("انتهت مهلة التحميل. الصق النص مباشرة.")
        clean, scheme, host, port, address = resolve_url(url, resolver, min(4, remaining))
        remaining = 20 - (time.monotonic() - started)
        if remaining <= 0:
            raise ArticleError("انتهت مهلة التحميل. الصق النص مباشرة.")
        parsed = urlsplit(clean)
        target = urlunsplit(("", "", parsed.path, parsed.query, ""))
        pool = pool_factory(scheme, host, port, address)
        response = None
        try:
            response = pool.urlopen(
                "GET",
                target,
                headers={
                    "Host": f"[{host}]" if ":" in host else host,
                    "User-Agent": "ArabicNewsSummarizer/1.0",
                    "Accept": "text/html,text/plain",
                    "Accept-Encoding": "identity",
                },
                timeout=urllib3.Timeout(
                    connect=min(4, remaining), read=min(4, remaining), total=min(12, remaining)
                ),
                retries=False,
                redirect=False,
                preload_content=False,
                decode_content=False,
            )
            if response.status in {301, 302, 303, 307, 308}:
                location = response.headers.get("Location")
                if not location:
                    raise ArticleError("تحويل رابط غير صالح. الصق النص مباشرة.")
                url = urljoin(clean, location)
                continue
            if response.status != 200:
                raise ArticleError("تعذر فتح المقال؛ قد يتطلب اشتراكًا. الصق النص مباشرة.")
            content_type = response.headers.get("Content-Type", "").lower()
            if not any(
                t in content_type for t in ("text/html", "application/xhtml+xml", "text/plain")
            ):
                raise ArticleError("الرابط لا يعرض صفحة مقال نصية.")
            if response.headers.get("Content-Encoding", "identity").lower() not in {"", "identity"}:
                raise ArticleError("تعذر قراءة ترميز الصفحة. الصق النص مباشرة.")
            try:
                if int(response.headers.get("Content-Length", "0")) > MAX_BYTES:
                    raise ArticleError("حجم الصفحة كبير جدًا. الصق نص المقال فقط.")
            except ValueError as exc:
                if isinstance(exc, ArticleError):
                    raise
                raise ArticleError("استجابة موقع غير صالحة.") from exc
            body = bytearray()
            for chunk in response.stream(16_384, decode_content=False):
                body.extend(chunk)
                if len(body) > MAX_BYTES or time.monotonic() - started > 20:
                    raise ArticleError("تجاوزت الصفحة حد الحجم أو مهلة التحميل. الصق النص مباشرة.")
            return bytes(body)
        except ArticleError:
            raise
        except (urllib3.exceptions.HTTPError, OSError) as exc:
            raise ArticleError("تعذر الاتصال بالموقع بأمان. الصق النص مباشرة.") from exc
        finally:
            if response is not None:
                response.close()
            pool.close()
    raise ArticleError("تحويلات كثيرة في الرابط. الصق النص مباشرة.")


def extract_article(url, **kwargs):
    import trafilatura

    html = fetch_html(url, **kwargs)
    text = trafilatura.extract(
        html, include_comments=False, include_tables=False, favor_precision=True
    )
    text = normalize(text or "")
    if len(text.split()) < 30:
        raise ArticleError(
            "لم نستطع استخراج مقال كافٍ؛ قد تعتمد الصفحة على JavaScript. الصق النص مباشرة."
        )
    return text
