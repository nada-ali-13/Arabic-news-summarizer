import socket

import pytest

from arabic_summarizer.articles import ArticleError, extract_article, fetch_html, resolve_url


def resolver(host, port, **kwargs):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com",
        "https://localhost",
        "http://127.0.0.1",
        "http://[::1]",
        "http://169.254.169.254",
        "https://u:p@example.com",
        "https://example.com:8080",
        "http://internal.local",
        "https://example.com\nattack",
    ],
)
def test_invalid_or_private_urls(url):
    def actual_addresses(host, port, **kwargs):
        # Test private literals without sending a network request.
        return [(2, 1, 6, "", (host, port))]

    with pytest.raises(ArticleError):
        resolve_url(url, actual_addresses)


@pytest.mark.parametrize(
    "address",
    [
        "10.0.0.1",
        "172.16.0.1",
        "192.168.1.1",
        "0.0.0.0",
        "100.64.0.1",
        "224.0.0.1",
        "::ffff:127.0.0.1",
        "fe80::1",
    ],
)
def test_dns_private_and_mixed_answers(address):
    def mixed(host, port, **kwargs):
        return resolver(host, port) + [(2, 1, 6, "", (address, port))]

    with pytest.raises(ArticleError):
        resolve_url("https://example.com", mixed)


class Response:
    def __init__(self, status=200, headers=None, body=b"<html>test</html>"):
        self.status = status
        self.headers = headers or {"Content-Type": "text/html"}
        self.body = body
        self.closed = False

    def stream(self, *_args, **_kwargs):
        yield self.body

    def close(self):
        self.closed = True


class Pool:
    def __init__(self, response, calls):
        self.response, self.calls = response, calls

    def urlopen(self, method, target, **kwargs):
        self.calls.append((method, target, kwargs))
        return self.response

    def close(self):
        pass


def test_connection_is_pinned_to_validated_ip_and_host_preserved():
    calls, factories = [], []
    response = Response()

    def factory(*args):
        factories.append(args)
        return Pool(response, calls)

    assert fetch_html("https://example.com/a?q=1#fragment", resolver, factory) == response.body
    assert factories == [("https", "example.com", 443, "93.184.216.34")]
    assert calls[0][1] == "/a?q=1"
    assert calls[0][2]["headers"]["Host"] == "example.com"
    assert calls[0][2]["redirect"] is False and calls[0][2]["retries"] is False
    assert response.closed


def test_tls_uses_original_hostname():
    from arabic_summarizer.articles import pinned_pool

    pool = pinned_pool("https", "example.com", 443, "93.184.216.34")
    assert pool.host == "93.184.216.34"
    assert pool.assert_hostname == "example.com"
    assert pool.conn_kw["server_hostname"] == "example.com"
    pool.close()


def test_redirect_revalidates_dns_and_blocks_rebinding():
    count = 0
    factories = []

    def changing(host, port, **kwargs):
        nonlocal count
        count += 1
        address = "93.184.216.34" if count == 1 else "127.0.0.1"
        return [(2, 1, 6, "", (address, port))]

    def factory(*args):
        factories.append(args)
        return Pool(Response(302, {"Location": "/next"}), [])

    with pytest.raises(ArticleError):
        fetch_html("https://example.com", changing, factory)
    assert len(factories) == 1


def test_redirect_to_metadata_is_blocked():
    def dns(host, port, **kwargs):
        address = "169.254.169.254" if host == "169.254.169.254" else "93.184.216.34"
        return [(2, 1, 6, "", (address, port))]

    with pytest.raises(ArticleError):
        fetch_html(
            "https://example.com",
            dns,
            lambda *args: Pool(Response(302, {"Location": "http://169.254.169.254/latest"}), []),
        )


@pytest.mark.parametrize(
    "response",
    [
        Response(403),
        Response(200, {"Content-Type": "image/png"}),
        Response(200, {"Content-Type": "text/html", "Content-Length": "3000000"}),
        Response(body=b"x" * 2_000_001),
        Response(200, {"Content-Type": "text/html", "Content-Encoding": "gzip"}),
    ],
)
def test_bad_or_oversized_responses(response):
    with pytest.raises(ArticleError):
        fetch_html("https://example.com", resolver, lambda *args: Pool(response, []))


def test_extract_real_arabic_html_fixture():
    article = (
        "أعلنت الجامعة افتتاح مكتبة جديدة تضم قاعات للقراءة ومصادر علمية للباحثين والطلاب. "
        "وتوفر المكتبة خدمات إلكترونية وتدريبًا للطلبة وتفتح أبوابها صباح كل يوم. "
    ) * 8
    html = f"<html lang='ar'><head><title>افتتاح المكتبة</title></head><body><nav>روابط</nav><article><h1>افتتاح المكتبة</h1><p>{article}</p></article></body></html>".encode()
    text = extract_article(
        "https://example.com",
        resolver=resolver,
        pool_factory=lambda *args: Pool(Response(body=html), []),
    )
    assert "أعلنت الجامعة" in text and len(text.split()) > 30


@pytest.mark.parametrize("address", ["2002:7f00:1::", "64:ff9b::7f00:1", "64:ff9b:1::a00:1"])
def test_ipv6_transition_addresses_are_rejected(address):
    def transition(host, port, **kwargs):
        return [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", (address, port, 0, 0))]

    with pytest.raises(ArticleError):
        resolve_url("https://example.com", transition)
