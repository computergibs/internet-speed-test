"""Tests for speedtest.py.

Tests that make requests start their own local HTTP server,
so no internet connection is needed.
"""

from __future__ import annotations

import contextlib
import io
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import speedtest

PAYLOAD = b"x" * 1_000_000  # 1 MB


class Handler(BaseHTTPRequestHandler):
    """Answers in different ways depending on the requested path."""

    server: Server

    def do_GET(self) -> None:
        self.server.paths.append(self.path)
        if self.path == "/file":
            self.send_body(PAYLOAD, content_length=len(PAYLOAD))
        elif self.path == "/no-length":  # the body ends when the connection closes
            self.send_body(PAYLOAD)
        elif self.path == "/truncated":  # announces more bytes than it sends
            self.send_body(PAYLOAD[:1000], content_length=len(PAYLOAD))
        elif self.path == "/no-response":  # closes the connection without answering
            pass
        elif self.path == "/hang":  # does not answer until the test is over
            self.server.test_finished.wait(timeout=10)
        else:
            self.send_error(404)

    def send_body(self, body: bytes, content_length: int | None = None) -> None:
        self.send_response(200)
        if content_length is not None:
            self.send_header("Content-Length", str(content_length))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        """Do not print a line per request to the test output."""


class Server(ThreadingHTTPServer):
    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), Handler)  # port 0: any free port
        self.paths: list[str] = []  # paths of the received requests, in order
        self.test_finished = threading.Event()


class ServerTestCase(unittest.TestCase):
    """Runs a local HTTP server during each test."""

    def setUp(self) -> None:
        self.server = Server()
        # Short poll interval, so that shutdown() returns quickly.
        thread = threading.Thread(target=self.server.serve_forever, args=(0.01,))
        thread.start()
        # Cleanups run in reverse order: release "/hang", stop, close, join.
        self.addCleanup(thread.join)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.addCleanup(self.server.test_finished.set)

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.server.server_port}{path}"


class DownloadTest(ServerTestCase):
    def test_reads_the_whole_body(self) -> None:
        measurement = speedtest.download(self.url("/file"), timeout=5)
        self.assertEqual(measurement.size_bytes, len(PAYLOAD))
        self.assertGreater(measurement.seconds, 0)

    def test_reads_a_body_without_content_length(self) -> None:
        measurement = speedtest.download(self.url("/no-length"), timeout=5)
        self.assertEqual(measurement.size_bytes, len(PAYLOAD))

    def test_incomplete_body_is_an_error(self) -> None:
        with self.assertRaisesRegex(speedtest.DownloadError, "1,000 of 1,000,000"):
            speedtest.download(self.url("/truncated"), timeout=5)

    def test_http_error_status_is_an_error(self) -> None:
        with self.assertRaisesRegex(speedtest.DownloadError, "HTTP 404"):
            speedtest.download(self.url("/missing"), timeout=5)

    def test_closed_connection_is_an_error(self) -> None:
        with self.assertRaises(speedtest.DownloadError):
            speedtest.download(self.url("/no-response"), timeout=5)

    def test_timeout_is_an_error(self) -> None:
        with self.assertRaisesRegex(speedtest.DownloadError, "timed out"):
            speedtest.download(self.url("/hang"), timeout=0.1)


class MainTest(ServerTestCase):
    def run_main(self, *argv: str) -> tuple[int, str, str]:
        """Run the script with *argv*; return the exit code, stdout and stderr."""
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = speedtest.main(argv)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_makes_ten_requests_and_prints_the_result(self) -> None:
        code, output, _ = self.run_main(self.url("/file"))
        self.assertEqual(code, 0)
        self.assertEqual(self.server.paths, ["/file"] * 10)
        self.assertIn("Total downloaded:      10.00 MB (10,000,000 bytes)", output)
        self.assertRegex(output, r"Average request time:  \d+\.\d{3} s")
        self.assertRegex(output, r"Download speed:        \d+\.\d{2} Mbit/s")

    def test_number_of_requests_can_be_changed(self) -> None:
        code, _, _ = self.run_main(self.url("/file"), "--requests", "3")
        self.assertEqual(code, 0)
        self.assertEqual(len(self.server.paths), 3)

    def test_stops_at_the_first_failed_request(self) -> None:
        code, _, errors = self.run_main(self.url("/missing"))
        self.assertEqual(code, 1)
        self.assertEqual(len(self.server.paths), 1)
        self.assertIn("request 1 of 10 failed: HTTP 404", errors)


class ArgumentsTest(unittest.TestCase):
    def test_invalid_arguments_are_rejected(self) -> None:
        for argv in (
            ["example.com/image.jpg"],  # no scheme
            ["ftp://example.com/image.jpg"],
            ["https://example.com/image.jpg", "--requests", "0"],
            ["https://example.com/image.jpg", "--timeout", "0"],
        ):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as context:
                    speedtest.parse_args(argv)
                self.assertEqual(context.exception.code, 2)

    def test_url_is_percent_encoded(self) -> None:
        self.assertEqual(
            speedtest.http_url("https://example.com/фото 1.jpg?q=я"),
            "https://example.com/%D1%84%D0%BE%D1%82%D0%BE%201.jpg?q=%D1%8F",
        )

    def test_valid_url_is_unchanged(self) -> None:
        url = "https://example.com/a%20b/(c).jpg?x=1&y=2"
        self.assertEqual(speedtest.http_url(url), url)


class SummarizeTest(unittest.TestCase):
    def test_speed_is_in_megabits_per_second(self) -> None:
        # 125,000 bytes = 1,000,000 bits.
        self.assertEqual(speedtest.to_mbps(125_000, 1.0), 1.0)

    def test_summary(self) -> None:
        summary = speedtest.summarize(
            [
                speedtest.Measurement(size_bytes=1_000_000, seconds=0.5),
                speedtest.Measurement(size_bytes=1_000_000, seconds=1.5),
            ]
        )
        self.assertEqual(summary.total_bytes, 2_000_000)
        self.assertAlmostEqual(summary.average_seconds, 1.0)
        # 16 Mbit in 2 s. The mean of per-request speeds, (16 + 5.33) / 2,
        # would give 10.67 and overstate it.
        self.assertAlmostEqual(summary.speed_mbps, 8.0)
