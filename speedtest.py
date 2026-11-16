#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["certifi>=2023.7.22"]
# ///
"""Measure download speed by fetching the same URL several times in a row.

The script sends N sequential GET requests (10 by default), waits until each
response body has been received in full and prints the average request time,
the total amount of downloaded data and the download speed.

Units: speed in Mbit/s (10**6 bits per second), size in MB (10**6 bytes),
time in seconds.
"""

from __future__ import annotations

import argparse
import http.client
import math
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass

import certifi

DEFAULT_REQUESTS = 10
DEFAULT_TIMEOUT = 30.0  # seconds
CHUNK_SIZE = 64 * 1024  # the body is read in chunks instead of all at once

BITS_PER_BYTE = 8
MEGA = 1_000_000  # decimal prefix used for network speeds: 1 Mbit = 10**6 bits

REQUEST_HEADERS = {
    # Some servers reject the default "Python-urllib/3.x" user agent.
    "User-Agent": "speedtest.py/1.0",
    # Get the file as is, without compression, so that we count its real size.
    "Accept-Encoding": "identity",
}

# Check HTTPS certificates against Mozilla's CA bundle, the same on every OS.
# On Windows Python otherwise trusts the system certificate stores as they are,
# and outdated certificates cached there can make valid sites fail to verify.
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())

# Characters that http_url() leaves as is: "%" (already encoded sequences)
# and the characters that have a special meaning in URLs.
URL_SAFE_CHARS = "%:/?#[]@!$&'()*+,;=~"

# Table columns: request number, size, time, speed.
ROW = "{:>3}  {:>10}  {:>9}  {:>14}"


class DownloadError(Exception):
    """A request failed or its response body was not received in full."""


@dataclass(frozen=True)
class Measurement:
    """Result of a single request."""

    size_bytes: int  # size of the response body
    seconds: float  # from sending the request to receiving the last byte


@dataclass(frozen=True)
class Summary:
    """Result of the whole series of requests."""

    total_bytes: int
    average_seconds: float
    speed_mbps: float  # megabits per second


def to_mbps(size_bytes: int, seconds: float) -> float:
    """Convert *size_bytes* transferred in *seconds* to megabits per second."""
    return size_bytes * BITS_PER_BYTE / MEGA / seconds


def download(url: str, timeout: float) -> Measurement:
    """Request *url* once, read the whole response body and time it.

    Raises:
        DownloadError: the request failed or the body was not received in full.
    """
    request = urllib.request.Request(url, headers=REQUEST_HEADERS)
    try:
        started = time.perf_counter()
        with urllib.request.urlopen(
            request, timeout=timeout, context=SSL_CONTEXT
        ) as response:
            expected = response.headers.get("Content-Length")
            size = 0
            while chunk := response.read(CHUNK_SIZE):
                size += len(chunk)
            seconds = time.perf_counter() - started
    except urllib.error.HTTPError as error:  # the server answered with 4xx/5xx
        raise DownloadError(f"HTTP {error.code} {error.reason}") from error
    except urllib.error.URLError as error:  # could not connect or send the request
        raise DownloadError(str(error.reason)) from error
    except (OSError, http.client.HTTPException) as error:  # failed during the response
        raise DownloadError(str(error)) from error

    # read() silently stops if the connection is closed too early,
    # so compare the size with the one announced by the server.
    if expected is not None and expected.isdigit() and size != int(expected):
        raise DownloadError(
            f"connection closed after {size:,} of {int(expected):,} bytes"
        )
    return Measurement(size_bytes=size, seconds=seconds)


def summarize(measurements: Sequence[Measurement]) -> Summary:
    """Combine the results of single requests into the final one."""
    total_bytes = sum(m.size_bytes for m in measurements)
    total_seconds = sum(m.seconds for m in measurements)
    return Summary(
        total_bytes=total_bytes,
        average_seconds=total_seconds / len(measurements),
        # Total size over total time: the real throughput of the whole test.
        speed_mbps=to_mbps(total_bytes, total_seconds),
    )


def http_url(text: str) -> str:
    """Argument type: an absolute http(s) URL.

    Characters that are not allowed in URLs, such as spaces or Cyrillic
    letters in a file name, are percent-encoded the same way browsers do it.
    """
    parts = urllib.parse.urlsplit(text)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise argparse.ArgumentTypeError(f"not an http(s) URL: {text!r}")
    return parts._replace(
        path=urllib.parse.quote(parts.path, safe=URL_SAFE_CHARS),
        query=urllib.parse.quote(parts.query, safe=URL_SAFE_CHARS),
    ).geturl()


def positive_int(text: str) -> int:
    """Argument type: an integer greater than zero."""
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be greater than 0, got {text}")
    return value


def positive_float(text: str) -> float:
    """Argument type: a finite number greater than zero."""
    value = float(text)
    if not (math.isfinite(value) and value > 0):
        raise argparse.ArgumentTypeError(f"must be greater than 0, got {text}")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments (sys.argv[1:] by default)."""
    parser = argparse.ArgumentParser(
        prog="speedtest.py",
        description=(
            "Measure download speed: request a URL several times in a row and "
            "print the average request time, the total downloaded size and "
            "the download speed in Mbit/s."
        ),
    )
    parser.add_argument(
        "url",
        type=http_url,
        help="http(s) address of the file to download, e.g. a large image",
    )
    parser.add_argument(
        "-n",
        "--requests",
        type=positive_int,
        default=DEFAULT_REQUESTS,
        metavar="N",
        help="number of sequential requests (default: %(default)s)",
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=positive_float,
        default=DEFAULT_TIMEOUT,
        metavar="SECONDS",
        help=(
            "how long to wait for the connection or the next piece of data "
            "(default: %(default)s)"
        ),
    )
    return parser.parse_args(argv)


def format_row(number: int, measurement: Measurement) -> str:
    """Format the result of one request as a table row."""
    speed = to_mbps(measurement.size_bytes, measurement.seconds)
    return ROW.format(
        number,
        f"{measurement.size_bytes / MEGA:.2f} MB",
        f"{measurement.seconds:.3f} s",
        f"{speed:.2f} Mbit/s",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the speed test and return the exit code."""
    args = parse_args(argv)
    print(f"URL:                   {args.url}")
    print(f"Requests:              {args.requests}, one after another")
    print()
    print(ROW.format("#", "Size", "Time", "Speed"), flush=True)

    measurements: list[Measurement] = []
    for number in range(1, args.requests + 1):
        try:
            measurement = download(args.url, args.timeout)
        except DownloadError as error:
            print(
                f"Error: request {number} of {args.requests} failed: {error}",
                file=sys.stderr,
            )
            return 1
        measurements.append(measurement)
        # flush: show progress even when the output is not a terminal.
        print(format_row(number, measurement), flush=True)

    summary = summarize(measurements)
    total_mb = summary.total_bytes / MEGA
    speed_mb_per_s = summary.speed_mbps / BITS_PER_BYTE
    print()
    print(f"Total downloaded:      {total_mb:.2f} MB ({summary.total_bytes:,} bytes)")
    print(f"Average request time:  {summary.average_seconds:.3f} s")
    print(
        f"Download speed:        {summary.speed_mbps:.2f} Mbit/s "
        f"({speed_mb_per_s:.2f} MB/s)"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)  # conventional exit code for Ctrl+C
