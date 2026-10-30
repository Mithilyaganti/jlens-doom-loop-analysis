"""Token-to-character offsets for byte-level BPE.

Qwen3.5-4B's tokenizer stores pieces in the GPT-2 byte-to-unicode alphabet
and decodes them by concatenating bytes, then UTF-8-decoding the whole
buffer. Mapping each piece on its own mis-counts characters whenever a
code point's bytes are split across tokens. Telugu (three UTF-8 bytes per
letter) hits that case often; English ASCII almost never does.

``TokenOffsetMap`` assigns every decoded code point to the token that
contains that code point's first byte. Continuation-only tokens get an
empty span, so a character index resolves to exactly one token.
"""

from __future__ import annotations

from dataclasses import dataclass


class OffsetMapError(ValueError):
    """Raised when token pieces cannot be mapped onto one Unicode string."""


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2 / Qwen byte-level alphabet. One Unicode scalar per byte."""
    visible = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    mapping: dict[int, str] = {byte: chr(byte) for byte in visible}
    next_code = 256
    for byte in range(256):
        if byte not in mapping:
            mapping[byte] = chr(next_code)
            next_code += 1
    return mapping


_BYTE_TO_UNICODE = _bytes_to_unicode()
_UNICODE_TO_BYTE = {ord(char): byte for byte, char in _BYTE_TO_UNICODE.items()}


def token_string_to_bytes(token: str) -> bytes:
    """Invert a byte-level token string.

    If every character is in the GPT-2 alphabet, map each character back to
    its byte. Otherwise the token is treated as already-decoded text (a
    literal special token that is not in that alphabet) and UTF-8-encoded.
    Printable ASCII is in the alphabet and round-trips either way.
    """
    if not token:
        return b""
    raw = bytearray()
    for char in token:
        byte = _UNICODE_TO_BYTE.get(ord(char))
        if byte is None:
            return token.encode("utf-8")
        raw.append(byte)
    return bytes(raw)


def _char_byte_starts(text: str, nbytes: int) -> list[int]:
    starts: list[int] = []
    cursor = 0
    for char in text:
        starts.append(cursor)
        cursor += len(char.encode("utf-8"))
    if cursor != nbytes:
        raise OffsetMapError(
            "UTF-8 re-encode of the decoded text does not match the byte buffer "
            f"({cursor} vs {nbytes}). Refusing to guess offsets."
        )
    return starts


@dataclass(frozen=True)
class TokenOffsetMap:
    """Character spans for one token sequence.

    ``spans[i]`` is the half-open code-point range of ``text`` whose first
    byte falls inside token ``i``. The range is empty when the token holds
    only continuation bytes of a character that started earlier.
    """

    text: str
    spans: tuple[tuple[int, int], ...]
    token_strings: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.spans) != len(self.token_strings):
            raise OffsetMapError("span count does not match token count")

    def __len__(self) -> int:
        return len(self.token_strings)

    def char_to_token_index(self, char_pos: int) -> int | None:
        """Return the token that owns this code-point index, if any."""
        if char_pos < 0 or char_pos >= len(self.text):
            return None
        for index, (start, end) in enumerate(self.spans):
            if start <= char_pos < end:
                return index
        return None

    def token_span(self, token_index: int) -> tuple[int, int]:
        if token_index < 0 or token_index >= len(self.spans):
            raise IndexError(token_index)
        return self.spans[token_index]


def map_token_strings(token_strings: list[str] | tuple[str, ...]) -> TokenOffsetMap:
    """Build offsets from byte-level token strings (``convert_ids_to_tokens``)."""
    pieces = [token_string_to_bytes(token) for token in token_strings]
    buffer = b"".join(pieces)
    try:
        text = buffer.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise OffsetMapError(
            "Concatenated token bytes are not valid UTF-8. "
            "Qwen's decoder would insert U+FFFD; this mapper refuses that "
            "rather than invent character boundaries."
        ) from exc
    starts = _char_byte_starts(text, len(buffer))
    spans: list[tuple[int, int]] = []
    cursor = 0
    for piece in pieces:
        byte_start = cursor
        byte_end = cursor + len(piece)
        cursor = byte_end
        owned = [
            char_index
            for char_index, char_byte in enumerate(starts)
            if byte_start <= char_byte < byte_end
        ]
        if not owned:
            if byte_start >= len(buffer):
                spans.append((len(text), len(text)))
            else:
                host = max(
                    (
                        char_index
                        for char_index, char_byte in enumerate(starts)
                        if char_byte <= byte_start
                    ),
                    default=len(text),
                )
                spans.append((host, host))
        else:
            spans.append((owned[0], owned[-1] + 1))
    if cursor != len(buffer):
        raise OffsetMapError("byte cursor did not consume the buffer")
    return TokenOffsetMap(
        text=text,
        spans=tuple(spans),
        token_strings=tuple(token_strings),
    )


def map_token_ids(token_ids: list[int], id_to_token: object) -> TokenOffsetMap:
    """Build offsets from ids plus ``id_to_token(int) -> str``."""
    if not callable(id_to_token):
        raise TypeError("id_to_token must be callable")
    strings: list[str] = []
    for token_id in token_ids:
        token = id_to_token(int(token_id))
        if not isinstance(token, str):
            raise OffsetMapError(f"id_to_token({token_id}) returned {type(token).__name__}, expected str")
        strings.append(token)
    return map_token_strings(strings)
