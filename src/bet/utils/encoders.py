import base64
from enum import StrEnum
from functools import reduce

from pyleetspeak.LeetSpeaker import LeetSpeaker
from unidecode import unidecode


class Encodings(StrEnum):
    BASE64 = "base64"
    ASCII = "ascii"
    UNICODE = "unicode"
    ROT13 = "rot13"
    ROT18 = "rot18"
    ROT47 = "rot47"
    SEPARATOR = "separator"
    LEETSPEAK_BASIC = "leetspeak_basic"
    LEETSPEAK_INTERMEDIATE = "leetspeak_intermediate"
    LEETSPEAK_ADVANCED = "leetspeak_advanced"


# NOTE: Encodings only work with letters and number

def encode_ascii(
    s: str, 
    sep: str = " "
) -> str:
    if not s:
        return ""
    char_codes = list(map(ord, s))
    if len(char_codes) == 1:
        return str(char_codes[0])
    return reduce(lambda x, y: str(x) + sep + str(y), char_codes)


def encode_base64(
    s: str,
) -> str:
    s = unidecode(s).encode("ascii")
    return base64.b64encode(s).decode("ascii")


def encode_unicode(
    s: str,
) -> str:
    return "".join([r"\u{:04X}".format(ord(char)) for char in s])


def encode_with_separator(
    s: str,
    sep: str = ".",
    n_sep: int = 1,
) -> str:
    return f"{sep * n_sep}".join(s.split())

def _encode_rotn(
    s: str,
    start: int,
    end: int,
    rotation: int
) -> str:
    """
    Generic ROT-N encoding function that rotates ASCII characters in a range.
    Non-ASCII characters are kept unchanged.
    
    Args:
        s: String to encode
        start: Start of ASCII range to rotate (inclusive)
        end: End of ASCII range to rotate (inclusive)
        rotation: Number of positions to rotate
    """
    result = []
    range_size = end - start + 1
    
    for char in s:
        code = ord(char)
        if start <= code <= end:
            # Rotate within the ASCII range
            result.append(chr(start + ((code - start + rotation) % range_size)))
        else:
            # Keep non-ASCII and out-of-range characters unchanged
            result.append(char)
    
    return ''.join(result)


def encode_rot13(
    s: str,
) -> str:
    """ROT13 encoding: rotates A-Z (65-90) and a-z (97-122) by 13 positions."""
    # Process uppercase and lowercase separately
    s = _encode_rotn(s, start=65, end=90, rotation=13)   # A-Z
    s = _encode_rotn(s, start=97, end=122, rotation=13)  # a-z
    return s


def encode_rot18(
    s: str,
) -> str:
    """ROT18 encoding: ROT13 for letters + ROT5 for digits."""
    s = _encode_rotn(s, start=65, end=90, rotation=13)   # A-Z
    s = _encode_rotn(s, start=97, end=122, rotation=13)  # a-z
    s = _encode_rotn(s, start=48, end=57, rotation=5)    # 0-9
    return s


def encode_rot47(
    s: str,
) -> str:
    """ROT47 encoding: rotates all printable ASCII characters (33-126) by 47 positions."""
    return _encode_rotn(s, start=33, end=126, rotation=47)

def encode_leetspeak_basic(
    s: str,
) -> str:
    return LeetSpeaker(mode="basic", seed=42).text2leet(s)

def encode_leetspeak_intermediate(
    s: str,
) -> str:
    return LeetSpeaker(mode="intermediate", seed=42).text2leet(s)

def encode_leetspeak_advanced(
    s: str,
) -> str:
    return LeetSpeaker(mode="advanced", seed=42).text2leet(s)


general_encoding_dict = {
    Encodings.BASE64: encode_base64,
    Encodings.ASCII: encode_ascii,
    Encodings.UNICODE: encode_unicode,
    Encodings.ROT13: encode_rot13,
    Encodings.ROT18: encode_rot18,
    Encodings.ROT47: encode_rot47,
    Encodings.SEPARATOR: encode_with_separator,
    Encodings.LEETSPEAK_BASIC: encode_leetspeak_basic,
    Encodings.LEETSPEAK_INTERMEDIATE: encode_leetspeak_intermediate,
    Encodings.LEETSPEAK_ADVANCED: encode_leetspeak_advanced
}
