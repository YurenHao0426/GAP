"""Unicode -> LaTeX cleaner for PutnamGAP dataset (v2).

Improvements over v1:
  - Pre-normalize via NFKD then strip combining diacritics so accented
    letters collapse to their ASCII base.
  - Group adjacent subscript/superscript runs into {...}: x_1_0 -> x_{10},
    x^2^3 -> x^{23}.
  - Wrap the argument of radical commands: \\sqrt-followed-by-X -> \\sqrt{X}
    where X is either an identifier/number run or a balanced paren/bracket
    group or a single \\-command (optionally followed by {...} arguments).
  - Explicit replacements for symbols that previously fell through:
    star, blacksquare/QED, fraction slash, dagger, etc.
  - Deletes lone combining diacritics and decorative box-drawing characters.

Operates IN PLACE on both dataset copies. Backup in a tarball first.
"""
from __future__ import annotations
import json
import re
import sys
import unicodedata
from pathlib import Path
from collections import Counter

DIRS = [
    Path("/home/yurenh2/gap/putnam-bench-anon/dataset"),
    Path("/home/yurenh2/gap/putnamsup/PutnamGAP"),
]

TOP_LEVEL_TEXT_FIELDS = ["question", "solution"]
VARIANT_TEXT_FIELDS = ["question", "solution"]
VARIANT_KEYS = [
    "descriptive_long",
    "descriptive_long_confusing",
    "descriptive_long_misleading",
    "garbled_string",
    "kernel_variant",
    "original_kernel_variant",
]


# Sentinels placed during char substitution, resolved in a later pass that
# can look at the following characters to extract the radical argument.
SENT_SQRT = "\x01SQRT\x01"
SENT_CBRT = "\x01CBRT\x01"
SENT_FRT = "\x01FRT\x01"

REPLACEMENTS: dict = {
    # Whitespace -> normal space
    "\u00A0": " ", "\u2002": " ", "\u2003": " ", "\u2004": " ",
    "\u2005": " ", "\u2006": " ", "\u2007": " ", "\u2008": " ",
    "\u2009": " ", "\u200A": " ", "\u200B": "", "\u200C": "",
    "\u200D": "", "\u202F": " ", "\u205F": " ", "\u3000": " ",
    "\uFEFF": "",

    # Dashes / hyphens
    # NOTE: in this dataset (kernel-variant LLM-generated math text) the
    # EN DASH is used pervasively as a math minus sign, not a typographic
    # en-dash, so we map it to a single hyphen-minus rather than the
    # typographic `--`. The EM DASH stays as `---` (prose convention).
    "\u2010": "-", "\u2011": "-",
    "\u2012": "-",       # FIGURE DASH
    "\u2013": "-",       # EN DASH (was `--`; common usage here is math minus)
    "\u2014": "---",     # EM DASH (typographic prose break)
    "\u2015": "---",     # HORIZONTAL BAR
    "\u2212": "-",

    # Quotation marks
    "\u2018": "`", "\u2019": "'", "\u201A": ",", "\u201B": "`",
    "\u201C": "``", "\u201D": "''", "\u201E": ",,",
    "\u00AB": "<<", "\u00BB": ">>",

    # Punctuation / miscellany
    "\u2022": "*",
    "\u2023": "*",
    "\u2027": ".",
    "\u2026": r"\ldots",
    "\u00B7": r"\cdot",
    "\u00B0": r"^\circ",
    "\u2032": "'", "\u2033": "''", "\u2034": "'''", "\u2035": "`",
    "\u2605": r"\star",
    "\u2606": r"\star",
    "\u25A0": r"\blacksquare",
    "\u25A1": r"\square",
    "\u220E": r"\blacksquare",
    "\u2020": r"\dagger",
    "\u2021": r"\ddagger",
    "\u2044": "/",

    # Sub/super digits
    "\u2070": "^0", "\u00B9": "^1", "\u00B2": "^2", "\u00B3": "^3",
    "\u2074": "^4", "\u2075": "^5", "\u2076": "^6", "\u2077": "^7",
    "\u2078": "^8", "\u2079": "^9",
    "\u207A": "^+", "\u207B": "^-", "\u207C": "^=", "\u207D": "^(", "\u207E": "^)",
    "\u2080": "_0", "\u2081": "_1", "\u2082": "_2", "\u2083": "_3",
    "\u2084": "_4", "\u2085": "_5", "\u2086": "_6", "\u2087": "_7",
    "\u2088": "_8", "\u2089": "_9",
    "\u208A": "_+", "\u208B": "_-", "\u208C": "_=", "\u208D": "_(", "\u208E": "_)",

    # Latin sub/super letters
    "\u2090": "_a", "\u2091": "_e", "\u2092": "_o", "\u2093": "_x",
    "\u2095": "_h", "\u2096": "_k", "\u2097": "_l", "\u2098": "_m",
    "\u2099": "_n", "\u209A": "_p", "\u209B": "_s", "\u209C": "_t",
    "\u2C7C": "_j",  # LATIN SUBSCRIPT SMALL LETTER J
    "\u1D30": "^D", "\u1D31": "^E", "\u1D33": "^G", "\u1D34": "^H",
    "\u1D35": "^I", "\u1D36": "^J", "\u1D37": "^K", "\u1D38": "^L",
    "\u1D39": "^M", "\u1D3A": "^N", "\u1D3C": "^O", "\u1D3E": "^P",
    "\u1D3F": "^R", "\u1D40": "^T", "\u1D41": "^U", "\u1D42": "^W",
    "\u1D43": "^a", "\u1D47": "^b", "\u1D48": "^d", "\u1D49": "^e",
    "\u1D4D": "^g", "\u1D4F": "^k", "\u1D50": "^m", "\u1D52": "^o",
    "\u1D56": "^p", "\u1D57": "^t", "\u1D58": "^u", "\u1D5B": "^v",
    "\u1D62": "_i", "\u1D63": "_r", "\u1D64": "_u", "\u1D65": "_v",
    "\u2071": "^i", "\u207F": "^n",

    # Greek lower case
    "\u03B1": r"\alpha", "\u03B2": r"\beta", "\u03B3": r"\gamma",
    "\u03B4": r"\delta", "\u03B5": r"\varepsilon", "\u03B6": r"\zeta",
    "\u03B7": r"\eta", "\u03B8": r"\theta", "\u03B9": r"\iota",
    "\u03BA": r"\kappa", "\u03BB": r"\lambda", "\u03BC": r"\mu",
    "\u03BD": r"\nu", "\u03BE": r"\xi", "\u03BF": "o",
    "\u03C0": r"\pi", "\u03C1": r"\rho", "\u03C2": r"\varsigma",
    "\u03C3": r"\sigma", "\u03C4": r"\tau", "\u03C5": r"\upsilon",
    "\u03C6": r"\varphi", "\u03C7": r"\chi", "\u03C8": r"\psi",
    "\u03C9": r"\omega",
    "\u03D5": r"\phi", "\u03D1": r"\vartheta", "\u03D6": r"\varpi",
    "\u03F1": r"\varrho", "\u03F5": r"\epsilon",
    # Greek upper case
    "\u0391": "A", "\u0392": "B", "\u0393": r"\Gamma",
    "\u0394": r"\Delta", "\u0395": "E", "\u0396": "Z",
    "\u0397": "H", "\u0398": r"\Theta", "\u0399": "I",
    "\u039A": "K", "\u039B": r"\Lambda", "\u039C": "M",
    "\u039D": "N", "\u039E": r"\Xi", "\u039F": "O",
    "\u03A0": r"\Pi", "\u03A1": "P", "\u03A3": r"\Sigma",
    "\u03A4": "T", "\u03A5": r"\Upsilon", "\u03A6": r"\Phi",
    "\u03A7": "X", "\u03A8": r"\Psi", "\u03A9": r"\Omega",

    # Math operators / relations
    "\u2200": r"\forall", "\u2203": r"\exists", "\u2204": r"\nexists",
    "\u2205": r"\emptyset",
    "\u2208": r"\in", "\u2209": r"\notin", "\u220B": r"\ni",
    "\u220F": r"\prod", "\u2210": r"\coprod", "\u2211": r"\sum",
    "\u2213": r"\mp", "\u00B1": r"\pm",
    "\u2214": r"\dotplus",
    "\u2217": "*", "\u2218": r"\circ", "\u2219": r"\cdot",
    "\u221D": r"\propto",
    "\u221E": r"\infty",
    "\u2220": r"\angle", "\u2221": r"\measuredangle",
    "\u2225": r"\parallel", "\u2226": r"\nparallel",
    "\u2227": r"\land", "\u2228": r"\lor",
    "\u2229": r"\cap", "\u222A": r"\cup",
    "\u222B": r"\int", "\u222C": r"\iint", "\u222D": r"\iiint",
    "\u222E": r"\oint", "\u222F": r"\oiint",
    "\u2234": r"\therefore", "\u2235": r"\because",
    "\u2236": ":", "\u2237": "::",
    "\u223C": r"\sim", "\u2243": r"\simeq", "\u2245": r"\cong",
    "\u2248": r"\approx", "\u224D": r"\asymp",
    "\u2250": r"\doteq",
    "\u2260": r"\neq", "\u2261": r"\equiv", "\u2262": r"\not\equiv",
    "\u2264": r"\leq", "\u2265": r"\geq",
    "\u2266": r"\leqq", "\u2267": r"\geqq",
    "\u226A": r"\ll", "\u226B": r"\gg",
    "\u2270": r"\not\leq", "\u2271": r"\not\geq",
    "\u2282": r"\subset", "\u2283": r"\supset",
    "\u2284": r"\not\subset", "\u2285": r"\not\supset",
    "\u2286": r"\subseteq", "\u2287": r"\supseteq",
    "\u2288": r"\not\subseteq", "\u2289": r"\not\supseteq",
    "\u228A": r"\subsetneq", "\u228B": r"\supsetneq",
    "\u2295": r"\oplus", "\u2296": r"\ominus",
    "\u2297": r"\otimes", "\u2298": r"\oslash", "\u2299": r"\odot",
    "\u22A2": r"\vdash", "\u22A3": r"\dashv",
    "\u22A4": r"\top", "\u22A5": r"\bot",
    "\u22A8": r"\models",
    "\u22C0": r"\bigwedge", "\u22C1": r"\bigvee",
    "\u22C2": r"\bigcap", "\u22C3": r"\bigcup",
    "\u22C5": r"\cdot", "\u22C6": r"\star",
    "\u22EE": r"\vdots", "\u22EF": r"\cdots",
    "\u22F1": r"\ddots",

    # Arrows
    "\u2190": r"\leftarrow", "\u2192": r"\to",
    "\u2191": r"\uparrow", "\u2193": r"\downarrow",
    "\u2194": r"\leftrightarrow", "\u2195": r"\updownarrow",
    "\u21A0": r"\twoheadrightarrow",
    "\u21A6": r"\mapsto",
    "\u21D0": r"\Leftarrow", "\u21D2": r"\Rightarrow",
    "\u21D1": r"\Uparrow", "\u21D3": r"\Downarrow",
    "\u21D4": r"\Leftrightarrow",
    "\u27F6": r"\longrightarrow", "\u27F5": r"\longleftarrow",
    "\u27F9": r"\Longrightarrow", "\u27F8": r"\Longleftarrow",
    "\u27FA": r"\Longleftrightarrow",

    # Delimiters
    "\u2016": r"\|",
    "\u2308": r"\lceil", "\u2309": r"\rceil",
    "\u230A": r"\lfloor", "\u230B": r"\rfloor",
    "\u27E8": r"\langle", "\u27E9": r"\rangle",
    "\u27EA": r"\llangle", "\u27EB": r"\rrangle",

    # Blackboard / script letters
    "\u2102": r"\mathbb{C}", "\u210D": r"\mathbb{H}",
    "\u2115": r"\mathbb{N}", "\u2119": r"\mathbb{P}",
    "\u211A": r"\mathbb{Q}", "\u211D": r"\mathbb{R}",
    "\u2124": r"\mathbb{Z}",
    "\u2113": r"\ell", "\u210F": r"\hbar",
    "\u2202": r"\partial", "\u2207": r"\nabla", "\u2118": r"\wp",
    "\u2133": r"\mathcal{M}", "\u2112": r"\mathcal{L}",
    "\u211B": r"\mathcal{R}", "\u2110": r"\mathcal{I}",
    "\u2130": r"\mathcal{E}", "\u2132": "F",

    # Fractions with precomposed forms
    "\u00BC": r"\frac{1}{4}", "\u00BD": r"\frac{1}{2}", "\u00BE": r"\frac{3}{4}",
    "\u2153": r"\frac{1}{3}", "\u2154": r"\frac{2}{3}",
    "\u2155": r"\frac{1}{5}", "\u2156": r"\frac{2}{5}",
    "\u2157": r"\frac{3}{5}", "\u2158": r"\frac{4}{5}",
    "\u2159": r"\frac{1}{6}", "\u215A": r"\frac{5}{6}",
    "\u215B": r"\frac{1}{8}", "\u215C": r"\frac{3}{8}",
    "\u215D": r"\frac{5}{8}", "\u215E": r"\frac{7}{8}",

    # Multiplication / division
    "\u00D7": r"\times", "\u00F7": r"\div",

    # Misc
    "\u00A7": r"\S",
    "\u00B6": r"\P",
    "\u00A9": "(c)", "\u00AE": "(R)", "\u2122": "(TM)",
    "\u00A3": r"\pounds", "\u20AC": "EUR",
    "\u00B5": r"\mu",

    # Additional math symbols
    "\u2216": r"\setminus",
    "\u2223": r"\mid",
    "\u2224": r"\nmid",
    "\u2225": r"\parallel",  # duplicate of above, safe
    "\u2226": r"\nparallel",
    "\u22BB": r"\veebar",
    "\u22BC": r"\barwedge",
    "\u2238": r"\dot{-}",
    "\u22C8": r"\bowtie",
    "\u22CE": r"\curlyvee",
    "\u22CF": r"\curlywedge",

    # Perp and triangle family
    "\u27C2": r"\perp",
    "\u22A5": r"\bot",       # already present but safe
    "\u25B3": r"\triangle",
    "\u25B4": r"\blacktriangle",
    "\u25BD": r"\triangledown",
    "\u25BE": r"\blacktriangledown",
    "\u25C1": r"\triangleleft",
    "\u25C2": r"\blacktriangleleft",
    "\u25B7": r"\triangleright",
    "\u25B8": r"\blacktriangleright",

    # Square / box operators
    "\u2293": r"\sqcap",
    "\u2294": r"\sqcup",
    "\u22A1": r"\boxdot",
    "\u229E": r"\boxplus",
    "\u229F": r"\boxminus",
    "\u22A0": r"\boxtimes",

    # Preceq / succeq family
    "\u227A": r"\prec",
    "\u227B": r"\succ",
    "\u227C": r"\preceq",
    "\u227D": r"\succeq",
    "\u2280": r"\nprec",
    "\u2281": r"\nsucc",
    "\u22E0": r"\npreceq",
    "\u22E1": r"\nsucceq",

    # Double-square brackets
    "\u27E6": r"\llbracket",
    "\u27E7": r"\rrbracket",

    # Card-suit decorative (drop)
    "\u2660": "",  # spade
    "\u2661": "",
    "\u2662": "",
    "\u2663": "",  # club
    "\u2664": "",
    "\u2665": "",  # heart
    "\u2666": "",  # diamond

    # Musical / dingbat decorations (drop)
    "\u266A": "",  # eighth note
    "\u266B": "",  # beamed eighth notes
    "\u2713": r"\checkmark",
    "\u2717": r"\times",

    # Curved delimiters / bracket extension pieces -- these are used by the
    # kernel generator to draw big parentheses/brackets around multi-line
    # expressions (like matrices). They are purely decorative in plain text
    # and we drop them.
    "\u239B": "", "\u239C": "", "\u239D": "",  # ( upper/mid/lower
    "\u239E": "", "\u239F": "", "\u23A0": "",  # ) upper/mid/lower
    "\u23A1": "", "\u23A2": "", "\u23A3": "",  # [ upper/mid/lower
    "\u23A4": "", "\u23A5": "", "\u23A6": "",  # ] upper/mid/lower
    "\u23A7": "", "\u23A8": "", "\u23A9": "",  # { upper/middle/lower
    "\u23AA": "",                               # { extension
    "\u23AB": "", "\u23AC": "", "\u23AD": "",  # } upper/middle/lower
    "\u23AE": "",                               # integral extension
    "\u23AF": "",                               # horizontal line extension
    "\u23B0": "", "\u23B1": "",                # upper/lower curly bracket
    "\u23B2": "", "\u23B3": "",                # summation top/bottom
    "\u23B4": "", "\u23B5": "",                # top/bottom square bracket
    "\u23B6": "", "\u23B7": "",                # bottom square bracket w/tick
    "\u23D0": "",                               # vertical line extension

    # Combining over/underlines are stripped by the combining-mark regex

    # Additional remaining symbols found after first clean pass
    "\u00AD": "",             # SOFT HYPHEN -> delete
    "\u2215": "/",            # DIVISION SLASH
    "\u25A2": r"\square",     # WHITE SQUARE WITH ROUNDED CORNERS
    "\u2718": r"\times",      # HEAVY BALLOT X
    "\u3008": r"\langle",     # CJK LEFT ANGLE BRACKET
    "\u3009": r"\rangle",     # CJK RIGHT ANGLE BRACKET
    "\u2254": ":=",           # COLON EQUALS
    "\u2255": "=:",           # EQUALS COLON
    "\u2198": r"\searrow",    # SOUTH EAST ARROW
    "\u2197": r"\nearrow",    # NORTH EAST ARROW
    "\u2199": r"\swarrow",
    "\u2196": r"\nwarrow",
    "\u21A9": r"\hookleftarrow",
    "\u21AA": r"\hookrightarrow",
    "\u21BC": r"\leftharpoonup",
    "\u21BD": r"\leftharpoondown",
    "\u21BE": r"\upharpoonright",
    "\u21BF": r"\upharpoonleft",
    "\u21C0": r"\rightharpoonup",
    "\u21C1": r"\rightharpoondown",
    "\u21C2": r"\downharpoonright",
    "\u21C3": r"\downharpoonleft",
    "\u21CC": r"\rightleftharpoons",
    "\u21E2": r"\dashrightarrow",
    "\u21E0": r"\dashleftarrow",
    "\u2277": r"\gtrless",
    "\u2276": r"\lessgtr",

    # Private Use Area characters are almost always OCR garbage or
    # font-specific glyphs; drop them.
    "\uF8EB": "", "\uF8F6": "",
    "\uF8FE": "", "\uF8FD": "", "\uF8FC": "", "\uF8FB": "",
    "\uF8EF": "", "\uF8F0": "", "\uF8F1": "", "\uF8F2": "",

    # A few more rare but meaningful math symbols
    "\u2322": r"\frown",
    "\u2323": r"\smile",
    "\u226D": r"\not\asymp",
    "\u22A7": r"\models",
    "\u22B2": r"\vartriangleleft",
    "\u22B3": r"\vartriangleright",
    "\u22B4": r"\trianglelefteq",
    "\u22B5": r"\trianglerighteq",

    # Small-caps letters sometimes emitted by OCR (collapse to plain letter)
    "\u026A": "I",   # LATIN LETTER SMALL CAPITAL I
    "\u1D00": "A",
    "\u1D04": "C",
    "\u1D05": "D",
    "\u1D07": "E",
    "\u0262": "G",
    "\u029C": "H",

    # Remaining math symbols found after pass 2
    "\u2A01": r"\bigoplus",
    "\u2A02": r"\bigotimes",
    "\u2A00": r"\bigodot",
    "\u2A03": r"\biguplus",
    "\u2A04": r"\biguplus",
    "\u2A05": r"\bigsqcap",
    "\u2A06": r"\bigsqcup",
    "\u2272": r"\lesssim",
    "\u2273": r"\gtrsim",
    "\u226E": r"\not<",
    "\u226F": r"\not>",
    "\u27EE": "(",     # MATHEMATICAL LEFT FLATTENED PARENTHESIS
    "\u27EF": ")",     # MATHEMATICAL RIGHT FLATTENED PARENTHESIS
    "\u2610": r"\square",   # BALLOT BOX
    "\u2611": r"\checkmark",
    "\u2612": r"\times",

    # Root sentinels (wrapped in a later pass)
    "\u221A": SENT_SQRT,
    "\u221B": SENT_CBRT,
    "\u221C": SENT_FRT,
}


_COMBINING_MARK_RE = re.compile(
    r"[\u0300-\u036F\u1AB0-\u1AFF\u1DC0-\u1DFF\u20D0-\u20FF\uFE20-\uFE2F]")
_BOX_DRAWING_RE = re.compile(r"[\u2500-\u257F\u2580-\u259F]")

# Characters from scripts that have no place in English/Greek mathematics
# and are clearly OCR noise when they appear. Drop them wholesale. Latin and
# Greek are preserved; extended Latin letters with diacritics are still
# handled by the NFKD fallback.
_OCR_NOISE_SCRIPTS_RE = re.compile(
    r"[\u0400-\u04FF"   # Cyrillic
    r"\u0500-\u052F"   # Cyrillic Supplement
    r"\u0530-\u058F"   # Armenian
    r"\u0590-\u05FF"   # Hebrew
    r"\u0600-\u06FF"   # Arabic
    r"\u0700-\u074F"   # Syriac
    r"\u0750-\u077F"   # Arabic Supplement
    r"\u0780-\u07BF"   # Thaana
    r"\u0900-\u097F"   # Devanagari
    r"\u0B80-\u0BFF"   # Tamil
    r"\u0C00-\u0C7F"   # Telugu
    r"\u0C80-\u0CFF"   # Kannada
    r"\u0D00-\u0D7F"   # Malayalam
    r"\u0D80-\u0DFF"   # Sinhala
    r"\u0E00-\u0E7F"   # Thai
    r"\u0E80-\u0EFF"   # Lao
    r"\u0F00-\u0FFF"   # Tibetan
    r"\u1000-\u109F"   # Myanmar
    r"\u10A0-\u10FF"   # Georgian
    r"\u1100-\u11FF"   # Hangul Jamo
    r"\u1400-\u167F"   # Unified Canadian Aboriginal Syllabics
    r"\u1680-\u169F"   # Ogham
    r"\u16A0-\u16FF"   # Runic
    r"\u1700-\u171F"   # Tagalog
    r"\u1780-\u17FF"   # Khmer
    r"\u1800-\u18AF"   # Mongolian
    r"\u1900-\u194F"   # Limbu
    r"\u3040-\u309F"   # Hiragana
    r"\u30A0-\u30FF"   # Katakana
    r"\u3000-\u303F"   # CJK Symbols and Punctuation (incl. ideographic full stop)
    r"\u3100-\u312F"   # Bopomofo
    r"\u3130-\u318F"   # Hangul Compatibility Jamo
    r"\u3190-\u319F"   # Kanbun
    r"\u3400-\u4DBF"   # CJK Extension A
    r"\u4E00-\u9FFF"   # CJK Unified Ideographs
    r"\uA000-\uA48F"   # Yi Syllables
    r"\uAC00-\uD7AF"   # Hangul Syllables
    r"\uE000-\uF8FF"   # Private Use Area
    r"\uFE00-\uFE0F"   # Variation Selectors
    r"\uFE30-\uFE4F"   # CJK Compatibility Forms (vertical presentation
                       # brackets that NFKD-decompose to literal { } [ ] etc.,
                       # which would corrupt our brace balance — drop them)
    r"\uFE50-\uFE6F"   # Small Form Variants (compatibility forms)
    r"\uFFFC\uFFFD"    # Object/Replacement Character
    r"]"
)

# Emoji and pictographs (outside the BMP, need surrogate handling)
_EMOJI_RE = re.compile(
    "["
    "\U0001F000-\U0001F9FF"   # Emoji blocks
    "\U0001FA00-\U0001FAFF"   # Symbols & Pictographs Extended-A
    "\U0001F1E6-\U0001F1FF"   # Regional indicator symbols
    "\U0001F3FB-\U0001F3FF"   # Emoji modifier fitzpatrick
    "\U00020000-\U0002FA1F"   # CJK Extensions B-F
    "]",
    flags=re.UNICODE
)


def prestrip(text: str) -> str:
    """Strip decorative and OCR-noise characters BEFORE char substitution.

    Important: we do NOT run NFKD here because NFKD decomposes subscript /
    superscript digits (e.g. \u2080 -> '0') before our explicit REPLACEMENTS
    entries can rewrite them as `_0`. NFKD is applied later only as a
    fallback for characters that survive the explicit substitution pass
    (e.g. accented Latin letters).
    """
    if not text:
        return text
    text = _BOX_DRAWING_RE.sub("", text)
    # Lone combining marks are orphaned when the base character was something
    # we otherwise transformed; strip them up front.
    text = _COMBINING_MARK_RE.sub("", text)
    # Strip OCR-noise scripts (Cyrillic / Arabic / CJK / etc.) that have no
    # place in English-Greek mathematical prose.
    text = _OCR_NOISE_SCRIPTS_RE.sub("", text)
    # Strip emoji / pictographs (clearly LLM-emitted noise in math text).
    text = _EMOJI_RE.sub("", text)
    return text


def char_substitute(text: str, unmapped: Counter) -> str:
    """Apply REPLACEMENTS char-by-char. Any char not in REPLACEMENTS is left
    in place so that _nfkd_fallback (run next) has a chance to handle it
    via compatibility decomposition. A trailing space is appended to bare
    `\\word` LaTeX commands so subsequent letters do not get absorbed into
    the command name.
    """
    out = []
    for ch in text:
        if ord(ch) <= 127 or ch == "\x01":
            out.append(ch)
            continue
        if ch in REPLACEMENTS:
            val = REPLACEMENTS[ch]
            # Bare `\word` (starts with `\\`, ends in a letter) needs a
            # trailing space so that `\cdot t` does not become `\cdott`.
            if (len(val) >= 2 and val[0] == "\\"
                    and val[-1].isalpha()
                    and not val.startswith("\x01")):
                val = val + " "
            out.append(val)
            continue
        # Unmapped: keep as-is and let _nfkd_fallback try compat decomposition.
        out.append(ch)
    return "".join(out)


def _merge_sub_sup(text: str) -> str:
    def _do(prefix, m):
        # Extract each ^X or _X token and concatenate the X parts.
        vals = re.findall(r"[\+\-\=\(\)a-zA-Z0-9]", m.group(0))
        # The regex captures the X char from each ^X or _X; above regex
        # finds ALL alnum/sign chars in the match. But `^+` etc. we want
        # to keep as-is. Simplest: split on the prefix.
        pieces = [p for p in re.split(r"[\^_]", m.group(0)) if p]
        joined = "".join(pieces)
        return f"{prefix}{{{joined}}}"

    text = re.sub(
        r"(?:\^[\+\-\=\(\)a-zA-Z0-9])(?:\^[\+\-\=\(\)a-zA-Z0-9])+",
        lambda m: _do("^", m), text)
    text = re.sub(
        r"(?:_[\+\-\=\(\)a-zA-Z0-9])(?:_[\+\-\=\(\)a-zA-Z0-9])+",
        lambda m: _do("_", m), text)
    return text


_SENTINEL_RE = re.compile(r"\x01(SQRT|CBRT|FRT)\x01")


def _skip_spaces(s: str, i: int) -> int:
    while i < len(s) and s[i] in " \t":
        i += 1
    return i


def _read_balanced(s: str, i: int, open_ch: str, close_ch: str):
    depth = 0
    j = i
    while j < len(s):
        if s[j] == open_ch:
            depth += 1
        elif s[j] == close_ch:
            depth -= 1
            if depth == 0:
                return j + 1
        j += 1
    return -1


def _read_latex_command(s: str, i: int):
    if i >= len(s) or s[i] != "\\":
        return -1
    j = i + 1
    while j < len(s) and (s[j].isalpha() or s[j] == "@"):
        j += 1
    while j < len(s) and s[j] == "{":
        end = _read_balanced(s, j, "{", "}")
        if end == -1:
            return j
        j = end
    return j


def _wrap_radical_arguments(text: str) -> str:
    out = []
    i = 0
    LATEX_FOR = {"SQRT": r"\sqrt", "CBRT": r"\sqrt[3]", "FRT": r"\sqrt[4]"}
    while i < len(text):
        m = _SENTINEL_RE.match(text, i)
        if not m:
            out.append(text[i])
            i += 1
            continue
        kind = m.group(1)
        latex_prefix = LATEX_FOR[kind]
        j = _skip_spaces(text, m.end())
        if j >= len(text):
            out.append(latex_prefix + "{}")
            i = j
            continue
        ch = text[j]
        if ch == "(":
            arg_end = _read_balanced(text, j, "(", ")")
            if arg_end != -1:
                arg = text[j + 1 : arg_end - 1]
                out.append(f"{latex_prefix}{{{arg}}}")
                i = arg_end
                continue
        if ch == "[":
            arg_end = _read_balanced(text, j, "[", "]")
            if arg_end != -1:
                arg = text[j + 1 : arg_end - 1]
                out.append(f"{latex_prefix}{{{arg}}}")
                i = arg_end
                continue
        if ch == "{":
            arg_end = _read_balanced(text, j, "{", "}")
            if arg_end != -1:
                arg = text[j + 1 : arg_end - 1]
                out.append(f"{latex_prefix}{{{arg}}}")
                i = arg_end
                continue
        if ch == "\\":
            arg_end = _read_latex_command(text, j)
            if arg_end != -1:
                arg = text[j:arg_end]
                out.append(f"{latex_prefix}{{{arg}}}")
                i = arg_end
                continue
        # Fallback: alnum run (and dots for things like 3.14)
        k = j
        while k < len(text) and (text[k].isalnum() or text[k] in "."):
            k += 1
        if k > j:
            arg = text[j:k]
            out.append(f"{latex_prefix}{{{arg}}}")
            i = k
            continue
        out.append(latex_prefix + "{}")
        i = m.end()
    return "".join(out)


def _nfkd_fallback(text: str, unmapped: Counter) -> str:
    """For characters that survived explicit substitution and are still
    non-ASCII (e.g. precomposed accented Latin letters like \u00E9 / e-acute,
    or classical Greek letters with breathing marks like \u1F42), run NFKD
    and drop combining marks, then re-apply REPLACEMENTS (because NFKD can
    unmask characters that do appear in REPLACEMENTS, e.g. \u1F42 -> \u03B3).
    Finally, any character that is still non-ASCII is logged and dropped.
    """
    has_non_ascii = any(ord(c) > 127 and c != "\x01" for c in text)
    if not has_non_ascii:
        return text
    text = unicodedata.normalize("NFKD", text)
    text = _COMBINING_MARK_RE.sub("", text)
    # Second pass of char_substitute now that NFKD has possibly surfaced
    # characters that were previously embedded in precomposed forms.
    text = char_substitute(text, unmapped)  # unmapped counter accumulates
    # Final drop of anything still non-ASCII
    out = []
    for c in text:
        if ord(c) <= 127 or c == "\x01":
            out.append(c)
        else:
            unmapped[c] += 1
    return "".join(out)


def clean_text(text: str, unmapped: Counter) -> str:
    if not text:
        return text
    text = prestrip(text)
    text = char_substitute(text, unmapped)
    text = _nfkd_fallback(text, unmapped)
    text = _merge_sub_sup(text)
    text = _wrap_radical_arguments(text)
    return text


def clean_problem(problem: dict, unmapped: Counter):
    for k in TOP_LEVEL_TEXT_FIELDS:
        if isinstance(problem.get(k), str):
            problem[k] = clean_text(problem[k], unmapped)
    variants = problem.get("variants") or {}
    for vk in VARIANT_KEYS:
        vd = variants.get(vk)
        if not isinstance(vd, dict):
            continue
        for k in VARIANT_TEXT_FIELDS:
            if isinstance(vd.get(k), str):
                vd[k] = clean_text(vd[k], unmapped)
    return problem


def process_dir(dataset_dir: Path):
    print(f"\n=== Cleaning {dataset_dir} ===")
    files = sorted(dataset_dir.glob("*.json"))
    unmapped = Counter()
    n_modified = 0
    for f in files:
        try:
            d = json.load(open(f))
        except Exception as e:
            print(f"  ! skip {f.name}: {e}")
            continue
        before = json.dumps(d, ensure_ascii=False)
        d = clean_problem(d, unmapped)
        after = json.dumps(d, ensure_ascii=False)
        if before != after:
            n_modified += 1
            with open(f, "w") as fh:
                json.dump(d, fh, ensure_ascii=False, indent=2)
    print(f"  files modified: {n_modified}/{len(files)}")
    if unmapped:
        print(f"  unmapped characters: {sum(unmapped.values())} occurrences, "
              f"{len(unmapped)} distinct")
        print(f"  top 20 unmapped:")
        for ch, n in unmapped.most_common(20):
            name = unicodedata.name(ch, "?")
            print(f"    {ch!r:<10} U+{ord(ch):04X} n={n} ({name})")
    else:
        print(f"  no unmapped characters")
    return unmapped


def main():
    all_unmapped = Counter()
    for d in DIRS:
        if d.exists():
            u = process_dir(d)
            all_unmapped.update(u)
    print(f"\n=== OVERALL ===")
    print(f"Total unmapped characters across both dataset copies: {sum(all_unmapped.values())}")
    print(f"Distinct unmapped: {len(all_unmapped)}")
    if all_unmapped:
        out_path = Path("/home/yurenh2/gap/analysis/unmapped_chars.json")
        json.dump({f"U+{ord(c):04X}": {"char": c, "name": unicodedata.name(c, "?"),
                                        "count": n}
                   for c, n in all_unmapped.most_common()},
                  open(out_path, "w"), indent=2, ensure_ascii=False)
        print(f"Saved unmapped list -> {out_path}")


if __name__ == "__main__":
    main()
