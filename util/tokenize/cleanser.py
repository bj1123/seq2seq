import re
from abc import abstractmethod, ABC


def clean_line(line):
    line = re.sub('\n+', '\t', line).strip('\r\n\t ')
    return line.lower()


def check_korean(sample):
    japanese = re.compile('/[一-龠]+|[ぁ-ゔ]+|[ァ-ヴー]+|[々〆〤]+/u')
    chinese = re.compile(u'[⺀-⺙⺛-⻳⼀-⿕々〇〡-〩〸-〺〻㐀-䶵一-鿃豈-鶴侮-頻並-龎]', re.UNICODE)
    searched_japanese = japanese.findall(sample)
    searched_chinese = chinese.findall(sample)
    if len(searched_japanese) > 10 or len(searched_chinese) > 10:
        return False
    return bool(re.search(pattern=r".*[가-힣].*", string=sample))


class Cleanser(ABC):
    def __init__(self):
        delete_tokens = ['\t', '\r', '\x1e', '¤', '«', '»', '®', '°', '\xad', 'Ν', 'Φ', 'α', '‰', '‱', '', 'β',
                         '\u2000',
                         '\u2003', '\u200b', 'ủ', 'ừ', 'γ', 'μ', 'ν', 'π', 'Б', 'И', 'К', 'Р', 'Ф', 'ב', 'ו', 'ט', 'כ',
                         'ל', 'מ', 'ע',
                         'ᄎ', '⁰', ]
        danwui = ['㎃', '㎈', '㎉', '㎍', '㎎', '㎏', '㎐', '㎑', '㎒', '㎓', '㎔', '㎕', '㎖', '㎗', '㎘', '㎚', '㎛',
                  '㎜', '㎝', '㎞', '㎟', '㎠', '㎡', '㎢', '㎣', '㎤', '㎥', '㎦', '㎧', '㎨', '㎩', '㎫', '㎬', '㎰', '㎲', '㎳',
                  '㎸', '㎹', '㎼', '㎽', '㎾', '㎿', '㏃', '㏄', '㏅', '㏈', '㏊', '㏓', '㏖', '㏜', ]
        todots = ['━', '│', '┃', '╗', '□', '▣', '▦', '▨', '▪', '△', '▴', '▷', '▸', '▹', '▼', '▽', '◇', '◈', '◉', '○',
                  '◎', '●', '◦', '◯', '◾', '☁', '☂', '★', '☆', '☎', '☛', '☞', '☼', '♡', '♣', '♥', '♪', '♭', '✔', '✕',
                  '✪',
                  '❍', '❑', ]

        self.changes = {'·': '·',
                        'à': 'a',
                        'á': 'a',
                        'ä': 'a',
                        'é': 'e',
                        'ê': 'e',
                        'ì': 'i',
                        'ö': 'o',
                        '÷': '%',
                        'ù': 'u',
                        'ā': 'a',
                        'ą': 'a',
                        'ž': 'z',
                        '˙': "'",
                        '΄': "'",
                        'Χ': 'X',
                        "ᆞ": '·',
                        "∙": '·',
                        '‧': '·',
                        "•": '·',
                        "√": '·',
                        "․": '.',
                        "′": "'",
                        "″": "'",
                        '∼': '~',
                        '∽': '~',
                        '～': '~',
                        '％': '%',
                        '｝': '}',
                        '‑': '-',
                        '–': '-',
                        '―': '-',
                        '‘': "'",
                        '’': "'",
                        '‛': "'",
                        '“': "'",
                        '”': "'",
                        '＇': "'",
                        '！': '!',
                        '＂': "'",
                        '＃': '#',
                        '‣': '·',
                        '‥': '…',
                        'ː': ':',
                        '１': '1',
                        '２': '2',
                        '４': '4',
                        '６': '6',
                        'ｅ': 'e',
                        'ｇ': 'g',
                        'ｓ': 's',
                        'ｍ': 'm',
                        'ｔ': 't',
                        'ｘ': 'x',
                        'Ａ': 'A',
                        'Ｋ': 'K',
                        'Ｘ': 'X',
                        '＋': '+',
                        '，': ',',
                        '－': '-',
                        '．': '.',
                        '／': '/',
                        '（': '(',
                        '）': ')',
                        '？': '?',
                        '０': '0',
                        '〃': "'",
                        '〈': "<",
                        '〉': ">",
                        '《': '<',
                        '》': '>',
                        '「': '<',
                        '」': '>',
                        '『': '<',
                        '』': '>',
                        '〔': '<',
                        '〕': '>',
                        '≪': '<',
                        '｢': '<',
                        '｣': '>',
                        '：': ':',
                        '；': ';',
                        '＜': '<',
                        '＝': '=',
                        '＞': '>',
                        '［': '<',
                        '］': '>',
                        '＿': '_',
                        '①': '1',
                        '②': '2',
                        '③': '3',
                        '④': '4',
                        '⑤': '5',
                        '⑥': '6',
                        '⑦': '7',
                        '⑧': '8',
                        '⑨': '9',
                        '⑩': '10',
                        '⑪': '11',
                        '⑫': '12',
                        '⑬': '13',
                        '⓵': '1',
                        '⓶': '2',
                        '⓹': '5',
                        '⓺': '6',
                        '➀': '1',
                        '➁': '2',
                        '➂': '3',
                        '➃': '4',
                        '➄': '5',
                        '➅': '6',
                        '➈': '9',
                        }

        self.delete_tokens = re.compile('[' + ''.join(delete_tokens) + ']')
        self.measures = re.compile('[' + ''.join(danwui) + ']')
        self.todots = re.compile(r'[' + ''.join(todots) + ']')
        self.special_symbols = re.compile(r'([^ ?!.,a-zA-Z0-9ㄱ-ㅎㅏ-ㅣ가-힣])')

    def cleanse_special_symbols(self, x):
        x = self.delete_tokens.sub('', x)
        x = self.measures.sub('㎃', x)
        x = self.todots.sub('·', x)
        x = self.special_symbols.sub(r' \1  ', x)
        for i in self.changes.keys():
            x = x.replace(i, self.changes[i])
        x = re.sub(' {2,}', ' ', x)
        x = re.sub(r'\n{2,}', r'\n', x)
        return x

    @abstractmethod
    def cleanse(self, x, **kwargs):
        pass


class NullCleanser(Cleanser):
    def __init__(self):
        super(NullCleanser, self).__init__()

    def cleanse(self, x, **kwargs):
        return x
