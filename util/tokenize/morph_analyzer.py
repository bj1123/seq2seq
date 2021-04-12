# import khaiii
import itertools
import regex as re
import jamotools
try:
    import mecab
    Mecab = mecab.MeCab
except ModuleNotFoundError:
    from konlpy.tag import Mecab


def space_normalize(text):
    return re.sub(' +', ' ', text.strip())


class NullAnalyzer:
    def __init__(self, **kwargs):
        pass

    @staticmethod
    def to_morphs(text, **kwargs):
        return text

    @staticmethod
    def to_texts(text, **kwargs):
        return text.replace('##','').strip()


class MecabAnalyzer:
    def __init__(self,space_symbol='ㅬ', jamo=False):
        self.space_symbol = space_symbol
        self.analyzer = Mecab()
        self.space_symbol = space_symbol
        self.jamo = jamo

    def to_morphs(self, text, **kwargs):
        text = space_normalize(text)
        try:
            morphs = self.analyzer.morphs(text)
        except ValueError:
            morphs = ''
        if morphs:
            res = [self.space_symbol+jamotools.split_syllables(morphs[0])] if self.jamo else [self.space_symbol+morphs[0]]
            text = text[len(morphs[0]):]
            for i in morphs[1:]:
                if text[0] == ' ':
                    temp = self.space_symbol
                    text = text[len(i)+1:]
                else:
                    temp = ''
                    text = text[len(i):]
                if self.jamo:
                    res.append(temp+jamotools.split_syllables(i))
                else:
                    res.append(temp+i)
            return ' '.join(res) + ' \n'
        else:
            return '\n'

    def to_texts(self, morph, **kwargs):
        if isinstance(morph,str):
            temp = morph.replace('##','').replace(' ','').replace(self.space_symbol,' ').strip()
            if self.jamo:
                return jamotools.join_jamos(temp)
            else:
                return temp
        else:
            temp = re.sub(self.space_symbol, ' ', ''.join(morph)).strip()
            if self.jamo:
                return jamotools.join_jamos(temp)
            else:
                return temp


class MultilingualAnalyzer:
    def __init__(self, **kwargs):
        self.mecab = MecabAnalyzer(**kwargs)
        self.null = NullAnalyzer(**kwargs)

    @staticmethod
    def detect_language(text):
        # to implement
        return None

    def to_morphs(self, text, **kwargs):
        lang = kwargs.get('lang', None)
        if not lang:
            print('this should not be called')
            lang = self.detect_language(text)
        if lang == 'ko':
            return self.mecab.to_morphs(text, **kwargs)
        else:
            return self.null.to_morphs(text)

    def to_texts(self, text, **kwargs):
        lang = kwargs.pop('lang', None)
        if not lang:
            lang = self.detect_language(text)
        if lang == 'ko':
            return self.mecab.to_texts(text)
        else:
            return self.null.to_texts(text)


class Khaiii_Tokenizer:
    def __init__(self,space_symbol='▁'):
        self.space_symbol = space_symbol
        self.khaiii = khaiii.KhaiiiApi()
        self.space_symbol =space_symbol
        self.flatten = lambda l: list(itertools.chain(*l))
        self.is_nested = lambda l: isinstance(l[0], list)

    def text_to_morphs(self, text, hierarchy=True):
        text = space_normalize(text)
        res = self.khaiii.analyze(text)
        morphs = []
        for word in res:
            morph = []
            for idx, m in enumerate(word.morphs):
                if idx == 0:
                    morph.append(self.space_symbol + m.lex)
                else:
                    morph.append(m.lex)
            morphs.append(morph)

        if hierarchy:
            return morphs[0]
        else:
            return self.flatten(morphs)[0]

    def morphs_to_text(self, morphs):
        if self.is_nested(morphs):
            morphs = self.flatten(morphs)
        temp = re.sub('▁', ' ', ''.join(morphs)).strip()
        splited = jamotools.split_syllables(temp)
        joined = jamotools.join_jamos(splited)
        return joined
