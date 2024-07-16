import stanza


def sentencize(text, min_words=5):
    nlp = stanza.Pipeline(lang='en', processors='tokenize', tokenize_no_ssplit=False, download_method=None, verbose=False)
    doc = nlp(text)
    return _combine_small_sentences([sent.text for sent in doc.sentences], min_words=min_words)


def _num_words(text):
    return len(text.split())


def _concat_sentences(*sentences):
    return " ".join(sentences)


def _combine_small_sentences(sentences, min_words=5):
    final_sentences = []
    prev_sentence_was_small = False
    for sentence in sentences:
        if _num_words(sentence) < min_words:
            prev_sentence_was_small = True
            if len(final_sentences) > 0:
                final_sentences[-1] = _concat_sentences(final_sentences[-1], sentence)
            else:
                final_sentences += [sentence]
        elif prev_sentence_was_small:
            prev_sentence_was_small = False
            final_sentences[-1] = _concat_sentences(final_sentences[-1], sentence)
        else:
            final_sentences += [sentence]
    return final_sentences


def print_completion(sentences):
    print("====OUTPUT====")
    for i, sentence in enumerate(sentences):
        print(f'====== Sentence {i+1} tokens =======')
        print(sentence)


def claude_sentencizer(text, max_sentences=None):
    text_left = text[:]
    sentences = []
    while len(text_left) > 0 and (max_sentences is None or len(sentences) < max_sentences):
        next_sentence = claude_sentencizer_first_sentence(text_left)
        if next_sentence is None:
            break
        istart = text_left.index(next_sentence)
        text_left = text_left[istart+len(next_sentence):]
        sentences.append(next_sentence)
    return sentences


def claude_sentencizer_first_sentence(text):
    from basic_langchain.chat_models import ChatAnthropic
    from basic_langchain.schema import SystemMessage, HumanMessage
    from util.general import get_by_xml_tag
    system = SystemMessage(content="Given a text discussing Torah topics will little to no punctuation, "
                                   "output the first sentence. Input is in <input> tags. The first sentence "
                                   "should be output verbatim as it appears in <input> wrapped in "
                                   "<first_sentence> tags. Since the input text has no punctuation, use your judgement as to where the first sentence ends. Prefer smaller sentences.")
    human = HumanMessage(content=f"<input>{text}</input>")
    llm = ChatAnthropic("claude-3-5-sonnet-20240620", temperature=0)
    response = llm([system, human])
    return get_by_xml_tag(response.content, "first_sentence")


if __name__ == '__main__':
    import django
    django.setup()
    from sefaria.model import *
    yo = """[26] See R. Joseph B. Soloveitchik, \"Tzedakah: Brotherhood and Fellowship,\" in Halakhic Morality: Essays on Ethics and Masorah, 126-127."""
    sup = Ref("Sifra, Vayikra Dibbura DeNedavah, Section 4:1").text('he').text
    # sents = sentencize(yo)
    # print_completion(sents)
    sents = claude_sentencizer(sup, 5)
    for sent in sents:
        print(sent)
    pass


