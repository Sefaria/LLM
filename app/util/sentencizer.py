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


if __name__ == '__main__':
    yo = """[26] See R. Joseph B. Soloveitchik, \"Tzedakah: Brotherhood and Fellowship,\" in Halakhic Morality: Essays on Ethics and Masorah, 126-127."""
    sents = sentencize(yo)
    print_completion(sents)
    pass


