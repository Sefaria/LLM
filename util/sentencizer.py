import stanza


def sentencize(text):
    nlp = stanza.Pipeline(lang='en', processors='tokenize', tokenize_no_ssplit=False)
    doc = nlp(text)
    return [sent.text for sent in doc.sentences]


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


