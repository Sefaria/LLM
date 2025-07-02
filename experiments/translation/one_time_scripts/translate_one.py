import django
django.setup()
from sefaria.model.text import Ref
from translation.translation import translate_text

if __name__ == '__main__':
    text = Ref("Yedei Moshe on Bereshit Rabbah 45:6:2").text('he').text
    print(translate_text(text))