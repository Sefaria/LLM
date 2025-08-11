import django
django.setup()
from sefaria.model.text import Ref
from translation.translation import translate_text

if __name__ == '__main__':
    text = Ref("Lechem Mishneh on Mishneh Torah, Hiring 10:1:1").text('he').text
    print(translate_text(text))