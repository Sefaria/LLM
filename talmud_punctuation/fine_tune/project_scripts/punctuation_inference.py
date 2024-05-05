import csv
import django
django.setup()
import typer
from sefaria.model import *
import os
import openai
import re
from tqdm import tqdm


api_key = os.getenv("OPENAI_API_KEY")
class PunctuationOracle:
    def __init__(self, model_name,
                 system_message='Punctuate this Talmudic passage based on the commentary I provide. Extract the relevant punctuation marks (, : . ? ! \"\" ; —) from the commentary and put them in the original. Output only the original Aramaic passage with punctuation without \"cantilation\" or \"nikud\".\n'
                 ):
        self.model_name = model_name
        self.system_message = system_message

    def _ask_OpenAI_model(self, original_passage, commentary):
        user_message = "Original Talmudic Passage:\n" + original_passage + '\n' + "Commentary:\n" + commentary
        messages = [
                {
                    "role": "system",
                    "content": self.system_message
                },
                {
                    "role": "user",
                    "content": user_message
                }

            ]
        params = {
            "temperature": 1,
            "max_tokens": 600,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0
        }
        try:
            response = openai.ChatCompletion.create(**params, model=self.model_name, messages=messages)
            inference = response["choices"][0]["message"]["content"]
        # hack to handle this weird exception: "Failed to create completion as the model generated invalid Unicode output."
        except:
            stream = openai.ChatCompletion.create(**params, model=self.model_name, messages=messages, stream=True)
            response_buffer = ''
            try:
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        # print(chunk.choices[0].delta.content, end="")
                        response_buffer += chunk.choices[0].delta.content
            except:
                messages.append({"role": "assistant", "content": response_buffer})
                messages.append({"role": "user", "content": "restart"})
                response = openai.ChatCompletion.create(**params, model=self.model_name, messages=messages)
                inference = response["choices"][0]["message"]["content"]

        return inference

    def _is_subsequence(self, sub, main):
        it = iter(main)
        return all(item in it for item in sub)
    def _remove_tokens(self, regex_pattern, text):
        return re.sub(regex_pattern, '', text)

    def _punctuate_single_word(slef, punctuated_word, unpunctuated_word):
        punctuations_end_one_char = {'.', ',', ';', ':', '!', '?', "״"}
        punctuations_end_two_chars = {'?!'}
        punctuated_word_no_heifen = punctuated_word.replace('—', '')

        if len(punctuated_word_no_heifen) >= 4 and punctuated_word[-3] in punctuations_end_one_char and \
                punctuated_word_no_heifen[-2] in punctuations_end_one_char and punctuated_word_no_heifen[
            -1] in punctuations_end_one_char:
            unpunctuated_word += punctuated_word_no_heifen[-3:]
        elif len(punctuated_word_no_heifen) >= 3 and punctuated_word_no_heifen[-2] in punctuations_end_one_char and \
                punctuated_word_no_heifen[-1] in punctuations_end_one_char:
            unpunctuated_word += punctuated_word_no_heifen[-2:]
        elif len(punctuated_word_no_heifen) >= 2 and punctuated_word_no_heifen[-1] in punctuations_end_one_char:
            unpunctuated_word += punctuated_word_no_heifen[-1]

        if len(punctuated_word_no_heifen) >= 2 and punctuated_word_no_heifen[0] == "״":
            unpunctuated_word = "״" + unpunctuated_word

        if punctuated_word.endswith('—'):
            unpunctuated_word += ' —'

        return unpunctuated_word
    def _punctuate_vocalized(self, punctuated_text: str, vocalised_text: str) -> str:
        #####
        punctuated_text = punctuated_text.replace("–", "—")

        punctuated_text_list = punctuated_text.replace(' —', '—').split()
        vocalised_text_list = vocalised_text.split()
        vocalised_text_list_suffix = vocalised_text.split()
        # if len(punctuated_text_list) != len(vocalised_text_list):
        #     print("Oh!")
        punctuationre = re.compile(
            r'[\.\!\?\:\,\u05F4]+(?![\u0591-\u05bd\u05bf-\u05c5\u05c7\u200d\u05d0-\u05eA](?:[\.\!\?\:\,\u05F4\s]|$))|—\s')
        matches = []
        global_vocalized_index = 0
        for puncutated_word in punctuated_text_list:
            unpuncutated_word = puncutated_word.replace('—', '')
            unpuncutated_word = self._remove_tokens(punctuationre, unpuncutated_word)
            for index, vocalised_word in enumerate(vocalised_text_list_suffix):
                if self._is_subsequence(list(unpuncutated_word), list(vocalised_word)):
                    vocalised_text_list_suffix = vocalised_text_list_suffix[index + 1:]
                    global_vocalized_index += index
                    matches += [(puncutated_word, vocalised_word, global_vocalized_index)]
                    vocalised_text_list[global_vocalized_index] = self._punctuate_single_word(puncutated_word, vocalised_word)
                    global_vocalized_index += 1
                    break

        return ' '.join(vocalised_text_list)

    def punctuate_and_vocalize_segments(self, segment_refs: list[Ref]):
        result = []
        for seg_ref in tqdm(segment_refs, desc="Processing segments", unit="segment"):
            result.append(self.punctuate_and_vocalize_segment(seg_ref))
        return result

    def punctuate_and_vocalize_segment(self, seg_ref: Ref):
        raw_text = seg_ref.text('he', "William Davidson Edition - Aramaic").text
        commentary = Ref("Steinsaltz on " + seg_ref.tref).text('he').text

        raw_punctuated = self._ask_OpenAI_model(raw_text, commentary)
        raw_vocalized = seg_ref.text('he', "William Davidson Edition - Vocalized Aramaic").text
        punctuated_vocalized = self._punctuate_vocalized(raw_punctuated, raw_vocalized)
        result = {"ref": seg_ref.tref, "punctuated": punctuated_vocalized}
        return result

    def punctuate_segments_and_write_to_csv(self, segment_refs: list[Ref], csv_filename):
        punctuated_segments = self.punctuate_and_vocalize_segments(segment_refs)
        with open(csv_filename, 'w', newline='') as csvfile:
            fieldnames = punctuated_segments[0].keys()
            csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            csv_writer.writeheader()
            csv_writer.writerows(punctuated_segments)

def punctuate_segments_and_write_to_csv(csv_filename, segment_refs, oracle: PunctuationOracle):
    existing_refs = set()
    try:
        with open(csv_filename, 'r', newline='') as fin:
            reader = csv.DictReader(fin)
            for row in reader:
                existing_refs.add(row['ref'])
    except FileNotFoundError:
        pass  # If the file doesn't exist yet, ignore the error and proceed


    fout = open(csv_filename, 'a', newline='')
    cout = csv.DictWriter(fout, ['ref', 'punctuated'])
    if fout.tell() == 0:
        cout.writeheader()
    for segment_oref in tqdm(segment_refs, desc=f"Punctuating"):
        if segment_oref.normal() in existing_refs:
            continue
        punctuated_dict = oracle.punctuate_and_vocalize_segment(segment_oref)
        cout.writerow(punctuated_dict)
    fout.close()










if __name__ == '__main__':
    oracle_new = PunctuationOracle(model_name="ft:gpt-3.5-turbo-0125:sefaria:he-punct:9K2PHRwC")
    oracle_old = PunctuationOracle(model_name="ft:gpt-3.5-turbo-0613:sefaria:he-punct:8ottZMB1")
    # refs = Ref('Bava Metzia 75b:14 - 119a:6').all_segment_refs()
    refs = Ref('Bava Metzia 75b:14 - 119a:6').all_segment_refs()
    # oracle_old.punctuate_and_vocalize_segment(Ref('Bava Metzia 75b:15'))
    # print(oracle_new.punctuate_and_vocalize_segment(Ref('Bava Metzia 75b:15')))
    # punctuate_segments_and_write_to_csv('bava_metzia_punctuated_75b_14_to_end.csv' , refs , oracle_new)

    punctuate_segments_and_write_to_csv('bava_metzia_punctuated_75b_14_to_end_new_model.csv', refs, oracle_new)
