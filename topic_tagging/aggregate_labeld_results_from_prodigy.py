import json


def find_distinct_refs(file_path):
    distinct_refs = set()
    with open(file_path, 'r') as f:
        data = json.load(f)
        if isinstance(data, list):
            for json_obj in data:
                if 'meta' in json_obj and 'Ref' in json_obj['meta']:
                    distinct_refs.add(json_obj['meta']['Ref'])
        else:
            raise ValueError("The JSON file does not contain an array of JSON objects.")
    return list(distinct_refs)

def find_elements_with_ref(file_path, ref):
    elements_with_ref = []
    with open(file_path, 'r') as f:
        data = json.load(f)
        if isinstance(data, list):
            for json_obj in data:
                if 'meta' in json_obj and 'Ref' in json_obj['meta'] and json_obj['meta']['Ref'] == ref:
                    elements_with_ref.append(json_obj)
        else:
            raise ValueError("The JSON file does not contain an array of JSON objects.")
    return elements_with_ref

def aggregate_accepted_values(elements):
    accepted_values = []
    for element in elements:
        if 'answer' in element and element['answer'] == 'reject':
            continue
        elif 'accept' in element:
            accepted_values.extend(element['accept'])
    return sorted(accepted_values)

def write_list_of_dicts_to_jsonl(filename, list_of_dicts):
    with open(filename, 'w') as jsonl_file:
        for item in list_of_dicts:
            jsonl_file.write(json.dumps(item, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    print("Hi")
    aggregated_results = []
    file_path = 'topic_tagging_output.json'
    all_refs = find_distinct_refs(file_path)
    for ref in all_refs:
        elements_for_ref = find_elements_with_ref(file_path, ref)
        labels_for_ref = aggregate_accepted_values(elements_for_ref)
        aggregated_results.append({'ref': ref, 'slugs': labels_for_ref})
    write_list_of_dicts_to_jsonl('golden_standard_labels.jsonl', aggregated_results)
