import csv
import json


def process_csv(input_file):
    output = []

    with open(input_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        reader.fieldnames = [field.strip() for field in reader.fieldnames]

        for row in reader:
            ref = row['Ref']

            # Parse slugs from the CSV columns
            current_slugs = set(eval(row['Currently Tagged Slugs']))
            possible_additional_slugs = set(eval(row['Possible Additioal Slugs']))
            add_slugs = set(eval(row['Add Slugs']))
            print(row['Remove Slugs'])
            remove_slugs = set(eval(row['Remove Slugs']))

            # Calculate the final set of slugs
            final_slugs = (current_slugs | add_slugs) - remove_slugs

            # Create the JSON object
            output.append({
                "ref": ref,
                "slugs": sorted(list(final_slugs))
            })

    return output


def save_json(data, output_file):
    with open(output_file, mode='w', encoding='utf-8') as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    input_file = "Revision of Golden Standard Topic Tagging - Sheet1.csv"  # Replace with your CSV file path
    output_file = "revised_gold.json"
    # Replace with your desired output file path

    aggregated_data = process_csv(input_file)
    save_json(aggregated_data, output_file)

    print(f"Aggregated data has been saved to {output_file}")