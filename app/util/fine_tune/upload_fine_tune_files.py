import typer
import json
from openai import OpenAI


def upload_files(training_filename: str, validation_filename: str):
    client = OpenAI()
    training_file = client.files.create(file=open(training_filename, "rb"), purpose='fine-tune')
    validation_file = client.files.create(file=open(validation_filename, "rb"), purpose='fine-tune')
    out = {
        "training_file_id": training_file.id,
        "validation_file_id": validation_file.id
    }
    with open("output/fine_tune_file_ids.json", "w") as fout:
        json.dump(out, fout)


if __name__ == '__main__':
    typer.run(upload_files)
