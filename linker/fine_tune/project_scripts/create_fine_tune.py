import typer
import os
import openai
from time import sleep
from tqdm import tqdm


def create_fine_tune_job(api_key: str, model: str, training_filename: str, validation_filename: str, suffix: str):
    training_file, validation_file = upload_files(api_key, training_filename, validation_filename)

    # wait until files are "processed" (:shrug:)
    print("waiting for 'processing'...")
    for _ in tqdm(range(30)):
        sleep(1)

    # Create the fine-tuning job
    fine_tuning_job = openai.FineTuningJob.create(
        api_key=api_key,
        model=model,
        training_file=training_file.id,
        validation_file=validation_file.id,
        suffix=suffix
    )

    return fine_tuning_job["id"]


def upload_files(api_key: str, training_filename: str, validation_filename: str):
    training_file = openai.File.create(api_key=api_key, file=open(training_filename, "r"), purpose='fine-tune')
    validation_file = openai.File.create(api_key=api_key, file=open(validation_filename, "r"), purpose='fine-tune')
    return training_file, validation_file


def monitor_fine_tune_job(job_id):
    import time

    while True:
        fine_tuning_status = openai.FineTune.get_status(job_id)
        status = fine_tuning_status["status"]
        print(f"Fine-tuning job status: {status}")

        if status in ["completed", "failed"]:
            break

        time.sleep(60)


if __name__ == '__main__':
    typer.run(create_fine_tune_job)

