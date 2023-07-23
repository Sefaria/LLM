import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")


def create_fine_tune_job():

    model_engine = "ada"
    n_epochs = 3
    batch_size = 4
    learning_rate = 1e-5
    max_tokens = 1024

    training_file = "/Users/nss/Downloads/gpt_citation_training.jsonl"
    validation_file = "/Users/nss/Downloads/gpt_citation_validation.jsonl"

    # Create the fine-tuning job
    fine_tuning_job = openai.FineTune.create(
        model="ada",
        n_epochs=15,
        batch_size=4,
        training_file=os.path.abspath(training_file),
        validation_file=os.path.abspath(validation_file),
    )

    return fine_tuning_job["id"]


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
    job_id = create_fine_tune_job()
    monitor_fine_tune_job(job_id)


"""
After you’ve fine-tuned a model, remember that your prompt has to end with the indicator string ` \n\n###\n\n` for the model to start generating completions, rather than continuing with the prompt. Make sure to include `stop=[" ###"]` so that the generated texts ends at the expected place.
Once your model starts training, it'll approximately take 3.54 minutes to train a `curie` model, and less for `ada` and `babbage`. Queue will approximately take half an hour per job ahead of you.
"""