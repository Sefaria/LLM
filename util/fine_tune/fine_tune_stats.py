import typer
import subprocess
import openai
import os
# from openai import File

def fine_tune(results_file_id: str, output_file: str):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    content = openai.File.download(results_file_id)
    with open(output_file, 'wb') as file:
        file.write(content)

if __name__ == "__main__":
    typer.run(fine_tune)
