import typer
import subprocess
from openai import File

def fine_tune(openai_api_key: str, results_file_id: str, output_file: str):
    # Check if required inputs are provided
    if not (openai_api_key and results_file_id and output_file):
        typer.echo("Error: Missing required parameters")
        typer.echo("Usage: ./fine_tune.py fine-tune <OPENAI_API_KEY> <RESULTS_FILE_ID> <OUTPUT_FILE>")
        raise typer.Exit(code=1)

    headers = {
        "Authorization": f"Bearer {openai_api_key}"
    }

    # Get via File object
    output_file_object = File.download(results_file_id, openai_api_key)

    # Get via File curl command
    curl_command = [
        "curl",
        f"https://api.openai.com/v1/files/{results_file_id}/content",
        "-H", f"-H Authorization: Bearer {openai_api_key}",
        "-o", output_file
    ]
    subprocess.run(curl_command, check=True, text=True)

if __name__ == "__main__":
    typer.run(fine_tune)
