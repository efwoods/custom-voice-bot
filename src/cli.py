import typer
import subprocess
from pathlib import Path

app = typer.Typer(help="ChatBot voice bot CLI")


@app.command()
def ocr(
    input_dir: str = "data/raw_images", out: str = "data/processed/chatbot_raw.jsonl"
):
    subprocess.run(
        [
            "python",
            "src/ocr_extract.py",
            "--input_dir",
            input_dir,
            "--output_file",
            out,
        ],
        check=True,
    )


@app.command()
def clean(
    inp: str = "data/processed/chatbot_raw.jsonl",
    out: str = "data/processed/chatbot_clean.jsonl",
):
    subprocess.run(
        ["python", "src/clean_messages.py", "--input_file", inp, "--output_file", out],
        check=True,
    )


@app.command()
def vector(
    inp: str = "data/processed/chatbot_clean.jsonl",
    persist: str = "data/processed/chroma_chatbot",
):
    subprocess.run(
        [
            "python",
            "src/build_vectorstore.py",
            "--input_file",
            inp,
            "--persist_dir",
            persist,
        ],
        check=True,
    )


@app.command()
def chat_prompt(
    vector_dir: str = "data/processed/chroma_chatbot",
    model_path: str = "models/qwen2-7b-instruct.gguf",
):
    subprocess.run(
        [
            "python",
            "src/chat_prompt_mode.py",
            "--vector_dir",
            vector_dir,
            "--model_path",
            model_path,
        ],
        check=True,
    )


@app.command()
def finetune(
    jsonl: str = "data/processed/chatbot_clean.jsonl",
    base_model: str = "Qwen/Qwen2-7B-Instruct",
    out: str = "data/models/lora/qwen2-7b-chatbot",
):
    subprocess.run(
        [
            "python",
            "src/finetune_lora.py",
            "--jsonl",
            jsonl,
            "--base_model",
            base_model,
            "--output_dir",
            out,
        ],
        check=True,
    )


@app.command()
def chat_lora(
    base_model: str = "Qwen/Qwen2-7B-Instruct",
    lora_dir: str = "data/models/lora/qwen2-7b-chatbot",
    vector_dir: str = "data/processed/chroma_chatbot",
):
    subprocess.run(
        [
            "python",
            "src/serve_lora.py",
            "--base_model",
            base_model,
            "--lora_dir",
            lora_dir,
            "--vector_dir",
            vector_dir,
        ],
        check=True,
    )


if __name__ == "__main__":
    app()
