#!/home/carlesoctav/personal/torchy2/wordle/.venv/bin/python
import argparse
import datetime
import json
import signal
import sys
from huggingface_hub import hf_hub_download
from jinja2.sandbox import ImmutableSandboxedEnvironment
import jinja2.ext

def raise_exception(message):
    raise jinja2.exceptions.TemplateError(message)

def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

def strftime_now(format_str):
    return datetime.datetime.now().strftime(format_str)

def build_jinja_renderer(model_name: str):
    cfg_path = hf_hub_download(model_name, "tokenizer_config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    env = ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=["jinja2.ext.loopcontrols"],
    )
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = strftime_now
    template = env.from_string(cfg["chat_template"])
    special_tokens = {k: v for k, v in cfg.items() if k.endswith("_token") and v}

    def render(messages, add_generation_prompt=False):
        return template.render(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            **special_tokens,
        )

    return render

def build_tokenizer_renderer(model_name: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)

    def render(messages, add_generation_prompt=False):
        return tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    return render

def main():
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)

    parser = argparse.ArgumentParser(
        description="Stream JSONL, apply Hugging Face chat_template, and output enriched JSONL or raw text for jq / visidata."
    )
    parser.add_argument("path", nargs="?", default="-", help="Path to JSONL file (default: '-' for stdin)")
    parser.add_argument("-m", "--model", default="Qwen/Qwen3-1.7B", help="Hugging Face model repository ID")
    parser.add_argument("-r", "--raw", action="store_true", help="Print raw rendered template text instead of JSON")
    parser.add_argument("--add-generation-prompt", action="store_true", help="Pass add_generation_prompt=True to chat template")
    parser.add_argument("--full-tokenizer", action="store_true", help="Use transformers.AutoTokenizer instead of fast Jinja2")
    args = parser.parse_args()

    if args.full_tokenizer:
        render = build_tokenizer_renderer(args.model)
    else:
        render = build_jinja_renderer(args.model)

    in_stream = sys.stdin if args.path == "-" else open(args.path)

    for line in in_stream:
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if "messages" in record:
            messages = record["messages"]
        else:
            messages = record["prompt"] + record["completion"]

        rendered = render(messages, add_generation_prompt=args.add_generation_prompt)

        if args.raw:
            sys.stdout.write(rendered + "\n")
        else:
            record["rendered"] = rendered
            sys.stdout.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
