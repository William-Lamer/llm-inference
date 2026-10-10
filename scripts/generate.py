import argparse
import subprocess
import sys
import tempfile

from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Tokenize a prompt, run the engine, stream the decoded text.")
    parser.add_argument("prompt")
    parser.add_argument("--model", default="models/Qwen3-0.6B")
    parser.add_argument("--max-new", type=int, default=64)
    parser.add_argument("--chat", action="store_true", help="wrap the prompt in the chat template (no thinking)")
    parser.add_argument("--temperature", type=float, default=0.0, help="0 is greedy")
    parser.add_argument("--seed", type=int, help="default: the engine picks one from the clock")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    text = args.prompt
    if args.chat:
        messages = [{"role": "user", "content": args.prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
    prompt_ids = tokenizer(text)["input_ids"]

    with tempfile.NamedTemporaryFile("w", suffix=".txt") as f:
        f.write(" ".join(map(str, prompt_ids)) + "\n")
        f.flush()
        cmd = ["./build/engine", args.model, "--tokens", f.name, "--max-new", str(args.max_new),
               "--temperature", str(args.temperature)]
        if args.seed is not None:
            cmd += ["--seed", str(args.seed)]
        # The engine's stderr is a few status lines, printed after the text so they don't interleave.
        engine = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        assert engine.stdout is not None and engine.stderr is not None

        print(args.prompt, end="\n" if args.chat else "", flush=True)
        ids = []
        printed = ""
        word = ""
        # The engine flushes "ID " after every token. Decoding the whole list each
        # time keeps characters that span several tokens intact.
        while True:
            c = engine.stdout.read(1)
            if c == "":
                break
            if not c.isspace():
                word += c
                continue
            if word == "":
                continue
            ids.append(int(word))
            word = ""
            decoded = tokenizer.decode(ids, skip_special_tokens=True)
            print(decoded[len(printed):], end="", flush=True)
            printed = decoded
        print(flush=True)
        print(engine.stderr.read(), end="", file=sys.stderr)
        sys.exit(engine.wait())


if __name__ == "__main__":
    main()
