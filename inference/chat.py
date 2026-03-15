from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import torch

MODEL_PATH = "D:/models/qwen3.5-2b"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float16, device_map="cuda")
print(f"Ready! ({torch.cuda.memory_allocated() / 1024**3:.1f} GB VRAM used)\n")

while True:
    try:
        prompt = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye!")
        break

    if not prompt:
        continue
    if prompt.lower() in ("exit", "quit"):
        print("Bye!")
        break

    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"formatted: {formatted}\n")
    inputs = tokenizer(formatted, return_tensors="pt").to("cuda")

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
        streamer=streamer,
    )

    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    print("Qwen: ", end="", flush=True)
    for token in streamer:
        print(token, end="", flush=True)
    print()

    thread.join()
