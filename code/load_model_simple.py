import os, argparse, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ.setdefault("HF_HOME", "/data/briankim0809/hf_cache")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/data/briankim0809/hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "/data/briankim0809/hf_cache")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="The correct answer to the multiple choice question is (")
    args = parser.parse_args()

    print(f"Loading {args.model} ...", flush=True)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    toker = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    print("GPU memory after load:", flush=True)
    for i in range(torch.cuda.device_count()):
        used = torch.cuda.memory_allocated(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i} ({torch.cuda.get_device_name(i)}): {used:.1f}/{total:.1f} GB", flush=True)

    # device_map='auto' distributes across GPUs; send input to first layer's device
    first_device = next(iter(model.hf_device_map.values())) if hasattr(model, "hf_device_map") else model.device
    inputs = toker(args.prompt, return_tensors="pt").to(first_device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    print("Output:", toker.decode(out[0], skip_special_tokens=True), flush=True)


if __name__ == "__main__":
    main()
