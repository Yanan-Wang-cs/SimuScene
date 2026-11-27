"""Offline batch driver for step 3: judge data/simuphy_vlm_<MODEL>.jsonl with vLLM, no server needed.

Reads the requests written by `step3_vlmjudge.py --process main` (qwen-vl-utils style messages:
local video, resized_height/width, fps 5). Frames are extracted with qwen_vl_utils.process_vision_info,
so sampling follows each request, and judged at temperature 0. Appends {**request, "responses": [text]}
to data/simuphy_vlm_<MODEL>_responses.jsonl, the file `--process process_jsonl` reads. Resumable by
video path. A video that cannot be decoded (e.g. a 1-frame file) gets "responses": [None], which
process_jsonl counts as null.

    python judge_vllm_offline.py --model-tag <MODEL> [--judge Qwen/Qwen2.5-VL-72B-Instruct] [--tp 2]
"""
import argparse
import json
from pathlib import Path

from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--judge", default="Qwen/Qwen2.5-VL-72B-Instruct")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--gpu-mem", type=float, default=0.40)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    src = Path(f"data/simuphy_vlm_{args.model_tag}.jsonl")
    dst = Path(f"data/simuphy_vlm_{args.model_tag}_responses.jsonl")
    items = [json.loads(line) for line in src.open(encoding="utf-8")]
    done = {json.loads(line)["video"] for line in dst.open(encoding="utf-8")} if dst.exists() else set()
    todo = [it for it in items if it["video"] not in done]
    print("TODO", len(todo), "DONE_BEFORE", len(done), flush=True)

    processor = AutoProcessor.from_pretrained(args.judge)
    llm = LLM(model=args.judge, tensor_parallel_size=args.tp, gpu_memory_utilization=args.gpu_mem,
              max_model_len=32768, limit_mm_per_prompt={"video": 1, "image": 0})
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    with dst.open("a", encoding="utf-8") as out:
        for start in range(0, len(todo), args.batch):
            chunk, reqs, kept = todo[start:start + args.batch], [], []
            for it in chunk:
                try:
                    _, videos, vkw = process_vision_info(it["messages"], return_video_kwargs=True, return_video_metadata=True)
                    # recent vLLM needs per-video metadata; keep the sampled fps that qwen-vl-utils reports without it
                    (_, m), = videos  # one video per request
                    vkw = {**vkw, "fps": len(m["frames_indices"]) / m["total_num_frames"] * m["fps"]}
                except Exception as e:  # unreadable video: no judgment, counted as null by process_jsonl
                    out.write(json.dumps({**it, "responses": [None], "judge_error": str(e)[:300]}, ensure_ascii=False) + "\n")
                    continue
                prompt = processor.apply_chat_template(it["messages"], tokenize=False, add_generation_prompt=True)
                reqs.append({"prompt": prompt, "multi_modal_data": {"video": videos}, "mm_processor_kwargs": vkw})
                kept.append(it)
            for it, res in zip(kept, llm.generate(reqs, sp) if reqs else []):
                out.write(json.dumps({**it, "responses": [res.outputs[0].text]}, ensure_ascii=False) + "\n")
            out.flush()
            print("JUDGED", start + len(chunk), "/", len(todo), flush=True)
    print("JUDGE_DONE", flush=True)


if __name__ == "__main__":
    main()
