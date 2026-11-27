import ast
import json
import re
import tempfile
import subprocess
import os
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
import gc
import psutil
import time
from code_utils import sanitize_code
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--process_batch",
    type=str,
    default="Qwen3-235B-A22B",
    help="model / process batch name"
)
parser.add_argument("--allow-unfenced", action="store_true",
                    help="when an answer has no ```python block, run the whole answer if it parses as Python")
parser.add_argument("--pass-stem", action="store_true",
                    help="run each script inside its video folder and pass only the file stem as `name`, as the prompt describes")
parser.add_argument("--retry-name-option", action="store_true",
                    help="when a script rejects the positional name with an argparse usage error, run it again with `--name <name>`")
parser.add_argument("--keep-pip-lines", action="store_true",
                    help="leave lines that contain `pip` as they are and block installs with PIP_NO_INDEX=1 instead")
parser.add_argument("--num-processes", type=int, default=max(1, int(cpu_count() * 0.75)))
args = parser.parse_args()

response_num = 8

process_batch = args.process_batch
ALLOW_UNFENCED, PASS_STEM = args.allow_unfenced, args.pass_stem
RETRY_NAME_OPTION, KEEP_PIP_LINES = args.retry_name_option, args.keep_pip_lines

num = 0

def clean_response(text):
    # Remove <think> tags and their content
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

def extract_triple_single_quote_json(text):
    # Regex pattern to match content inside ''' '''
    pattern = r"```json\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches

def extract_triple_single_quote_code(text):
    # Regex pattern to match content inside ''' '''
    pattern = r"```python\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches

def is_python(text):
    try:
        ast.parse(text)
    except SyntaxError:
        return False
    return True

def no_install(*args, **kwargs):
    raise RuntimeError("No install")

def run_limited_subprocess(cmd, mem_limit_mb=2048, timeout=300, cwd=None, env=None):
    if env is None:
        env = os.environ.copy()  
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    p = psutil.Process(proc.pid)
    start_time = time.time()

    try:
        while proc.poll() is None:
            try:
                mem_mb = p.memory_info().rss / (1024 * 1024)
                if mem_mb > mem_limit_mb:
                    proc.kill()
                    print(f"[MEM KILL] PID {proc.pid} memory {mem_mb:.2f}MB > {mem_limit_mb}MB")
                    out, err = proc.communicate(timeout=2)
                    if out: print("[STDOUT]\n", out)
                    if err: print("[STDERR]\n", err)
                    return False
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break

            if time.time() - start_time > timeout:
                proc.kill()
                print(f"[TIMEOUT] PID {proc.pid} exceeded {timeout}s")
                out, err = proc.communicate(timeout=2)
                if out: print("[STDOUT]\n", out)
                if err: print("[STDERR]\n", err)
                return False
            time.sleep(0.25)

        out, err = proc.communicate(timeout=2)
        if proc.returncode != 0:
            if err: 
                raise Exception(err)
            return False
        return True
    finally:
        try:
            proc.kill()
        except Exception:
            pass


def process_video(args):
    """Process a single video generation task"""
    id, video_id, response = args
    tmp_path = None
    save_path = f"save_video/{process_batch}/video_{id}"
    video_name = f"video_{id}_{video_id}.mp4"
    response = clean_response(response)
    question = extract_triple_single_quote_code(response)
    if not question and ALLOW_UNFENCED and response.strip() and is_python(response):
        question = [response]

    if not question:
        # print(f"Warning: No Python code found for video_{id}_{video_id}")
        return False        
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(f'code/{process_batch}', exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(suffix=".py", mode='w', delete=False, dir=f'code/{process_batch}') as tmp:
            code = question[0]
            for i in range(len(question)):
                if len(question[i]) > len(code):
                    code = question[i]
            code = code.replace('\"', '"')
            code = sanitize_code(code)
            if not KEEP_PIP_LINES:
                pattern = r"^(\s*).*pip.*$"
                code = re.sub(pattern, r"\1pass", code, flags=re.M)
            tmp.write(code)
            tmp_path = tmp.name
        s = f'python3 {tmp_path} {os.path.join(save_path, video_name)}'
        with open(f'log/code_running_{process_batch}.txt', "a", encoding="utf-8") as f:
            if not s.endswith("\n"):          
                s += "\n"
            f.write(s)
        # Run the video generation
        cmd = [sys.executable, os.path.abspath(tmp_path)]
        name = video_name[:-len('.mp4')] if PASS_STEM else os.path.join(save_path, video_name)
        run = partial(
            run_limited_subprocess,
            cwd=save_path if PASS_STEM else None,
            mem_limit_mb=3584,
            timeout=300,
            env={**os.environ, 'PIP_NO_INDEX': '1'} if KEEP_PIP_LINES else os.environ.copy(),
            )
        try:
            result = run(cmd + [name])
        except Exception as e:
            if not (RETRY_NAME_OPTION and 'usage:' in str(e) and 'error:' in str(e)):
                raise
            result = run(cmd + ['--name', name])
            
            # Clean up temporary file
        # if tmp_path and os.path.exists(tmp_path):
        #     os.unlink(tmp_path)
        del result
        gc.collect()
        
        return True
            
            # if result.returncode == 0:
            #     print(f"Successfully generated: {video_name}")
            # else:
            #     print(f"Error generating {video_name}: {result.stderr}")
            
    except Exception as e:
        try:
            s = f'video_{id}_{video_id}: {e.text}: {str(e)}'
        except:
            s = f'video_{id}_{video_id}: {str(e)}'
        with open(f'log/code_error_{process_batch}.txt', "a", encoding="utf-8") as f:
            s.replace('\n', ' ')
            if not s.endswith("\n"):          
                s += "\n"
            f.write(s)
        pass
        # print(f"Exception processing video_{id}_{video_id}: {str(e)}")

def main():
    # Read all data first
    all_tasks = []
    
    with open(f'../data/simuphy_genvideo_trained_responses8_{process_batch}.jsonl', 'r') as f:
        for id, i in enumerate(f):
            item = json.loads(i)
            print(f"Processing item {id} with {len(item['responses'])} responses")
            
            for video_id, response in enumerate(item['responses']):
                if response:
                    all_tasks.append((item['video_index'], video_id, response))
    
    print(f"Total tasks to process: {len(all_tasks)}")
    # with open('all_tasks.json', 'w') as f:
    #     json.dump(all_tasks, f)
    # Determine number of processes (use 75% of available CPUs to avoid overwhelming the system)
    num_processes = args.num_processes
    print(f"Using {num_processes} processes")
    
    # Process videos in parallel
    with Pool(processes=num_processes, maxtasksperchild=1) as pool:
        for _ in tqdm(pool.imap(process_video, all_tasks, chunksize=1), total=len(all_tasks)):
            pass

    # with Pool(processes=num_processes) as pool:
    #     # Use imap for better memory management with large datasets
    #     for i, _ in enumerate(pool.imap(process_video, all_tasks, chunksize=10)):
    #         if (i + 1) % 100 == 0:
    #             print(f"Processed {i + 1}/{len(all_tasks)} tasks")

if __name__ == "__main__":
    subprocess.check_call = no_install
    if response_num == 8:
        main()
    
     
    
           
            
