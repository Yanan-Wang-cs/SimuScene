from utils import LLMClient
import asyncio  
import json
import os
import re
from tqdm import tqdm
import cv2
import argparse

def clean_response(text):
    # Remove <think> tags and their content
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
def extract_triple_single_quote_json(text):
    # Regex pattern to match content inside ''' '''
    pattern = r"```json(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches

parser = argparse.ArgumentParser()
parser.add_argument(
    "--process",
    default="main",
    choices=["main", "process_jsonl"],
)
parser.add_argument(
    "--model",
    default="Qwen3-32B",
)
parser.add_argument("--per-video-errors", action="store_true",
                    help="skip only a video OpenCV cannot open, instead of every video of that scenario")
args = parser.parse_args()

savejsonl = True
json_arr = []

test_model = args.model
prompt_index = 7
# Simple usage
def prompt7(questions):
    return f"""You are given a video and several visual verification questions. Your task is to judge each question as true or false based only on what can be seen or reasonably inferred from the video. If the visual evidence is insufficient to confirm the statement, or if the statement directly contradicts the video, answer "false".

Visual reasoning guidelines:

1. Perspective changes: Objects can look different from different angles. For example, a circular path may look curved, straight, or wavy depending on the viewpoint, and a cylinder may look like a circle from the top view or a rectangle from the side view.

2. Simplified or symbolic drawings: The video may use simple shapes or colors to represent real objects. If the motion or layout still matches the description, treat it as true.

3. Container boundaries: If no container is drawn, assume that the video frame is the boundary. If a container is shown, treat it as see-through only if the interior is visible. If something is not visible, do not assume that it is inside the container.

4. Shape and motion: Focus on shape and motion, and ignore assumptions about material, weight, texture, or color. If the described motion does not match the video, such as "slides down" versus actually moving up, answer false.

5. Occlusion: If something is partly hidden, use nearby visual cues to infer how it is moving or positioned.

Input:

- Questions: {questions}

Output format:

Return a JSON list of objects. Do not include conversational filler or Markdown formatting tags. Each object should contain the following fields:

- "index": The question index.
- "question": The full question text.
- "analysis": A concise step-by-step breakdown of what was observed versus what the question claims.
- "result": "true" if visually confirmed, and "false" if contradicted or if the evidence is insufficient.

Example format:

[
  {{
    "index": "1",
    "question": "The ball rolls along the circular path.",
    "analysis": "The red sphere moves in a consistent arc that completes a 360-degree loop.",
    "result": "true"
  }}
]
"""

def get_size_opencv(path: str) -> tuple[int, int]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h

def resize_to_constraints(W, H, max_area=280*480, divisor=28):
    aspect_ratio = W / H
    candidates = []

    # 遍历满足条件的所有可能的尺寸
    for target_h in range(divisor, 2048, divisor):
        target_w = int(target_h * aspect_ratio)
        if target_w <= 0:
            continue
        area = target_w * target_h
        if area <= max_area:
            candidates.append((area, target_w, target_h))

    if not candidates:
        raise ValueError("No valid size found under constraints.")

    # 选面积最大的那一组
    _, final_w, final_h = max(candidates, key=lambda x: x[0])
    return final_w, final_h

async def main():
    valid_question_num = 0
    Model_name = "Qwen/Qwen2.5-VL-72B-Instruct"
    index = 0
    client = None if savejsonl else LLMClient(model=Model_name, timeout_seconds=10000, temperature=0.0)
    # for i in range(0, 443, 1):
    for i in range(0, 25035, 1):
        folder_id = 0
        json_path = f'human_modify/val/video_{i}.json'
        video_folder = f'video/save_video/{test_model}/video_{i}'
        if not os.path.exists(json_path) or not os.path.exists(video_folder):
            continue
        try:
        # if True:
            with open(json_path, "r", encoding="utf-8") as f:
                item = json.load(f)
            origin_question = item['question']
            
            try:
                after_verify = item['vlm_questions']
            except:
                print(type(item['vlm_questions']))
                print(f'after_verify error: {json_path}')
                break
            if after_verify['enableAnnotator'] == 'No' or len(after_verify['vlm_questions']) == 0:
                continue
            valid_question_num += 1
            all_questions = ''
            all_weight= []
            try:
                for vlmindex in range(len(after_verify['vlm_questions'])):
                    subquestion = after_verify['vlm_questions'][vlmindex]['question']
                    all_questions += f'{vlmindex+1}. {subquestion}\n'
                    all_weight.append(after_verify['vlm_questions'][vlmindex].get('weight', 1.0))
            except:
                print(json_path)

            
            for video_name in os.listdir(video_folder):
                try:
                    new_width, new_height = resize_to_constraints(*get_size_opencv(f'{video_folder}/{video_name}'))
                except (RuntimeError, ValueError) as e:
                    if not args.per_video_errors:
                        raise
                    print('skip unreadable video:', f'{video_folder}/{video_name}', e)
                    continue
                input_message = prompt7(all_questions)
                messages = [
                    {
                        "role": "user",
                        "content": 
                            [{
                                "type": "text", 
                                "text": input_message
                            },
                            {
                                "type": "video",
                                "video":f"{video_folder}/{video_name}",
                                "resized_height": new_height,
                                "resized_width": new_width,
                                "fps": 5,
                            }]
                    }]
                    #                             
                if savejsonl:
                    json_arr.append({
                        "messages": messages,
                        "id": i,
                        "document": json_path,
                        "video": f"{video_folder}/{video_name}"
                    })
                else:
                    responses = await client.process_all_inputs([messages], model=Model_name)
                    try:
                        final_json = extract_triple_single_quote_json(responses[0][0])
                        if final_json:
                            print(final_json)
                    except:
                        print(video_name, responses)
        except Exception as e:
            print('error:', json_path, e)
    if savejsonl:
        with open(f"data/simuphy_vlm_{test_model}.jsonl", "w", encoding="utf-8") as f:
            for item in json_arr:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f'valid_question_num: {valid_question_num}')
    print(f'python call_local_llm_process_file.py --input-file data/simuphy_vlm_{test_model}.jsonl --model Qwen/Qwen2.5-VL-72B-Instruct --num-generations 1 --output-file data/simuphy_vlm_{test_model}_responses.jsonl --temperature 0.0')
    return
def transfer_signal(text):
    if '\n' in text:
        text = text[3:-3]
    text = text.replace("\"", "\\\"")
    return text
def process_jsonl():
    total_num = 0
    process_num = 0
    null_num = 0
    batch_index = 0
    with open(f"data/simuphy_vlm_{test_model}_responses.jsonl", "r", encoding="utf-8") as f:
        for i, row in tqdm(enumerate(f)):
            item = json.loads(row)
            total_num += 1
            folder_id = int(item['id']) // 2500
            id = item['id']
            document = item['document']
            video = item['video']
            with open(document, "r", encoding="utf-8") as f:
                setting = json.load(f)
            origin_question = setting['question']
            

            for index, response in enumerate(item['responses']):             
                if response is None:
                    null_num += 1
                    continue
                save_name = f"{os.path.basename(video).replace('.mp4', '')}_{batch_index}.json"

                try:
                    result = clean_response(response)
                except:
                    print('first:', response)
                    raise Exception(f'{json_folder}/{save_name}')
                    break
                try:
                    json.loads(result)
                except:
                    filter_result = extract_triple_single_quote_json(result)
                    if len(filter_result) > 0:
                        result = filter_result
                if type(result) != list:
                    result = [result]
                try:  
                    result[0] = result[0].replace('\\(', '\\\\(').replace('\\)', '\\\\)')
                    result[0] = result[0].replace('"\n', '",\n')
                    result[0] = re.sub(r',\n\s*}', r'\n}', result[0])
                    result[0] = re.sub(r':(\s)\n', r':\1', result[0])
                    result = json.loads(result[0])
                except:
                    try:
                        matches = re.findall(r': ".*"}?,\s*\n', result[0])
                        confidence_matches = re.findall(r'"confidence_score":\s*"?(\d+)"?,?\n', result[0])
                        new_result = []
                        for jindex in range(0, len(matches), 4):
                            new_result.append({
                                "index": transfer_signal(matches[jindex]),
                                "question": transfer_signal(matches[jindex+1]),
                                "analysis": transfer_signal(matches[jindex+2]),
                                "result": transfer_signal(matches[jindex+3]),
                                "confidence_score": confidence_matches[jindex//4]
                            })
                        result = new_result
                    except:
                        print(f'second error:{json_folder}/{save_name}')
                        # raise Exception(f'second error:{json_folder}/{save_name}')
                json_folder = f'result/valresult/{test_model}/batch_0/video_{id}'
                os.makedirs(json_folder, exist_ok=True)
                save_json = {
                    "document": document,
                    "question": origin_question,
                    "video": video,
                    "vlm_result": result,
                }
                with open(f'{json_folder}/{save_name}', 'w', encoding='utf-8') as f:
                    json.dump(save_json, f, indent = 4, ensure_ascii=False)
                process_num += 1
        print(f'process_num: {process_num}, total_num: {total_num}, null_num: {null_num}, error_num: {total_num - process_num - null_num}')

            

if __name__ == "__main__":
    mode = args.process

    if mode == 'main':
        print('get run.jsonl')
        asyncio.run(main())
    elif mode == 'process_jsonl':
        print('process_jsonl')
        process_jsonl()