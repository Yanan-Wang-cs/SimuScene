import argparse
import json
import os
import shutil
from collections import Counter
import glob
import re
def count_log(txt_file):
    pattern = re.compile(r"video_(\d+)")  

    video_id_num = set()
    video_nums = set()
    with open(txt_file, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                video_id = match.group(1)  
                video_id_num.add(video_id)    
            m = re.search(r'([^/\s]+\.mp4)', line)
            if m:
                video_name = m.group(1)  
                video_nums.add(video_name)   
    return len(video_nums), len(video_id_num)

def test_filter(test_model):
    human_label_arr = []
    with open(f'result/humanlabel.txt', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                human_label_arr.append(line.split('/')[-1].replace('.mp4.mp4', '.mp4'))
    video_ids = set()
    video_playable = set()
    concept_video_counts = Counter()
    concept_question_counts = Counter()
    domain_big_video_counts = Counter()
    domain_big_question_counts = Counter()
    domain_diff_level_counts = Counter()
    domain_diff_level_question_counts = Counter()
    diff_level_val_question_counts = Counter()
    diff_level_val_video_counts = Counter()
    test_total_domain_counts = Counter()

    check_folder = f'result/valresult/{test_model}'

    threshold_pass = 0

    file_error = 0
    video_error = 0

    subquestion_true_num = 0
    subquestion_false_num = 0


    subquestion_tt = 0
    subquestion_ff = 0
    subquestion_tf = 0
    subquestion_ft = 0

    for i in range(0, 25040):
        folder_id = 0
        if not os.path.exists(f'{check_folder}/batch_{folder_id}/video_{i}'):
            continue
        with open(f'human_modify/val/video_{i}.json', 'r', encoding='utf-8') as df:
            domain_json = json.load(df)

        concept = domain_json['type']
        diff_level = domain_json['Level']
        domain_big = domain_json['domain']

        diff_level_val_question_counts[diff_level] += 1
        test_total_domain_counts[domain_big] += 1

        question_save = False
        video_playable.add(i)
        for j in range(8):
            json_path = f'{check_folder}/batch_0/video_{i}/video_{i}_{j}_0.json'
            if not os.path.exists(json_path):
                continue
            diff_level_val_video_counts[diff_level] += 1
            gt_video = False
            judge_video = False
            
            if f'video_{i}_{j}.mp4' not in human_label_arr:
                gt_video = False
            else:
                gt_video = True

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data['vlm_result'], (dict, list)):
                video_error += 1
                continue

            vlm_result = data['vlm_result']
            score = 0
            for k in range(len(vlm_result)):
                if isinstance(vlm_result[k], str):
                    file_error += 1
                    score += 1
                    continue
                
                if vlm_result[k]['result'] != 'true' and vlm_result[k]['result'] != 'True' :
                    score += 1
            if score <= threshold_pass:
                concept_video_counts[concept] += 1
                domain_big_video_counts[domain_big] += 1
                domain_diff_level_counts[diff_level] += 1
                if not question_save:
                    domain_big_question_counts[domain_big] += 1
                    concept_question_counts[concept] += 1
                    domain_diff_level_question_counts[diff_level] += 1
                    question_save = True
                    
                judge_video = True
                video_ids.add(i) 
            else:
                judge_video = False

            video_path = f'video/save_video/{test_model}/video_{i}/video_{i}_{j}.mp4'
            # if judge_video:
            #     os.makedirs(f'human_rl/{test_model}/true', exist_ok=True)
            #     shutil.copy(video_path, f'human_rl/{test_model}/true/{os.path.basename(video_path)}')
            # else:
            #     os.makedirs(f'human_rl/{test_model}/false', exist_ok=True)
            #     shutil.copy(video_path, f'human_rl/{test_model}/false/{os.path.basename(video_path)}')
                
            if judge_video and gt_video:
                subquestion_tt += 1
            elif not judge_video and not gt_video:
                subquestion_ff += 1
            elif judge_video and not gt_video:
                if test_model == 'deepseek0528':
                    os.makedirs(f'human_qwen2/{test_model}/true_false', exist_ok=True)
                    try:
                        shutil.copy(video_path, f'human_qwen2/{test_model}/true_false/{os.path.basename(video_path)}')
                    except:
                        pass
                subquestion_tf += 1
            elif not judge_video and gt_video:
                if test_model == 'deepseek0528':
                    os.makedirs(f'human_qwen2/{test_model}/false_true', exist_ok=True)
                    try:
                        shutil.copy(video_path, f'human_qwen2/{test_model}/false_true/{os.path.basename(video_path)}')
                    except: 
                        pass
                subquestion_ft += 1
            else:
                raise Exception('?????')

            if judge_video:
                subquestion_true_num += 1
            else:
                subquestion_false_num += 1
    output_file = f'result/output/{test_model}.txt'

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    log_video, log_question = count_log(f'video/log/code_running_{test_model}.txt')
    print(log_video, log_question)
    with open(output_file, "w", encoding="utf-8") as f:  # 'a' 追加模式，'w' 覆盖模式
        f.write(f"{test_model}: file_error: {file_error}, video_error: {video_error}\n\n\n")
        if test_model == 'deepseek0528':
            f.write('VLM & human eval:')
            f.write(f"{test_model}: subquestion_tt: {subquestion_tt}, subquestion_ff: {subquestion_ff}, subquestion_tf: {subquestion_tf}, subquestion_ft: {subquestion_ft}\n")
            f.write(f'Agreement: {(subquestion_tt+subquestion_ff)*100/(subquestion_true_num+subquestion_false_num)}')
        else:
            f.write('VLM evaluate:')
            f.write(f"{test_model}: true: {subquestion_true_num}, false: {subquestion_false_num}\n")

# Avg@8
        f.write(f'''
videos:
Executable Rate (E.R.): {log_video*100/(334*8)},
Render Rate (R.R.):{len(glob.glob(f'video/save_video/{test_model}/video_*/*.mp4'))*100/(334*8)}, 
Playable Rate (P.R.):{(subquestion_true_num+subquestion_false_num)*100/(334*8)},
Acc:{subquestion_true_num*100/(334*8)}
''')
# Pass@8
        f.write(f'''
questions:
Executable Rate (E.R.): {log_question*100/334},
Render Rate (R.R.):{len(os.listdir(f'video/save_video/{test_model}'))*100/334}, 
Playable Rate (P.R.): {len(video_playable)*100/334},
Acc:{len(video_ids)*100/334}\n\n\n
''')

        
        rows = []
        for domain, count in domain_big_video_counts.items():
            rows.append((domain, count))
        rows_sorted = sorted(rows, key=lambda x: (x[0], -x[1]))  # (big asc, count desc)
        f.write("Correct answers in each domain Avg8:\n")
        f.write("Domain,\t\tCount\n")
        for concept, count in rows_sorted:
            f.write(f"{concept}\t\t{count*100/(8*test_total_domain_counts[concept])}\n")

        f.write("\n\n\n\n")

        rows = []
        for domain, count in domain_big_question_counts.items():
            rows.append((domain, count))
        rows_sorted = sorted(rows, key=lambda x: (x[0], -x[1]))  # (big asc, count desc)
        f.write("Correct answers in each domain pass8:\n")
        f.write("Domain,\t\tCount\n")
        for domain, count in rows_sorted:
            f.write(f"{domain}\t\t{count*100/test_total_domain_counts[domain]}\n")

        f.write("\n\n\n\n")
        rows = []
        for concept, count in concept_question_counts.items():
            rows.append((concept, count))
        rows_sorted = sorted(rows, key=lambda x: (x[0], -x[1]))  # (big asc, count desc)
        f.write("Correct answers in each topic:\n")
        f.write("Topic,\t\tCount\n")
        for topic, count in rows_sorted:
            f.write(f"{topic},\t\t{count}\n")

        # f.write('\n\n\n\n')
        # f.write(f"{test_model}: domain_big_video_counts: {domain_big_video_counts}\n")
        # f.write(f"{test_model}: domain_big_question_counts: {domain_big_question_counts}\n")
        # f.write(f"{test_model}: test_total_domain_counts: {test_total_domain_counts}\n")

        # for level, count in domain_diff_level_counts.items():
        #     f.write(f"{test_model}: video_level_counts:\n")
        #     f.write(f"{level}: {count/diff_level_val_video_counts[level]}\n") 
        # f.write('\n\n\n\n')

        # for level, count in domain_diff_level_question_counts.items():
        #     f.write(f"{test_model}: question_level_counts:\n")
        #     f.write(f"{level}: {count/diff_level_val_question_counts[level]}\n") 
        # f.write('\n\n\n\n')
        # f.write("\n") 

    
    
# count_arr =  ['gemini-2.5-pro','claude-3.7-sonnet', 'o3', 'o4-mini', 'gpt-oss-20b', 'gpt-oss-120b', 'Qwen3-Coder-30B-A3B-Instruct', 'qwen32_baseline',  'Qwen3-235B-A22B', 'DeepSeek-R1-0528', 'DeepSeek-V3.1', '7b_baseline' ]
# Qwen3-32B：简单prompt,带sys
# Qwen3-32B_model: 复杂prompt，带sys
count_arr = ['DeepSeek-R1-0528', 'DeepSeek-R1-Distill-Qwen-7B', 'DeepSeek-R1-Distill-Qwen-7B_nosys', 'DeepSeek-V3.1', 'Qwen3-235B-A22B', 'Qwen3-32B', 'Qwen3-32B_model', 'Qwen3-32B_nosys', 'deepseek-r1-distill-qwen-7b_sft-multiple', 'deepseek-r1-distill-qwen-7b_sft-multiple_nosys', 'deepseek-r1-distill-qwen-7b_sft-single', 'deepseek-r1-distill-qwen-7b_sft-single_nosys', 'gpt-oss-120b', 'gpt-oss-20b', 'qwen3-32b_sft-single', 'qwen3-32b_sft-single_nosys', 'qwen3-32b_sft1-multiple', 'qwen3-32b_sft1-multiple_nosys']
count_arr = ['deepseek-r1-distill-qwen-7b_sft-multiple_openthought-10000_nosys', 'deepseek-r1-distill-qwen-7b_sft-single_openthought-10000_nosys', 'gemini-2.5-pro','gpt-5-medium']
count_arr = ['deepseek-r1-distill-qwen-7b_sft-multiple-rl_ladder_20260117122955_global_step_100', 'deepseek-r1-distill-qwen-7b_sft-multiple-rl_ladder_20260117122955_global_step_80']
count_arr = ['vlm_as_judge_reward_llm_base_7b_deepseek_sft_multiple_rl']

count_arr = ['DeepSeek-R1-Distill-Qwen-7B_nosys', 'deepseek-r1-distill-qwen-7b_sft-multiple_nosys', 'deepseek-r1-distill-qwen-7b_sft-multiple-rl_ladder_20260117122955_global_step_100', 'Qwen3-32B_model', 'qwen3-32b_sft1-multiple_nosys', 'vlm_as_judge_reward_ladder_base_32b_qwen3_multiple_rl']
#   o3
#   o4-mini-high]

cli = argparse.ArgumentParser()
cli.add_argument('--model', nargs='+', help='model names to score (default: the count_arr list above)')
count_arr = cli.parse_args().model or count_arr
for txt in count_arr:
    test_filter(txt)