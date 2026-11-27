================================================================================
SimuScene
Training and Benchmarking Code Generation for Physics-Inspired Animations
Text -> Code -> Video evaluation for Physics-Inspired Animations
================================================================================
[Paper](https://arxiv.org/abs/2602.10840) | [Code](https://github.com/Yanan-Wang-cs/SimuScene) | [Dataset](https://huggingface.co/datasets/yananwang-MBZUAI/SimuScene)

SimuScene is a task, dataset, and benchmark that evaluates whether LLMs can
generate executable Python code whose rendered animation is visually aligned
with a natural-language physical-scenario description. The pipeline is:

    scenario description  ->  LLM-generated Python code  ->  rendered MP4 video
                          ->  VLM answers True/False verification questions
                          ->  a response passes only if ALL questions are True.

The dataset covers 52 physics concepts across 5 domains (Mechanics,
Electromagnetism, Optics, Fluid Mechanics, Thermodynamics), with a
334-scenario human-verified test set and quality-controlled training data.


--------------------------------------------------------------------------------
1. REPOSITORY LAYOUT
--------------------------------------------------------------------------------

  data_construction/      Dataset construction pipeline
    step1_scenarios_generation.py   GPT-4o: basic scenarios + conditions +
                                    difficulty-controlled dynamic scenarios
    step2_plausibility_and_vlm_question.py
                                    GPT-4o: plausibility filter + generate
                                    True/False visual verification questions
    step3_sufficiency.py            DeepSeek-R1-0528: check that a question set
                                    fully covers the described motion
    utils.py                        LLMClient / LLMProcessor for a self-hosted
                                    vLLM / SGLang cluster (OpenAI-compatible)
    generated/                      Intermediate per-topic CSV artifacts
                                    (basicquestions/, conditions/, questions/, ...)

  evaluation/             Benchmark evaluation pipeline
    step1_gencode.py                Build code-generation prompts for the test set
    step2_execute_code.py           Extract, sanitize, and execute generated code
                                    (memory + timeout limits) to render MP4 videos
    step3_vlmjudge.py               Qwen2.5-VL-72B-Instruct: answer verification
                                    questions over each rendered video
    step4_remove_empty.py           Housekeeping: delete empty output folders
    step5_getresult.py              Compute E.R. / R.R. / P.R. / Acc.
                                    (Avg@8 and Pass@8) + domain/topic breakdown
    code_utils.py                   sanitize_code(): AST-based guard that strips
                                    install/blocking calls before execution
    utils.py                        Same LLMClient as data_construction/utils.py
    result/                         Evaluation outputs

  data/                   Released dataset (JSONL)
    val.jsonl                       334 human-verified test scenarios
    RL.jsonl                        4,325 RL training scenarios
    SFT.jsonl                       SFT training data (with reasoning "think" traces)
    README.md                       License (CC-BY-4.0)

  docs/                   Project web page (index.html + example videos)

  SimuScene.pdf, SimuScene_Supplementary.pdf   Paper + supplementary


--------------------------------------------------------------------------------
2. ENVIRONMENT INSTALLATION
--------------------------------------------------------------------------------

Tested with Python 3.11.

--- Quick install (recommended): use the bundled script ---

  bash install.sh              # create a Python 3.11 venv in ./.venv (default)
  bash install.sh --conda      # create a conda env named "simuscene" instead
  bash install.sh --no-system  # skip the apt system packages (e.g. no sudo)
  bash install.sh --help       # show usage

install.sh installs the system packages (ffmpeg / libgl1 / libglib2.0-0),
creates the environment, and installs all Python dependencies below. When it
finishes it prints the activate command:

  venv:   source .venv/bin/activate
  conda:  conda activate simuscene

--- Manual install (equivalent to what install.sh does) ---

  # Option A: conda
  conda create -n simuscene python=3.11 -y
  conda activate simuscene

  # Option B: venv
  python3.11 -m venv .venv
  source .venv/bin/activate

Install Python packages:

  pip install --upgrade pip
  pip install \
      openai \
      pandas \
      numpy \
      tqdm \
      opencv-python \
      psutil \
      pillow \
      datasets \
      qwen-vl-utils

System package for code execution / video rendering:

  # OpenCV (cv2) needs a video backend + basic graphics libs on headless servers.
  # Debian/Ubuntu:
  sudo apt-get update && sudo apt-get install -y ffmpeg libgl1 libglib2.0-0

Notes:
  - Generated code is executed as untrusted subprocesses. code_utils.sanitize_code
    removes install / interactive-blocking calls, and step2_execute_code enforces a
    memory cap (~3.5 GB) and a 300 s timeout per script, and disables pip installs.
    Still, run evaluation inside an isolated / sandboxed environment.


--------------------------------------------------------------------------------
3. MODEL BACKENDS (required before running)
--------------------------------------------------------------------------------

The pipeline uses two kinds of inference backends:

(a) OpenAI API  -- used by data_construction step1 and step2 (model: gpt-4o).
    Set your key (the committed api_key="xx" is a placeholder):

        export OPENAI_API_KEY="sk-..."

    and replace the placeholder in:
        data_construction/step1_scenarios_generation.py
        data_construction/step2_plausibility_and_vlm_question.py

(b) Self-hosted vLLM / SGLang cluster -- used by:
        data_construction/step3_sufficiency.py   -> DeepSeek-R1-0528
        evaluation/step3_vlmjudge.py             -> Qwen2.5-VL-72B-Instruct

    utils.LLMClient talks to these servers over the OpenAI-compatible API and
    discovers endpoints from JSON registry files it expects under:

        <utils.py dir>/server_status/server_instances_complete_<backend>_<model>_*.json

    Each entry lists {"model", "ip", "port", "backend"}. Stand up your own
    vLLM/SGLang servers for the two models above and write a matching
    registry file, or adapt get_server_ips() in utils.py to your setup.

    The batch driver call_local_llm_process_file.py (referenced by
    step3_vlmjudge.py's printed command) is part of your local serving infra
    and is NOT included here; it turns a *_vlm_*.jsonl request file into a
    *_responses.jsonl file using the served model.

    Alternatively, evaluation/judge_vllm_offline.py does the same with vLLM
    offline inference (no server or registry file needed):

        python judge_vllm_offline.py --model-tag <MODEL> --tp 2

    It extracts frames with qwen_vl_utils.process_vision_info, so sampling
    follows each request (fps 5). Tested with vLLM 0.34 and qwen-vl-utils
    0.0.14; if the default video reader fails on your machine, set
    FORCE_QWENVL_VIDEO_READER=decord.


--------------------------------------------------------------------------------
4. DATA CONSTRUCTION PIPELINE
--------------------------------------------------------------------------------

Run from inside data_construction/.

  Step 1 - Generate scenarios (GPT-4o)
      python step1_scenarios_generation.py
    Produces, per topic:
      generated/basicquestions/<Topic>.csv   basic visualizable problems
      generated/conditions/<Topic>.csv        extra visualizable conditions
      generated/questions/questions_<Topic>.csv
                                               dynamic scenarios (Questions, Level),
                                               where Level = number of conditions
                                               (difficulty).

  Step 2 - Plausibility filter + verification questions (GPT-4o)
      python step2_plausibility_and_vlm_question.py
    Reads the dynamic scenarios and, for each plausible one, generates a set of
    True/False visual verification questions, writing:
      generated/filtered_questions_<i>.csv     columns: Questions, Level,
                                               VisionQuestions (JSON list)
    Implausible scenarios (model returns "no" / empty) are dropped.

  Step 3 - Question sufficiency review (DeepSeek-R1-0528)
      python step3_sufficiency.py main            # build run.jsonl (prompts)
      # -> run run.jsonl through your local DeepSeek server (batch driver)
      #    to obtain run_responses.jsonl
      python step3_sufficiency.py process_jsonl   # parse responses -> per-video JSON
    Keeps only scenarios whose questions sufficiently cover the described motion
    (enableAnnotator = "Yes"); insufficient sets are routed back for regeneration.

  Together, step2 + step3 reduce the raw generated scenarios down to the final
  validated set (~7,659 scenarios) used for the released splits in data/.

  NOTE: intermediate CSV/JSONL artifacts are large and are meant to be
  regenerated locally; some hard-coded input/output paths and filename
  conventions (per-topic vs per-index) may need to be adjusted to your own
  directory layout.


--------------------------------------------------------------------------------
5. EVALUATION PIPELINE
--------------------------------------------------------------------------------

Run from inside evaluation/. The test-set scenario JSONs are expected under
human_modify/val/video_<i>.json (regenerated from data/val.jsonl).

  Step 1 - Build code-generation prompts
      python step1_gencode.py
    Produces simuscene_val_getcode.jsonl (system + user prompt per test scenario).
    Run this file through the LLM under test to obtain 8 responses per scenario,
    saved as:
      ../data/simuphy_genvideo_trained_responses8_<MODEL>.jsonl
    (each line: {"video_index", "responses": [ ... 8 code strings ... ]})

  Step 2 - Execute code -> render videos
      python step2_execute_code.py --process_batch <MODEL>
    Extracts the Python code from each response, sanitizes it, and runs it with
    memory/timeout limits, saving:
      save_video/<MODEL>/video_<id>/video_<id>_<k>.mp4
      code/<MODEL>/...          (the executed scripts)
      log/code_running_<MODEL>.txt, log/code_error_<MODEL>.txt
    Optional flags (defaults reproduce the paper's protocol):
      --allow-unfenced  if a response has no ```python block, run the whole
                        response when it parses as Python (the prompt asks
                        for "only the complete Python code", and some models
                        answer with bare code)
      --pass-stem       run each script inside its video folder and pass only
                        the file stem as `name`, as the prompt describes; by
                        default the full output path is passed
      --retry-name-option
                        when a script rejects the positional `name` with an
                        argparse usage error (it declared `--name`), run it
                        again with `--name <name>`; the prompt does not say
                        how `name` is passed
      --keep-pip-lines  by default every line containing `pip` is replaced
                        by `pass`, which also hits lines such as
                        `def ensure(import_name, pip_name):` and leaves the
                        script invalid; with this flag the code is left as
                        is and installs are blocked with PIP_NO_INDEX=1
      --num-processes   worker count (default: 75% of CPU cores)

  Step 3 - VLM verification (Qwen2.5-VL-72B-Instruct)
      python step3_vlmjudge.py --process main --model <MODEL>
      # -> run data/simuphy_vlm_<MODEL>.jsonl through the local Qwen2.5-VL server
      #    (batch driver) to obtain data/simuphy_vlm_<MODEL>_responses.jsonl
      python step3_vlmjudge.py --process process_jsonl --model <MODEL>
    Writes per-video judgments to:
      result/valresult/<MODEL>/batch_0/video_<id>/...
    By default, a video OpenCV cannot open (e.g. a stub left by a crashed
    script) skips the whole scenario; --per-video-errors skips only that video.

  Step 4 - (optional) clean empty folders
      python step4_remove_empty.py

  Step 5 - Compute metrics
      python step5_getresult.py --model <MODEL> [<MODEL> ...]
      # without --model, scores the count_arr list at the bottom of the file
    Writes result/output/<MODEL>.txt with, for both Avg@8 and Pass@8:
      Executable Rate (E.R.), Render Rate (R.R.), Playable Rate (P.R.),
      Accuracy (Acc., passes all verification questions)
    plus per-domain and per-topic breakdowns. Metrics are normalized by
    334 (scenarios) and 334*8 (responses).


--------------------------------------------------------------------------------
6. DATASET FILE FORMAT (data/*.jsonl)
--------------------------------------------------------------------------------

  val.jsonl / RL.jsonl fields:
    data_index    scenario id
    question      the physical-scenario description
    Level         difficulty (number of added conditions)
    vlm_questions {enableAnnotator, summarize, vlm_questions:[{index,question,weight}]}
    prompt        scenario text used as the code-generation input
    type          fine-grained concept (e.g. Simple_Pendulum)
    domain        one of Mechanics / Electromagnetism / Optics /
                  Fluid_Mechanics / Thermodynamics

  SFT.jsonl: same fields plus
    think         the reasoning trace + code response used for supervised fine-tuning

  Counts: val = 334, RL = 4,325. Every test scenario has 4-12 verification
  questions (most commonly 6-10).


--------------------------------------------------------------------------------
7. LICENSE
--------------------------------------------------------------------------------

Dataset: CC-BY-4.0 (see data/README.md).
