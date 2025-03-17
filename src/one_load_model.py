# NIH Clinical Center
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
# 2025-03-17
# Ricardo Bigolin Lanfredi
# Use `python <python_file> --help` to check inputs and outputs
# Description:
# This file runs the LLM labeler for all of the reports in a csv file or dataset

import sys
import os
import cli
import torch
import argparse
import importlib
import traceback
import math
import time

def add_model_args(parser):
    parser.add_argument(
        "--model-path",
        type=str,
        default="lmsys/vicuna-7b-v1.5",
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Hugging Face Hub model revision identifier",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps", "xpu", "npu"],
        default="cuda",
        help="The device type",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="A single GPU like 1 or multiple GPUs like 0,2",
    )
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="The maximum memory per GPU for storing model weights. Use a string like '13Gib'",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU and float32 on CPU.",
        default=None,
    )
    parser.add_argument(
        "--load-8bit", action="store_true", help="Use 8-bit quantization"
    )
    parser.add_argument(
        "--cpu-offloading",
        action="store_true",
        help="Only when using 8-bit quantization: Offload excess weights to the CPU that don't fit on the GPU",
    )
    parser.add_argument(
        "--gptq-ckpt",
        type=str,
        default=None,
        help="Used for GPTQ. The path to the local GPTQ checkpoint.",
    )
    parser.add_argument(
        "--gptq-wbits",
        type=int,
        default=16,
        choices=[2, 3, 4, 8, 16],
        help="Used for GPTQ. #bits to use for quantization",
    )
    parser.add_argument(
        "--gptq-groupsize",
        type=int,
        default=-1,
        help="Used for GPTQ. Groupsize to use for quantization; default uses full row.",
    )
    parser.add_argument(
        "--gptq-act-order",
        action="store_true",
        help="Used for GPTQ. Whether to apply the activation order GPTQ heuristic",
    )
    parser.add_argument(
        "--awq-ckpt",
        type=str,
        default=None,
        help="Used for AWQ. Load quantized model. The path to the local AWQ checkpoint.",
    )
    parser.add_argument(
        "--awq-wbits",
        type=int,
        default=16,
        choices=[4, 16],
        help="Used for AWQ. #bits to use for AWQ quantization",
    )
    parser.add_argument(
        "--awq-groupsize",
        type=int,
        default=-1,
        help="Used for AWQ. Groupsize to use for AWQ quantization; default uses full row.",
    )
    parser.add_argument(
        "--enable-exllama",
        action="store_true",
        help="Used for exllamabv2. Enable exllamaV2 inference framework.",
    )
    parser.add_argument(
        "--exllama-max-seq-len",
        type=int,
        default=4096,
        help="Used for exllamabv2. Max sequence length to use for exllamav2 framework; default 4096 sequence length.",
    )
    parser.add_argument(
        "--exllama-gpu-split",
        type=str,
        default=None,
        help="Used for exllamabv2. Comma-separated list of VRAM (in GB) to use per GPU. Example: 20,7,7",
    )
    parser.add_argument(
        "--exllama-cache-8bit",
        action="store_true",
        help="Used for exllamabv2. Use 8-bit cache to save VRAM.",
    )
    parser.add_argument(
        "--enable-xft",
        action="store_true",
        help="Used for xFasterTransformer Enable xFasterTransformer inference framework.",
    )
    parser.add_argument(
        "--xft-max-seq-len",
        type=int,
        default=4096,
        help="Used for xFasterTransformer. Max sequence length to use for xFasterTransformer framework; default 4096 sequence length.",
    )
    parser.add_argument(
        "--xft-dtype",
        type=str,
        choices=["fp16", "bf16", "int8", "bf16_fp16", "bf16_int8"],
        help="Override the default dtype. If not set, it will use bfloat16 for first token and float16 next tokens on CPU.",
        default=None,
    )

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument(
        "--conv-system-msg", type=str, default=None, help="Conversation system message."
    )
    
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    
    parser.add_argument("--no-history", action="store_true")

    parser.add_argument(
        "--judge-sent-end",
        action="store_true",
        help="Whether enable the correction logic that interrupts the output of sentences due to EOS.",
    )

    for action in parser._actions:
        if action.dest == 'model_path':
            action.default = "Qwen/Qwen2-72B-Instruct"
        if action.dest == 'num_gpus':
            action.default = 2
        if action.dest == 'device':
            action.default = "cuda"

    parser.add_argument("--temperature", type=float, default=0.0, help='''The temperature with which to run the model. A temperature of 0 will run the model determinitically and always have the same outputs. The higher the temperature, the more "creative" the model gets. The temperature that the model was validated with was 0. Default: 0''')
    parser.add_argument("--start_index", type=int, default = 0, help="The index of the first report to run the script for. Useful for dividing runs for full datasets into several clusters. Default: 0")
    parser.add_argument("--end_index", type=int, default = None, help='''The index of the last report to run the script for plus one. For example, use "--start_index=20000 --end_index=40000" to run the script for the 20000 reports from report 20000 to report 39999. Default: Run until the last of the reports in the list/dataset.''')
    parser.add_argument("--download_location", type=str, default = "./models_caches/", help='''The folder where to download the model to, or load the model from it it has already been downloaded, in case a huggingface model name is provided in the --model argument.''')
    parser.add_argument("--result_root", type=str, default = './results/', help='''A folder where to save the results. The results will be saved to the "parsing_results_llm.csv" file. If any results were already present in that csv for file with the same ID (report filename), that report file will be skipped. The script can be run in parallel in several servers with no problems to writing the outputs, since access to the file is locked when writing each row. Default: ./results/''')
    parser.add_argument("--keep_model_loaded_in_memory", type=str2bool, default = "false", help='''Turning this flag to True will allow rerun of the script with modifications of the cli.py file without reloading the model from disk to memory (~6 minutes). This flag was mainly used for development. Default: False''')
    parser.add_argument("--reports_csv", type=str, default = './reports.csv', help='''The path to the csv file containing the dataset reports. The reports IDs should be in a columns named 'image1', and reports in 'anonymized_report'. This type of input is used if --single_file and --test_list are not used. Default: ./reports.csv''')
    parser.add_argument('--single_file', type=str, default = None, nargs='*', help='''Paths to files to run the script for. For example, use "--single_file ./report_1.txt ./report_2.txt" to run the script for the files ./report_1.txt and ./report_2.txt. Be sure to not set the --dataset flag when using the --single_file flag. Default: not set.''')
    parser.add_argument('--test_list', type=str, default = None, help='''The path to a txt file containing a list of txt file paths, where each txt file in the list is in a separate line in the file and stores a radiology report to run the script for. Be sure to not set the --single_file flag when using the --test_list flag. Default: not using a test_list file''')
    from global_ import organ_denominations
    parser.add_argument('--do_labels', type=int, default = [1]*len(organ_denominations), nargs=len(organ_denominations), help=f'''List of {len(organ_denominations)} 0s or 1s used to indicate which labels to run the script for. The indices of the labels are: {', '.join([f'{index_abn}-> {abn}' for index_abn, abn in enumerate(organ_denominations)])}. Default: run it for all labels''')
    parser.add_argument('--n_jobs', type=int, default = 8, help='''Number of parallel jobs to run in this script. No speedup was seen for running it with more than 2 jobs. Default: 8 jobs.''')
    parser.add_argument('--split_report', type=str, default = "fast_extensive", choices = ["no", "fast", "extensive","fast_extensive"], help='''If no, there is no sentence extraction and the full report is used in the prompts. If fast, only the first step of sentence filtration is performed, with a single prompt per organ asking which sentences are informative. If extensive, only performs the second step of sentence filtration, by going sentence-by-sentence in the report and asking if it is informative for weach organ. If fast_extensive, then both steps are performed, and the second step is only performed if the sentence was not deemed informative in the first step.''')
    parser.add_argument('--prompt_to_use', type=str, default = "leavs", choices = ["maplez", "leavs"], help='''Set it to leavs o use a multiple-choice question for finding uncertainty assessment. Set it to maplez to use the Yes/No tree-based questions prompt from the MAPLEZ paper for finding uncertainty assessment.''')
    parser.add_argument('--use_cot', type=str2bool, default = "true", help='''If true, use chain-of-thought reasoning prompts for multiple choice questions and Yes/No questions.''')
    parser.add_argument('--definition', type=str, choices = ["multiple", "individual"], default = "multiple", help='If multiple, ask a multiple-choice question of the types of finding. If individual')
    parser.add_argument('--use_vllm', type=str2bool, default = "true", help='''If true, use the vllm library for inference.''')
    parser.add_argument('--labels_abnormality', type=str, default = "organs", choices = ["organs", "maplez"], help='''Use "maplez" for the abnormalities from the MAPLEZ paper, and "organs" for looking at abnormal organs as the LEAVS paper''')

    args = parser.parse_args()
    args.model = args.model_path
    download_location = args.download_location
    args.cot_for_uncertain = False
    args.do_tasks = [1,0,0,0]
    args.top_k = None
    args.length_multiplier = 4
    args.include_multiple_conv = False
    args.include_finding_types_sentence = True
    args.go_through_negative_after_multiple = True
    args.include_finding_description = True
    args.include_report_in_final_prompt = False

    model_name = args.model
    num_gpus = args.num_gpus
    reloaded = False
    try:
        importlib.reload(cli)
        reloaded = True
    except Exception as e: 
            traceback.print_exception(*sys.exc_info())
    # Model
    os.environ["HF_HUB_CACHE"] = download_location
    os.environ["HF_HOME"] = download_location
    os.environ["TRANSFORMERS_CACHE"] = download_location
    os.environ["HF_DATASETS_CACHE"] = download_location
    if not args.use_vllm:
        

        from fastchat.modules.awq import AWQConfig
        from fastchat.modules.gptq import GptqConfig
        from fastchat.modules.exllama import ExllamaConfig
        from fastchat.modules.xfastertransformer import XftConfig
        
        from fastchat.utils import str_to_torch_dtype
        from fastchat.model.model_adapter import load_model

        if args.device == "cuda":
            kwargs = {"torch_dtype": torch.float16}
            if num_gpus == "auto":
                kwargs["device_map"] = "auto"
            else:
                num_gpus = int(num_gpus)
                if num_gpus != 1:
                    kwargs.update({
                        "device_map": "auto",
                        "max_memory": {i: f"{159.5/num_gpus}GiB" for i in range(num_gpus)},
                    })
        elif args.device == "cpu":
            kwargs = {}
        else:
            raise ValueError(f"Invalid device: {args.device}")
        
        gptq_config=GptqConfig(
                    ckpt=args.gptq_ckpt or args.model_path,
                    wbits=args.gptq_wbits,
                    groupsize=args.gptq_groupsize,
                    act_order=args.gptq_act_order,
                )
        
        awq_config=AWQConfig(
                    ckpt=args.awq_ckpt or args.model_path,
                    wbits=args.awq_wbits,
                    groupsize=args.awq_groupsize,
                )
        
        if args.enable_exllama:
            exllama_config = ExllamaConfig(
                max_seq_len=args.exllama_max_seq_len,
                gpu_split=args.exllama_gpu_split,
                cache_8bit=args.exllama_cache_8bit,
            )
        else:
            exllama_config = None
        if args.enable_xft:
            xft_config = XftConfig(
                max_seq_len=args.xft_max_seq_len,
                data_type=args.xft_dtype,
            )
            if args.device != "cpu":
                print("xFasterTransformer now is only support CPUs. Reset device to CPU")
                args.device = "cpu"
        else:
            xft_config = None

        model, tokenizer = load_model(
            args.model_path,
            device=args.device,
            num_gpus=num_gpus,
            max_gpu_memory=args.max_gpu_memory,
            dtype=str_to_torch_dtype(args.dtype),
            load_8bit=args.load_8bit,
            cpu_offloading=args.cpu_offloading,
            gptq_config=gptq_config,
            awq_config=awq_config,
            exllama_config=exllama_config,
            xft_config=xft_config,
            revision=args.revision,
            debug=False,
        )

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)

        if args.device == "cuda" and num_gpus == 1:
            model.cuda()
    else:
        from transformers import AutoTokenizer
        import importlib.util
        import sys
        module_name = 'vllm'

        if module_name in sys.modules:
            module = sys.modules[module_name]
        else:
            # Use importlib.util.find_spec to locate the module
            spec = importlib.util.find_spec(module_name)
            if spec is None:
                raise ImportError(f"Module {module_name} not found")

            # Load the module from the found specification
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.usage.usage_lib import UsageContext
        engine_args = AsyncEngineArgs(
                model=args.model_path,
                tensor_parallel_size=num_gpus, 
                gpu_memory_utilization=0.95, 
                max_model_len=4096*args.length_multiplier, 
                disable_log_requests=True, 
                disable_log_stats = True,
                dtype = 'float16',
            )
        model = AsyncLLMEngine.from_engine_args(engine_args = engine_args, usage_context = UsageContext.LLM_CLASS)
        if not model.is_running:
            #start loop that is running the vllm model on the background
            import asyncio
            import threading
            model.start_background_loop()
            loop =  model._background_loop_unshielded.get_loop()
            asyncio.set_event_loop(loop)
            import nest_asyncio
            nest_asyncio.apply(loop)
            def run_event_loop():
                loop.run_forever()
            loop_thread = threading.Thread(target=run_event_loop)
            loop_thread.start()
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    try:
        cli.main(args, tokenizer, model)

    except Exception as e: 
        traceback.print_exception(*sys.exc_info())

    if args.keep_model_loaded_in_memory:
        while True:
            if reloaded:
                try:
                    cli.main(args, tokenizer, model)

                except Exception as e: 
                    time.sleep(1)
                    traceback.print_exception(*sys.exc_info())
            reloaded = False
            print("Press enter to re-run the script, CTRL-C to exit")
            
            sys.stdin.readline()
            try:
                importlib.reload(cli)

                reloaded = True
            except Exception as e: 
                traceback.print_exception(*sys.exc_info())
    if args.use_vllm:
        #stop loop that is running the vllm model on the background
        def stop_loop_after(loop):
            loop.call_soon_threadsafe(loop.stop)
        stop_thread = threading.Thread(target=stop_loop_after, args=(loop,))
        stop_thread.start()
        stop_thread.join()
        loop_thread.join()
        loop.close()