## paths
.ONESHELL:
.PHONY: help
.DEFAULT_GOAL := help

SHELL = /bin/bash
CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate


## print a help msg to display the comments
help:
	@grep -hE '^[A-Za-z0-9_ \-]*?:.*##.*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

USER := $(shell whoami)
PWD := $(shell pwd)
ROOT := $(shell cd ..;  pwd)

## build a clean environment
setup_env:
	conda create --name causalvqa_eval python=3.12

## make certain to activate your conda environment -- makefiles handle this poorly
## for using Qwen2.5VL and several other elements.
setup_vllm:
	pip install vllm==0.7.3
	pip install fire submitit pandas qwen-vl-utils 

## our testing harness
setup_lmms_eval:
	cd ~/CausalVQA/lmms-eval/
	pip install -e .
	cd ~

## to produce reuslts with PerceptionLM
setup_plm:
	cd ~
	git clone https://github.com/facebookresearch/perception_models.git
	cd perception_models
	pip install -e .
	cd ~

setup_cleanup:
	pip install datasets==2.16.1
	pip install numpy==1.26.4
	pip install sentencepiece==0.1.99

## Note -- test dataset cannot be built as targets are held back
prep_test_data:
	python scripts/prep_mllm_eval.py --vid "test" --annotation_file "./data/test/test_metadata.csv" \
	--outp_loc ./experiments/Causal_VQA_test_set

prep_debug_data:
	python scripts/prep_mllm_eval.py --vid "debug" --annotation_file "./data/debug/debug_metadata.csv" \
	--outp_loc ./experiments/Causal_VQA_debug_set

 # setup files for lmms-eval to run
prep_evals:
	mkdir ~/CausalVQA/lmms-eval/lmms_eval/tasks/causalvqa
	cp -Rf tasks/ ~/CausalVQA/lmms-eval/lmms_eval/tasks/causalvqa/
	cp -Rf models/*.py ~/CausalVQA/lmms-eval/lmms_eval/models/
	
run_internvl2_5:
	cd ~/CausalVQA/lmms-eval/
	python -m accelerate.commands.launch \
    	--num_processes=8 \
    	-m lmms_eval \
		--model internvideo2_5 \
		--model_args pretrained=OpenGVLab/InternVL2_5-8B,max_frames_num=16,modality=video \
		--tasks debug_causalvqa \
		--batch_size 1 \
		--log_samples \
		--log_samples_suffix internvl2_5 \
		--output_path ./logs

run_llava_onevision:
	cd ~/CausalVQA/lmms-eval/
	python -m accelerate.commands.launch \
    	--num_processes=8 \
    	-m lmms_eval \
		--model llava_onevision \
		--model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,max_frames_num=16,device_map=auto \
		--tasks debug_causalvqa \
		--batch_size 1 \
		--log_samples \
		--log_samples_suffix llava_onevision \
		--output_path ./logs

run_qwen2_5vl_vllm:
	cd ~/CausalVQA/lmms-eval/
	python -m accelerate.commands.launch \
    	--num_processes=8 \
    	-m lmms_eval \
		--model vllm \
		--model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_num_frames=16,force_sample=True,tensor_parallel_size=1 \
		--tasks debug_causalvqa \
		--batch_size 1 \
		--log_samples \
		--log_samples_suffix vllm_qwen2_5vl \
		--output_path ./logs

run_plm:
	cd ~/CausalVQA/lmms-eval/
	python -m accelerate.commands.launch \
    	--num_processes=8 \
    	-m lmms_eval \
		--model plm \
		--model_args pretrained=facebook/Perception-LM-8B \
		--tasks debug_causalvqa \
		--batch_size 1 \
		--log_samples \
		--log_samples_suffix plm \
		--output_path ./logs

# gemini on the openai api requires a gemini api key
GOOGLE_API_KEY:=""

run_gemini_oai:
	cd ~/CausalVQA/lmms-eval/
	API_TYPE=gemini GOOGLE_API_KEY=$(GOOGLE_API_KEY)
	python -m accelerate.commands.launch \
    	--num_processes=1 \
    	-m lmms_eval \
		--model gpt4v \
		--model_args model_version=gemini-2.5-flash-preview-04-17,timeout=300,continual_mode=False \
		--tasks debug_causalvqa \
		--batch_size 1 \
		--log_samples \
		--log_samples_suffix gemini-2.5-flash-preview-04-17 \
		--output_path ./logs

# GPT-4o on azure requires an api endpoint and deployment, plus an api key
HOST:=""
DEPLOYMENT:=gpt-4o
API_KEY:=""

run_gpt4o:
	cd ~/CausalVQA/lmms-eval/
	API_TYPE=azure AZURE_ENDPOINT=https://$(HOST)/openai/deployments/$(DEPLOYMENT)/chat/completions?api-version=2024-10-21 AZURE_API_KEY=$(API_KEY)
	python -m accelerate.commands.launch \
    	--num_processes=1 \
    	-m lmms_eval \
		--model gpt4v \
		--model_args max_frames_num=10,timeout=360 \
		--tasks debug_causalvqa \
		--batch_size 1 \
		--log_samples \
		--log_samples_suffix gpt4o \
		--output_path ./logs
