<h1 align="center">
CausalVQA
</h1>
<h3 align="center">
<a href="https://ai.meta.com/research/publications/causalvqa-a-physically-grounded-causal-reasoning-benchmark-for-video-models">Paper</a> &nbsp; | &nbsp;
<a href="https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks">Blog</a>
</h3>

We introduce CausalVQA, a benchmark dataset for video question answering (VQA) composed of question-answer pairs that probe models’ understanding of causality in the physical world. Existing VQA benchmarks either tend to focus on surface perceptual understanding of real-world videos, or on narrow physical reasoning questions created using simulation environments. CausalVQA fills an important gap by presenting challenging questions that are grounded in real-world scenarios, while focusing on models’ ability to predict the likely outcomes of different actions and events through five question types – counterfactual, hypothetical, anticipation, planning and descriptive. 

We designed quality control mechanisms that prevent models from exploiting trivial shortcuts, requiring models to base their answers on deep visual understanding instead of linguistic cues. 

![CS25_AIBlog_WorldModels_CausalVQA_Benchmark_v3](https://github.com/user-attachments/assets/49bf514d-9d0a-4ed9-bcb5-449567cd19cd)

We find that current frontier multimodal models fall substantially below human performance on the benchmark, especially on anticipation and hypothetical questions. This highlights a challenge for current systems to leverage spatial-temporal reasoning, understanding of physical principles, and comprehension of possible alternatives to make accurate predictions in real-world settings. 

See our [leaderboard](https://huggingface.co/spaces/facebook/pwm_leaderboard)! CausalVQA is one of several benchmarks released to assess physical world models -- our peer benchmarks are [IntPhys2](https://github.com/fairinternal/IntPhys2) and [MVPBench](https://github.com/fairinternal/MoravecBench).

## Benchmark Composition

<img width="835" alt="Screenshot 2025-06-05 at 8 07 40 PM" src="https://github.com/user-attachments/assets/78ca6c1f-5827-48ad-becb-e38a7198cf89" />

## Questions Examples

Our questions fall into five categories: counterfactual, hypothetical, anticipation, planning and descriptive. Difficulty levels are empirically aligned with results from human trials with 273 non-expert annotators.

<img width="616" alt="Screenshot 2025-06-03 at 10 34 01 AM" src="https://github.com/user-attachments/assets/14321f0b-47de-484c-9831-9b12a9bed994" />


## Sign up for an Ego4D license.
Please see [https://ego4ddataset.com/egoexo-license/](url)
You will need to accept the terms of the license agreement and wait up to 48 hrs for approval.
You will receive an id and secret to use with the AWS S3 CLI tool via email.

## Get the AWS cli tool and set up a profile
Choose an installation from https://github.com/aws/aws-cli/tree/v2
Then 
    aws configure
And follow the prompts to insert the id and key. No need to enter a region or output format.

## Download the dataset
Get the dataset using the aws s3 cli:

    aws s3 cp s3://ego4d-consortium-sharing/egoexo-public/v2/causal_vqa/CausalVQA.zip \<your location\>\\CausalVQA.zip

## Clone the CausalVQA repo

## Copy the contents to the repo directory
    cd CausalVQA
    mkdir data
    cd ..
    unzip CausalVQA.zip -d CausalVQA_data
    mv CausalVQA_data/CausalVQA/test CausalVQA/data
    mv CausalVQA_data/CausalVQA/debug CausalVQA/data
    
The directory structure should look like this:

```text
CausalVQA/
├── lmms-eval/
├── models/
├── scripts/
├── tasks/
├── data/
    └── debug/
    └── test/
```

## Build the environment and dependencies, prep dataset
We have included a makefile to assist

    make setup_env
    conda activate causalvqa_eval
    make setup_vllm
    make setup_lmms_eval
    make setup_plm
    make setup_cleanup
    make prep_debug_data
    
Follow any instructions. Each of these may take a wile to build/install.
**Note:** Metrics will only be produced for the debug set. The video segments, questions, and answer options for the test set are provided, but the correct answers are withheld.

## Prep evaluation
This will write copies of the task to lmms_eval and overwrite/add some models.
Given lmms_eval itself may evolve, we also supply the model scripts to aid with running. 
**Critical:** Replace \<add absolute ref\> in the dataset_path with the correct location,
or the dataset will not load. Then, 

    make prep_evals

## Run evals
We supply the parameters we used in our evals in the makefile. gemini_oai and gpt4o will require
api keys and host locations to work. See the make file.
    
    make run_internvl2_5

    make run_llava_onevision
    
    make run_qwen2_5vl_vllm
    
    make run_plm
    
    make run_gemini_oai
    
    make run_gpt4o

## Each annotation file contains the following:
- qid - a question identifier that gets used for pairing
- type - the question type (anticipation, counterfactual, descriptive, planning, hypothetical)
- question - the text of the question
- choices1 - the multiple choices
- correct1 - the target for choices1 (removed from the test set)
- choices2 - a perturbed and reordered set of multiple choices
- correct2 - the target for choices2 (removed from the test set)
- difficulty - the difficulty level from human baselines
- renamed_video - the videofile name

## References
Code in /models are slightly repaired versions scripts found in lmms-eval, an open source tool to evaluate multimodal LMMs,
and are the intellectual property of their original creators. Our very minor modifications do not constitute ownership 
and are not intended as redistribution. They are only for the review of our benchmark.

## Licenses
The benchmark is available under the [EgoExo License](https://ego4d-data.org/pdfs/Ego-Exo4D-Model-License.pdf), available through the [Ego-Exo4D project](https://ego-exo4d-data.org/). 

## Citations
    @misc{causalvqa,
          title={CausalVQA: A Physically Grounded Causal Reasoning Benchmark for Video Models}, 
          author={Aaron Foss and Chloe Evans and Sasha Mitts and Koustuv Sinha and Ammar Rizvi and Justine T Kao},
          year={2025},
          eprint={},
          archivePrefix={arXiv},
          primaryClass={cs.CL}
    }
