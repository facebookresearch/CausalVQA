import json
import os
import re
import string
from pathlib import Path
from typing import Union

import numpy as np
import yaml
from loguru import logger as eval_logger


def causalvqa_doc_to_visual(doc):
    video_path = doc["video_path"]
    if not os.path.exists(video_path):
        raise Exception(f"video path:{video_path} does not exist, please check")
    return [video_path]


def causalvqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    option_prompt = ""
    option_list = doc["choices"]
    for option_letter, option in option_list:
        option_prompt += f"({option_letter}) {option}\n"
    full_text = "Question:" + doc["question"] + "\nOption:\n" + option_prompt + lmms_eval_specific_kwargs["post_prompt"]
    return full_text


# MCQ Accuracy implementation from MVBench


def mcq_acc(answer, pred):
    periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
    commaStrip = re.compile("(\d)(\,)(\d)")
    punct = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!"]

    def processPunctuation(inText):
        outText = inText
        for p in punct:
            if (p + " " in inText or " " + p in inText) or (re.search(commaStrip, inText) != None):
                outText = outText.replace(p, "")
            else:
                outText = outText.replace(p, " ")
        outText = periodStrip.sub("", outText, re.UNICODE)
        return outText

    def process(answer):
        option_regex = re.compile(r"(?<=Your Answer Letter:)[\s\S]*?([A-E])[\s\S]*?(?=END)", re.IGNORECASE)
        match = option_regex.match(answer.strip())

        if match:
            # If matched, return the option letter in uppercase
            letter = str(match.group(1))
            letter = letter.strip()
            return letter.upper()
        else:
            # If no match, process the answer as before
            answer = answer.replace("\n", " ")
            answer = answer.replace("\t", " ")
            answer = answer.strip()
            answer = processPunctuation(answer)
            answer = answer.strip("'")
            answer = answer.strip('"')
            answer = answer.strip(")")
            answer = answer.strip("(")
            answer = answer.strip().lower()

            # Try to find any single letter (A-E) in the processed answer
            letter_match = re.findall(r"\b([A-E])\b", answer, re.IGNORECASE)
            if letter_match:
                return letter_match[-1].upper()

            return answer

    pred = process(pred)
    answer = process(answer)

    if pred == answer:
        score = 1
    else:
        score = 0

    return score


def causalvqa_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case mvbench_perception_score), value: metric value
    """
    pred = results[0]

    # Get the ground truth option letter
    gt_option_letter = doc["answer"]

    qid = doc['id']

    # Calculate the score using mcq_acc function
    score = mcq_acc(gt_option_letter, pred)

    data_dict = {"qid": qid, "pred_answer": pred, "gt_answer": gt_option_letter, "score": score}

    return {"single_accuracy": data_dict, "paired_accuracy": data_dict}


def causalvqa_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    pair_correct_count = 0
    single_correct_count = 0

    result_by_qid = {}
    for answer_dict in results:
        qid = answer_dict["qid"]
        if qid not in result_by_qid.keys():
            result_by_qid[qid] = int(answer_dict['score'])
        else:
            result_by_qid[qid] = result_by_qid[qid] + int(answer_dict['score'])

    for qid, val in result_by_qid.items():

        if int(val) == 2:
            pair_correct_count += 1
            single_correct_count += 1
        elif int(val) == 1:
            single_correct_count += .5


    single_accuracy = single_correct_count / len(result_by_qid)
    paired_accuracy = pair_correct_count / len(result_by_qid)

    return single_accuracy * 100, paired_accuracy * 100

def single_accuracy(results,  task="", subset="causalvqa"):
    sa, _ = causalvqa_aggregate_results(results)
    return sa

def paired_accuracy(results,  task="", subset="causalvqa"):
    _, pa = causalvqa_aggregate_results(results)
    return pa

def easy_des_pair_accuracy(results, args):
    return paired_accuracy(
        results,  task="Descriptive_easy", subset="causalvqa"
    )

def easy_ant_pair_accuracy(results, args):
    return paired_accuracy(
        results,  task="Anticipation_easy", subset="causalvqa"
    )

def easy_cou_pair_accuracy(results, args):
    return paired_accuracy(
        results,  task="Counterfactual_easy", subset="causalvqa"
    )

def easy_pla_pair_accuracy(results, args):
    return paired_accuracy(
        results,  task="Planning_easy", subset="causalvqa"
    )

def easy_hyp_pair_accuracy(results, args):
    return paired_accuracy(
        results,  task="Hypothetical_easy", subset="causalvqa"
    )

def easy_des_sing_accuracy(results, args):
    return single_accuracy(
        results,  task="Descriptive_easy", subset="causalvqa"
    )

def easy_ant_sing_accuracy(results, args):
    return single_accuracy(
        results,  task="Anticipation_easy", subset="causalvqa"
    )

def easy_cou_sing_accuracy(results, args):
    return single_accuracy(
        results,  task="Counterfactual_easy", subset="causalvqa"
    )

def easy_pla_sing_accuracy(results, args):
    return single_accuracy(
        results,  task="Planning_easy", subset="causalvqa"
    )

def easy_hyp_sing_accuracy(results, args):
    return single_accuracy(
        results,  task="Hypothetical_easy", subset="causalvqa"
    )

def med_des_pair_accuracy(results, args):
    return paired_accuracy(
        results,  task="Descriptive_med", subset="causalvqa"
    )

def med_ant_pair_accuracy(results, args):
    return paired_accuracy(
        results,  task="Anticipation_med", subset="causalvqa"
    )

def med_cou_pair_accuracy(results, args):
    return paired_accuracy(
        results,  task="Counterfactual_med", subset="causalvqa"
    )

def med_pla_pair_accuracy(results, args):
    return paired_accuracy(
        results,  task="Planning_med", subset="causalvqa"
    )

def med_hyp_pair_accuracy(results, args):
    return paired_accuracy(
        results,  task="Hypothetical_med", subset="causalvqa"
    )

def med_des_sing_accuracy(results, args):
    return single_accuracy(
        results,  task="Descriptive_med", subset="causalvqa"
    )

def med_ant_sing_accuracy(results, args):
    return single_accuracy(
        results,  task="Anticipation_med", subset="causalvqa"
    )

def med_cou_sing_accuracy(results, args):
    return single_accuracy(
        results,  task="Counterfactual_med", subset="causalvqa"
    )

def med_pla_sing_accuracy(results, args):
    return single_accuracy(
        results,  task="Planning_med", subset="causalvqa"
    )

def med_hyp_sing_accuracy(results, args):
    return single_accuracy(
        results,  task="Hypothetical_med", subset="causalvqa"
    )

def hard_des_pair_accuracy(results, args):
    return paired_accuracy(
        results,  task="Descriptive_hard", subset="causalvqa"
    )

def hard_ant_pair_accuracy(results, args):
    return paired_accuracy(
        results,  task="Anticipation_hard", subset="causalvqa"
    )

def hard_cou_pair_accuracy(results, args):
    return paired_accuracy(
        results,  task="Counterfactual_hard", subset="causalvqa"
    )

def hard_pla_pair_accuracy(results, args):
    return paired_accuracy(
        results,  task="Planning_hard", subset="causalvqa"
    )

def hard_hyp_pair_accuracy(results, args):
    return paired_accuracy(
        results,  task="Hypothetical_hard", subset="causalvqa"
    )

def hard_des_sing_accuracy(results, args):
    return single_accuracy(
        results,  task="Descriptive_hard", subset="causalvqa"
    )

def hard_ant_sing_accuracy(results, args):
    return single_accuracy(
        results,  task="Anticipation_hard", subset="causalvqa"
    )

def hard_cou_sing_accuracy(results, args):
    return single_accuracy(
        results,  task="Counterfactual_hard", subset="causalvqa"
    )

def hard_pla_sing_accuracy(results, args):
    return single_accuracy(
        results,  task="Planning_hard", subset="causalvqa"
    )

def hard_hyp_sing_accuracy(results, args):
    return single_accuracy(
        results,  task="Hypothetical_hard", subset="causalvqa"
    )
