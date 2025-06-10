# In this script, we match the video with the paths, depending on cluster we need to fix the paths
# We save the result in a local HF dataset, which will be used by lmms-evals
import fire 
import pandas as pd
from pathlib import Path
from datasets import Dataset, DatasetDict
import string 
import yaml

VIDEO_DIR = {
    "debug": "<insert absolute reference>/CausalVQA/data/debug/videos/", #key, value with file locations for debug videos
    "test": "<insert absolute reference>/CausalVQA/data/test/videos/", #key, value with file locations for test videos
}

def format_choices(row, choices_col="choices1", answer_col="correct1"):
    """
    Split the annotation file choices columns on delimiter "|"
    Remove the number "1. X", instead create a list of choices and rewrite answer in string
    returns: 
        - clean choices list
        - roman answer
        - None for held back targets
    """
    option_list = row[choices_col].split("|")
    option_letters = string.ascii_uppercase
    new_options = []
    num2char = {}
    for char_index, option in enumerate(option_list):
        option_letter = option_letters[char_index]
        op = option.lstrip().rstrip()
        option_num, option_text = op[0], op[3:]
        num2char[option_num] = option_letter
        new_options.append((option_letter, option_text))
    
    answer_num = row[answer_col] 
    if answer_num is not float('nan'): 
        gt_option_letter = num2char[f"{answer_num}"]
    else:
        gt_option_letter = None
    return new_options, gt_option_letter

 

def main(vid="debug", annotation_file="", question_col="question", choices_cols=["choices1","choices2"], answer_cols=["correct1","correct2"], outp_loc=""):
    assert vid in VIDEO_DIR, "Download the videos and add an entry in the dict above -- "
    video_prefix = VIDEO_DIR[vid]
    assert annotation_file.endswith(".csv"), "Expecting a csv annotation file"
    df = pd.read_csv(annotation_file)
    rows = []
    for n in range(len(choices_cols)):
        for i,row in df.iterrows():
            video_path = Path(video_prefix) / row['renamed_video']
            if video_path.exists():
                choices, answer = format_choices(row, choices_col=choices_cols[n], answer_col=answer_cols[n])
                data_row = {
                    'id': row['qid'],
                    'task': row['type'],
                    'subset': row['difficulty'],
                    'n' : n,
                    'video_path': str(video_path),
                    'question': row[question_col],
                    'choices': choices,
                    'answer': answer
                }
                rows.append(data_row)
            else: 
                print(row['renamed_video']," video skipped due to video paths not being found!")
        
    new_df = pd.DataFrame(rows)
    new_df['strata'] = new_df.apply(lambda x: str(x['task'])+'_'+str(x['subset']), axis=1)
    for elem in new_df['strata'].unique():
        df = new_df[new_df['strata']==elem]
        new_dt = DatasetDict({'valid': Dataset.from_pandas(df)})
        new_dt.save_to_disk(outp_loc+'/{}'.format(elem)) 
    print(i, new_df.shape[0], "Done")

if __name__ == '__main__':
    fire.Fire(main)
