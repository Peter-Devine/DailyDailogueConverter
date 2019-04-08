import argparse
import sys
import os
import pandas as pd
from io import open

# Takes input and output directories as arguments
parser=argparse.ArgumentParser()
parser.add_argument('--input', default=".", help='The file path of the unzipped DailyDialog dataset')
parser.add_argument('--output', default="./data", help='The file path of the output dataset')
args = parser.parse_args()
INPUT_PATH = args.input
OUTPUT_PATH = args.output

database_types = ["train", "validation", "test"]

# Read .txt file, and convert to .tsv file
for database_type in database_types:
    FOLDER_PATH = INPUT_PATH + "/" + database_type + "/" + database_type

    TEXT_FILE_PATH = FOLDER_PATH + "/dialogues_" + database_type + ".txt"
    EMOTION_FILE_PATH = FOLDER_PATH + "/dialogues_emotion_" + database_type + ".txt"
    ACT_FILE_PATH = FOLDER_PATH + "/dialogues_act_" + database_type + ".txt"

    dialogue_output = open(TEXT_FILE_PATH,"r", encoding="utf8").read().splitlines()
    emotion_output = open(EMOTION_FILE_PATH,"r", encoding="utf8").read().splitlines()
    act_output = open(ACT_FILE_PATH,"r", encoding="utf8").read().splitlines()

    pd.DataFrame({"dialogue": dialogue_output,
              "emotion": emotion_output,
              "act": act_output}).to_csv(FOLDER_PATH + "/"+database_type+".tsv", sep='\t')

# Create output directory if not already created
if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)

# Split each line of previously created .tsv into separate line for each turn of dialogue
for database_type in database_types:
    FILE_PATH = INPUT_PATH + "/" + database_type + "/" + database_type+"/"+database_type
    dialogue_data = pd.read_csv(FILE_PATH + ".tsv", sep='\t', index_col=0)

    def split_dialogue(whole_dialogue):
        split_dialogue_list = whole_dialogue.split("__eou__")
        split_dialogue_list = list(filter(None, split_dialogue_list))

        return split_dialogue_list

    def split_numbers(number_list):
        split_number_list = number_list.split(" ")
        split_number_list = list(filter(None, split_number_list))

        return split_number_list

    def strip_string(text):
        return text.strip()

    def list_length(input_list):
        return len(input_list)

    dialogue_list_series = dialogue_data['dialogue'].apply(split_dialogue)
    emotion_list_series = dialogue_data['emotion'].apply(split_numbers)
    act_list_series = dialogue_data['act'].apply(split_numbers)

    list_lengths = dialogue_list_series.apply(list_length)
    conversation_ids = [[i]*list_lengths[i] for i in range(len(list_lengths))]

    discretised_dialogue_series = [y for x in dialogue_list_series for y in x]
    discretised_emotion_series = [y for x in emotion_list_series for y in x]
    discretised_act_series = [y for x in act_list_series for y in x]
    discretised_conversation_id_series = [y for x in conversation_ids for y in x]

    discretised_dialogue_series = pd.Series(discretised_dialogue_series).apply(strip_string)

    discretised_emotion_dialogue_df = pd.DataFrame({"dialogue": discretised_dialogue_series,
              "emotion": discretised_emotion_series,
              "act": discretised_act_series,
              "convo_id": discretised_conversation_id_series})

    if database_type == "validation":
        discretised_emotion_dialogue_df.to_csv(OUTPUT_PATH+"/dev.tsv", sep='\t')
    else:
        discretised_emotion_dialogue_df.to_csv(OUTPUT_PATH+"/"+database_type+".tsv", sep='\t')
