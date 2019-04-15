import argparse
import sys
import os
import pandas as pd
from io import open

# Takes input and output directories as arguments
parser=argparse.ArgumentParser()
parser.add_argument('--input', default=".", help='The file path of the unzipped DailyDialog dataset')
parser.add_argument('--output', default="./data", help='The file path of the output dataset')
parser.add_argument('--separator', default=r"[TRN]", help='The separator token between context turns')
parser.add_argument('--turns', default="1", help='The number of previous turns to include in the context')
args = parser.parse_args()
INPUT_PATH = args.input
OUTPUT_PATH = args.output
SEPARATOR = args.separator
CONTEXT_LEVEL = int(args.turns)


database_types = ["train", "validation", "test"]

# Read .txt file, and convert to .tsv file
for database_type in database_types:
    FOLDER_PATH = INPUT_PATH + "/" + database_type

    TEXT_FILE_PATH = FOLDER_PATH + "/dialogues_" + database_type + ".txt"
    EMOTION_FILE_PATH = FOLDER_PATH + "/dialogues_emotion_" + database_type + ".txt"
    ACT_FILE_PATH = FOLDER_PATH + "/dialogues_act_" + database_type + ".txt"

    dialogue_output = open(TEXT_FILE_PATH,"r", encoding="utf8").read().splitlines()
    emotion_output = open(EMOTION_FILE_PATH,"r", encoding="utf8").read().splitlines()
    act_output = open(ACT_FILE_PATH,"r", encoding="utf8").read().splitlines()

    pd.DataFrame({"dialogue": dialogue_output,
              "emotion": emotion_output,
              "act": act_output}).to_csv(FOLDER_PATH + "/"+database_type+".tsv", sep='\t', encoding="utf-8")

# Create output directory if not already created
if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)

# Split each line of previously created .tsv into separate line for each turn of dialogue
for database_type in database_types:
    FILE_PATH = INPUT_PATH + "/" + database_type+"/"+database_type
    dialogue_data = pd.read_csv(FILE_PATH + ".tsv", sep='\t', index_col=0, encoding="utf-8")

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
    
    def context_dataframe_builder(context_level, dialogue, convo_id):
        context_df = pd.DataFrame()
    
        for context_level_counter in range(1, CONTEXT_LEVEL+1):

            # We shift the dialogue series by x in order to align it with the dialogue that is x steps ahead of it
            # This shifted list acts as our context series
            shifted_dialogue = dialogue[0:(len(dialogue)-context_level_counter)]
            current_turn = dialogue[context_level_counter:(len(dialogue))]

            # We make sure that the shifted conversations are from the same conversation - otherwise they are not context
            previous_convo_id = convo_id[0:(len(convo_id)-context_level_counter)]
            current_convo_id = convo_id[context_level_counter:(len(convo_id))]
            convo_match_mask = ~pd.Series(previous_convo_id).eq(pd.Series(current_convo_id))

            # Changing to DataFrame and then back to Series is done to use the mask method
            # Mask selects the previous dialogue if it matches the convo_id of the current dialogue, and
            # selects an empty string otherwise.
            context_dialogue = pd.DataFrame(shifted_dialogue, columns = ["Dialogue"]).mask(convo_match_mask, "")["Dialogue"]

            context_padding = pd.Series([""]*(context_level_counter))

            context_dialogue = context_padding.append(context_dialogue, ignore_index=True)

            context_df["context"+str(context_level_counter)] = context_dialogue
        
        return(context_df)
    
    def convert_context_dataframe_to_series(context_df):
        context_series = pd.Series([""] * context_df.shape[0])
        
        for context_column in context_df.columns[::-1]:
            # Here, we make sure the separator is here if the dialogue turn is not empty
            empty_string_selector = context_df[context_column] == ""
            separator_repeated = [SEPARATOR]*context_df.shape[0]
            separator_repeated_as_df = pd.DataFrame(separator_repeated, columns=["Dialogue"]).mask(empty_string_selector, "")
            separator_repeated_selected = separator_repeated_as_df.mask(empty_string_selector, "")["Dialogue"]
            
            context_series = context_series + separator_repeated_selected + context_df[context_column]
        
        # We take away the first separator for the context, as it preceeds every single context
        context_series = context_series.str.replace(pat=str(SEPARATOR), repl="", n=1, regex=False)
        
        return(context_series)

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
    
    context_df = context_dataframe_builder(CONTEXT_LEVEL, discretised_dialogue_series, discretised_conversation_id_series)
    context_series = convert_context_dataframe_to_series(context_df)
    
    discretised_emotion_dialogue_df = pd.DataFrame({"dialogue": discretised_dialogue_series,
              "emotion": discretised_emotion_series,
              "act": discretised_act_series,
              "convo_id": discretised_conversation_id_series,
              "context": context_series})

    if database_type == "validation":
        discretised_emotion_dialogue_df.to_csv(OUTPUT_PATH+"/dev.tsv", sep='\t', encoding="utf-8")
    else:
        discretised_emotion_dialogue_df.to_csv(OUTPUT_PATH+"/"+database_type+".tsv", sep='\t', encoding="utf-8")
