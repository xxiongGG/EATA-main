import openai
import pandas as pd
from tqdm import tqdm

'''
    Use ChatGPT4 labelling exercises.
'''
def get_HOT_chatGPT(Q_content):
    openai.api_type = "azure"
    openai.api_base = "https://research-hxl-gpt4.openai.azure.com/"
    openai.api_version = "2023-05-15"
    openai.api_key = "YOUR KEY"
    response = openai.ChatCompletion.create(
        engine="gpt4-32k",  # engine = "deployment_name".
        messages=[
            {"role": "system", "content": "You are an expert in the field of education in the subject of C language."},
            {"role": "user",
             "content": "The level of assessment of a topic can be categorized into high and low cognitive levels based on the text of the C topic, in conjunction with Bloom's taxonomy of cognitive objectives, for example:"
                        "Remembering: requires students to be able to memorize and understand basic grammar rules and keywords;"
                        "Understanding: requires students to be able to explain the function and logic of code;"
                        "Applying: requires students to be able to apply what they have learned to real-world situations;"
                        "Analyzing: requires students to be able to analyze and understand the structure and execution of code;"
                        "Evaluating: requires students to be able to assess the effectiveness of code and the possibilities for optimization;"
                        "Creating: requires students to be able to creatively solve problems and design new code."},
            {"role": "user",
             "content": Q_content + "The above topic C language topic is how to categorize according to Bloom's Cognitive Objective Taxonomy. Return the result in the following form"
                                    "[Classification]:[Reason]. [Reason] is controlled to 30 characters or less."}
        ]
    )

    # print(response)
    response_str = response['choices'][0]['message']['content']
    print(response_str)
    return response_str


def get_Q_content(Q_map):
    Q_content_list = Q_map['q_content'].to_list()
    print('【Q_content】 number of Q is :', len(Q_content_list))
    return Q_content_list


def get_Q_HOTs(Q_map):
    HOTs = []
    Q_content_list = get_Q_content(Q_map)
    for Q_content in tqdm(Q_content_list):
        HOT = get_HOT_chatGPT(Q_content)
        print(HOT)
        HOTs.append(HOT)
    Q_map['HOT'] = HOTs
    return Q_map


path = "C:/Users/xx/OneDrive/Files/4-code/Datasets/20230908/q_list.xlsx"
Q_map = pd.read_excel(path)
Q_map_1 = Q_map.iloc[:len(Q_map) // 2, :]
Q_map_2 = Q_map.iloc[len(Q_map) // 2:, :]
# print(Q_map_1)
# print(Q_map_2)

Q_map_HOTs_1 = get_Q_HOTs(Q_map_1)
Q_map_HOTs_1.to_excel("C:/Users/xx/OneDrive/Files/4-code/Datasets/20230908/q_list_hot(1).xlsx", index=False)
# print(Q_map_HOTs_1)
Q_map_HOTs_2 = get_Q_HOTs(Q_map_2)
Q_map_HOTs_2.to_excel("C:/Users/xx/OneDrive/Files/4-code/Datasets/20230908/q_list_hot(2).xlsx", index=False)
# print(Q_map_HOTs_2)
