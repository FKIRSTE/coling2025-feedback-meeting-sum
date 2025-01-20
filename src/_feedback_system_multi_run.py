import os
import json
import logging
import time
import random
import re
import pandas as pd

from datasets import load_dataset, Features, Value
from openai import AzureOpenAI


from error_type_loader import build_from_files
import dictionary_errors as errors

# Load the API key from the environment
# openai.api_key = os.getenv("OPENAI_API_KEY")

# ************************************************************

OUTPUT_PATH = "projects_2024/emnlp2024_LLM-Feedback/EMNLP2024_Feedback/Feedback 2024/experiment_results"
CONFIG_PATH = 'projects_2024/emnlp2024_LLM-Feedback/EMNLP2024_Feedback/Feedback 2024/config.json'

# ************************************************************

with open(CONFIG_PATH) as config_file:
    config = json.load(config_file)

API_KEY = config["api_key"]
API_VERSION = config["api_version"]
ENDPOINT = config["endpoint"]
MODEL_NAME = config["model"]

client = AzureOpenAI(
    api_key=API_KEY,
    api_version= API_VERSION,
    azure_endpoint=ENDPOINT,
)

# Set some constants
# model related
# MODEL_NAME = "gpt-4-turbo"
MAX_TOKENS_FEEDBACK = 4000
MAX_TURNS_FEEDBACK = 5
MAX_TOKENS_REFINED_SUMMARY = 200
MAX_TURNS_FEEDBACK_REFINER = 5

# dataset related
DATASET_NAME = "projects_2024/emnlp2024_LLM-Feedback/EMNLP2024_Feedback/Feedback 2024/feedback_system"
DATASET_PATH = "projects_2024/emnlp2024_LLM-Feedback/EMNLP2024_Feedback/Feedback 2024/dataset_builder/datasets/huggingface/" # projects_2024/emnlp2024_LLM-Feedback/EMNLP2024_Feedback/Feedback 2024/datasets/test"

# Define the prompt dictionary for the different feedback error types
FEEDBACK_PROMPTS = build_from_files()

FEEDBACK_VERBOSITY_PROMPTS = {
    "location": "Please provide feedback on the location of the error. Where in the text does the error occur? Is it at the beginning, middle, or end of the text?",
    "reasoning": "Please provide reasoning why you think this passage contains an error? What makes you think this is an error?",
    "correction": "Please provide feedback on how to correct the error. What changes would you suggest to improve the passage?",
    "CoT": "Let's think step by step and describe every step you consider which leads you to the result that an error occurs or not."
}

FEEDBACL_SCORING_PROMPTS = {
    "existence": "Please provide feedback on the existence of the error. Does this passage contain an error? Answer 'yes' or 'no'. Then provide feedback on the quality of the passage. On a scale of 1 to 5, how much is the summary quality impacted by this error? 1 being the not impacted much and 5 being highly impacted. Assign a score of 0 if the error is not observed.",
}

def init_model(config_path, client_type="openai"):
    with open(config_path) as config_file:
        config = json.load(config_file)

    API_KEY = config.get("api_key")
    API_VERSION = config.get("api_version")
    ENDPOINT = config.get("endpoint")
    MODEL_NAME = config.get("model")
    
    if client_type == "openai":
        client = AzureOpenAI(
            api_key=API_KEY,
            api_version= API_VERSION,
            azure_endpoint=ENDPOINT,
        )

    return client, MODEL_NAME

def model_call(message, max_tokens):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=message,
            max_tokens=max_tokens,
            n=1,
            stop=None,
            temperature=0.0,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )

        return response.choices[0].message.content.strip()
    except Exception as e:
        raise Exception(f"API call failed: {str(e)}") from e
   

def safe_model_call(message, max_tokens, max_attempts=6, base_delay=3.0):
    attempt = 0
    while attempt < max_attempts:
        try:
            response = model_call(message, max_tokens)
            return response
        except Exception as e:
            if "429" in str(e):  # Check if the exception is due to rate limiting
                # Exponential backoff with jitter
                sleep_time = (2 ** (attempt+1)) + \
                    (random.randint(0, 1000) / 1000)
                logging.warning(
                    "Rate limit hit, backing off for %s seconds.", sleep_time)
                time.sleep(sleep_time)
                attempt += 1
            else:
                logging.error("Error encountered: %s", str(e))
                break  # Break the loop on non-rate limiting errors
        finally:
            # Sleep to ensure compliance with API rate limits
            time.sleep(base_delay)
            
    


class FeedbackSystem:
    """
        A class to handle the feedback generation process.
    """
    def __init__(self, feedback_protocol, ensemble=True, scoring_type="existence", feedback_discussion=False):
        self.ensemble = ensemble
        self.feedback_protocol = feedback_protocol
        self.scoring_type = scoring_type
        self.feedback_discussion = feedback_discussion

    def example_builder(self, example):
        # Extract components of the example
        transcript = example.get('transcript', '')
        summary = example.get('summary', None)
        if summary is None:
            raise errors.MissingExampleSummaryError(
                f"Summary for example '{example}' is missing.")
        score = example.get('score', None)
        if score is None:
            raise errors.MissingExampleScoreError(
                f"Score for example '{example}' is missing.")
        explanation = example.get('explanation', None)
        if explanation is None:
            raise errors.MissingExampleExplanationError(
                f"Explanation for example '{example}' is missing.")

        # Construct the prompt
        prompt = (
            f"Transcript: >>{transcript}<<\n"
            f"Predicted Summary: >>{summary}<<\n"
            f"Assgined Score: {score}\n"
            f"Explanation: {explanation}\n"
            f"---\n"
        )

        return prompt, transcript

    def expert_prompt_builder(self):
        """
        Construct the expert prompt for the feedback generation.
        """
        expert_prompt = (
            "You are an experienced linguist and you will be given one summary for a meeting."
            "Your task is to rate the summary based on the existence of the below provided error type."
            "Please make sure you read and understand these instructions carefully."
            "Please keep this document open while reviewing, and refer to it as needed."
            "Please do not be to harsh and only point out errors if they are really an issue. Otherwise be more of a friendly reviewer."
            "Following is the error type(s) you should look for: \n"
        )
        return expert_prompt

    def fewshot_prompt_builder(self, error_type):
        """
        Construct the few-shot prompt for the feedback generation.
        """
        definition_prompt = FEEDBACK_PROMPTS.get(
            error_type, {}).get("definition", None)
        if definition_prompt is None:
            raise errors.MissingDefinitionError(
                f"Definition for error type '{error_type}' is missing.")

        low_example = FEEDBACK_PROMPTS.get(
            error_type, {}).get('example', {}).get('low', None)
        if low_example is None:
            raise errors.MissingExampleError(
                f"Example for error type '{error_type}' is missing.")
        low_example_prompt, _ = self.example_builder(low_example)

        high_example = FEEDBACK_PROMPTS.get(
            error_type, {}).get('example', {}).get('high', None)
        if low_example is None:
            raise errors.MissingExampleError(
                f"Example for error type '{error_type}' is missing.")
        high_example_prompt, example_transcript = self.example_builder(
            high_example)

        examples_prompt = (
            f"Below are two examples demonstrating the different impact levels of the previously described error type."
            f"Please learn from these examples the concept and how the rating works. \n"
            f"---\n"
            f"Example 1: \n"
            f"{low_example_prompt}\n"
            f"Example 2: \n"
            f"{high_example_prompt}\n"
        )

        return definition_prompt, examples_prompt, example_transcript

    def task_prompt_builder(self, scoring_type):
        verbosity_prompt = ""
        for verbosity_type in FEEDBACK_VERBOSITY_PROMPTS:
            verbosity_prompt += f"{FEEDBACK_VERBOSITY_PROMPTS[verbosity_type]}\n" if self.feedback_protocol[verbosity_type] else ""

        scoring_prompt = FEEDBACL_SCORING_PROMPTS.get(scoring_type)
        return verbosity_prompt, scoring_prompt

    def evaluation_steps_prompt_builder(self):
        prompt = (
            "Evaluation steps: \n"
            "1. Read the transcript, if available, carefully and identify main topic and key points. \n"
            "2. Read the predicted summary and compare if it contains instances of the described error type. Note every instance you observe that is part of the error type. Only consider the error type and no other mistakes else. \n"
            "3. Rate the summary based on the existence of the error type with yes when at least on instance of the error type is found or no if the summary does not exhibit the error type. (primary task). \n"
            "4. You may be given secondary tasks, such as thinking step by step, explaining your decision, or pointing out the locations of each individual instance of the error type. These secondary tasks are designed to help you become more certain about your decision. \n"
            "5. Provide your findings in the desired format, so that your final output is a report on the existance of the error type in the given summary. \n"
            "Tip: Consider the whole input, i.e., the transcript and the predicted summary, provided in the user's prompt to make a good decision that humans will agree on. \n"
            "Please do not be too harsh. See it as a if the summary is by a student you want to pass the exam. \n"
        )
        return prompt

    def system_prompt_builder(self, error_type, scoring_type):

        expert_prompt = self.expert_prompt_builder()
        definition_prompt, examples_prompt, example_transcript = self.fewshot_prompt_builder(
            error_type)
        verbosity_prompt, scoring_prompt = self.task_prompt_builder(
            scoring_type)

        evaluation_steps_prompt = self.evaluation_steps_prompt_builder()

        system_prompt = (
            f"{expert_prompt}\n\n"
            f"Definiton: \n"
            f"\"{definition_prompt}\"\n"
            f"{evaluation_steps_prompt}\n\n"
            f"{examples_prompt}\n\n"
            f"Your secondary task: {verbosity_prompt}\n\n"
            f"Your primary task: {scoring_prompt}"
        )

        return system_prompt, example_transcript

    def user_prompt_builder(self, predicted_summary, transcript, example_transcript=True):

        transcript_prompt = ""
        if example_transcript:
            transcript_prompt = (
                f"If required, you can use the original transcript for look up: \n"
                f"\"{transcript}\n"
            )

        resources_prompt = (
            f"You should now perform the error search on the following predicted summary: \n"
            f"\"{predicted_summary}\"\n"
            f"{transcript_prompt}"
        )

        formatting_prompt = (
            "Please follow the following structure for your output and fill in the blanks: \n"
            "## Error type: <error type> !!\n"
        )

        if FEEDBACK_PROTOCOL["location"]:
            formatting_prompt += "## Location: <location> !!\n"
        if FEEDBACK_PROTOCOL["reasoning"]:
            formatting_prompt += "## Explanation: <reasoning> !!\n"
        if FEEDBACK_PROTOCOL["correction"]:
            formatting_prompt += "## Correction: <correction> !!\n"
        if FEEDBACK_PROTOCOL["CoT"]:
            formatting_prompt += "## Chain-of-Thought: <CoT> !! \n"


        formatting_prompt += "## Existence result: <primary task: existence result> !!\n"
        formatting_prompt += "## Likert score result: <primary task: impact on quality result wiht (1) minor and (5) severe> !!\n"


        user_prompt = (
            f"{resources_prompt}\n\n"
            f"{formatting_prompt}\n\n"
        )

        return user_prompt

    def prompt_builder(self, error_type, scoring_type, predicted_summary, transcript):
        logging.info("SYSTE: >> building prompts")
        system_prompt, example_transcript = self.system_prompt_builder(
            error_type, scoring_type)
        user_prompt = self.user_prompt_builder(
            predicted_summary, transcript, example_transcript)

        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return prompt

    def parse_feedback(self, feedback):
        parts = feedback.split(" !!\n")
        feedback_dict = {}
        for part in parts:
            if ": " in part:
                key, value = part.split(": ", 1)
                feedback_dict[key.strip("# ").strip()] = value.strip()
        return feedback_dict

    def collect_feedback_to_df(self, dataset):
        # Flatten the dataset to include one row per error type per example
        rows = []
        for example in dataset:
            row = {}
            row['Input'] = example.get(
                'input', 'Transcript could not be retrieved!')
            row['Predicted'] = example.get(
                'predicted', 'Predicted summary could not be retrieved!')
            row['Gold'] = example.get(
                'gold', 'Gold summary could not be retrieved!')
            for error_type, feedback in example.items():
                if isinstance(feedback, dict):
                    for key, value in feedback.items():
                        if key == 'Error type':
                            continue
                        column_name = f"{error_type}-{key}"
                        row[column_name] = value
                elif error_type == 'full_feedback':
                    row[error_type] = feedback
                else:
                    logging.error("FEEDBACK MISTAKE! -- %s", key)
            rows.append(row)
        df = pd.DataFrame(rows)
        return df

    def generate_feedback_multi(self, predicted_summary, transcript, base_delay=3.0, max_attempts=6):
        feedback_entries = {}
        full_feedback = ""
        for error, prompt in FEEDBACK_PROMPTS.items():
            prompt = self.prompt_builder(
            error, self.scoring_type, predicted_summary, transcript)
            feedback = safe_model_call(prompt, MAX_TOKENS_FEEDBACK)
            structured_feedback = self.parse_feedback(feedback)
            feedback_entries[error] = structured_feedback
            full_feedback += f"\n\n {feedback}"

        feedback_entries['full_feedback'] = full_feedback
        return feedback_entries

    def generate_feedback_single(self, predicted_summary, transcript, max_attempts=6):
        expert_prompt = self.expert_prompt_builder()

        error_prompts_total = ""
        for error, prompt in FEEDBACK_PROMPTS.items():
            definition_prompt, examples_prompt, _ = self.fewshot_prompt_builder(
                error)
            error_prompt = (
                f"Definiton: \n"
                f"\"{definition_prompt}\"\n"
                f"{examples_prompt}\n\n"
                f"------------------------\n\n"
            )
            error_prompts_total += error_prompt

        verbosity_prompt, scoring_prompt = self.task_prompt_builder(
            self.scoring_type)

        system_prompt = (
            f"{expert_prompt}\n\n"
            f"{error_prompts_total}\n\n"
            f"Your secondary task: {verbosity_prompt}\n\n"
            f"Your primary task: {scoring_prompt} for all previously defined error types. Start each error part with '$$<ERROR TYPE>"
        )
        user_prompt = self.user_prompt_builder(
            predicted_summary, transcript, True)

        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        feedback_entries = {}
        full_feedback = ""
        full_feedback = safe_model_call(prompt, MAX_TOKENS_FEEDBACK)
        feedback_splitted = split_feedback_into_dict(full_feedback)
        
        for errory_type, feedback_string in feedback_splitted.items():
            structured_feedback = self.parse_feedback(feedback_string)
            feedback_entries[errory_type] = structured_feedback
        
        feedback_entries['full_feedback'] = full_feedback
        return feedback_entries

    def generate_feedback_dataset(self, dataset):
        logging.info("SYSTE: >> generate dataset")
        updated_dataset = []
        generation_function = self.generate_feedback_multi if self.ensemble else self.generate_feedback_single
        # updated_dataset = dataset.map(append_feedback)

        ex = [dataset[1]]

        for example in dataset:
            predicted_summary = example["Predicted"]
            transcript = example["Input"]
            feedback = generation_function(predicted_summary, transcript)
            feedback['input'] = transcript
            feedback['predicted'] = predicted_summary
            feedback['gold'] = example["Gold"]
            updated_dataset.append(feedback)

        return updated_dataset

    def consolidate_feedback(self, dataset):
        # Save the original feedback
        dataset["long_feedback"] = dataset["feedback"]
        for example in dataset:
            long_feedback = " ".join(example["feedback"].values())
            consolidation_prompt = "Please provide a consolidated feedback for the errors in the passage."
            consolidated_feedback = model_call(
                f"{consolidation_prompt}\n\n{long_feedback}", MAX_TOKENS_FEEDBACK)
            example["feedback"] = consolidated_feedback


class RefinementSystem:
    """
    A class to handle the refinement process.
    """
    def __init__(self, feedback_protocol, transition_protocol="feedback"):
        self.transition_protocol = transition_protocol
        self.feedback_protocol = feedback_protocol

    def pre_process_feedback(self, feedback, base_delay=3.0, max_attempts=6):

        sections_to_ignore = []
        if (not self.feedback_protocol.get("CoT", False)):
            sections_to_ignore.append("Chain-of-Thought")

        if (not self.feedback_protocol.get("correction", False)):
            sections_to_ignore.append("Correction")

        feedback_splits = split_feedback_into_dict(feedback, sections_to_ignore)


        if (self.transition_protocol == "feedback"):
            return feedback_splits

        considered_splits = {
            error_type: content
            for error_type, content in feedback_splits.items()
            if "N/A" not in content and "Overall result: No" not in content and "Overall result: 0" not in content
        }

        feedback_splits = considered_splits

        if (self.transition_protocol == "consolidate"):
            positive_feedback = assemble_feedback_from_dict(feedback_splits)
            prompt = [
                {"role": "system", "content": "You are a professional feedback summarizer, that provides a comprehensive, direct version of a feedback report. You condensed version should be usable for someone to improve their previous summary effectively. So you are allowed to structure it in the most effective way to address the feedback. The refinement should be successfull purely from your feedback and the previous summary so include all relevant details given in the report."},
                {"role": "user", "content": f"Please consolidate the following feedback into a plan and provide a usable feedback: {positive_feedback}. Use the output format 'Add: <Add the information of ...> \n Remove: <Remove the information of ...> \n Rephrase: <Rephrase the information of ...> \n Simplify: <Shorten the summary regarding ...> \n Keep: <Keep the summary unchanged at ...>'. Include all details from the feedback. Make your answer super precise like you are micro-managing a really limited user and need to provide every little detail and plan how to reslove the issues. As the refiner does not know about the original meeting transcript, you need to show clear examples for each error, why it is a mistake."},
            ]
            
            consolidated_feedback = safe_model_call(prompt, MAX_TOKENS_FEEDBACK)
            
            return consolidated_feedback

    def refine_summary(self, predicted_summary, feedback, base_delay=3.0, max_attempts=6):
        system_message = (
            "You are an expert in refining and improving summaries."
            "Your task is to  of conversations based on a given feedback report."
            "All the content to improve the original summary and make it the very best is provided in the review, as the reviewer provides all details."
        )

        user_message = (
            f"Please improve this summary: \n"
            f"\'{predicted_summary}\' \n"
            f"considering this review: \n"
            f"\'{feedback}\' \n"
        )

        prompt = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        refined_summary = safe_model_call(prompt, MAX_TOKENS_REFINED_SUMMARY)
        return refined_summary

    def refine_summary_full(self, predicted_summary, feedback_dict, transcript, assemble=False):
        if assemble:
            feedback = assemble_feedback_from_dict(feedback_dict)
        else:
            feedback = feedback_dict
        
        if self.feedback_protocol.get("transcript", False):
            feedback = (
                f"{feedback} \n\n"
                f"You may also consider the original meeting transcript for lookup and refinement. \n"
                f"{transcript}"
            )

        refined = self.refine_summary(predicted_summary, feedback)
        return refined

    def refine_summary_iterative(self, predicted_summary, feedback_dict):
        summary = predicted_summary
        for _, feedback in feedback_dict.items():
            refined = self.refine_summary(summary, feedback)
            summary = refined

        return summary

    # Define the function to refine the summaries for the entire dataset
    # This function will modify the dataset in place
    # It will add a new key "refined_summary" to each example

    def refine_summaries_dataset(self, dataset):
        # Assuming 'refine_summary' is a method that takes predicted summary and feedback, returning a refined summary.
        # Create a new column 'refined_summary' by applying a function row-wise

        refine_function = self.refine_summary_full
        name = f"refined_summary-{self.transition_protocol}"

        dataset[name] = dataset.apply(lambda row: refine_function(
            row['Predicted'], self.pre_process_feedback(row['full_feedback']), row['Input']), axis=1)


def find_csv_filenames(path_to_dir, suffix=".csv"):
    filenames = os.listdir(path_to_dir)
    # filenames = ['feedback_led.csv']
    print(filenames)
    return [os.path.join(path_to_dir, filename) for filename in filenames if filename.endswith(suffix)]

# Define the function to load the dataset from a filedd


def get_dataset(file_path, features, format="csv", delimiter=","):
    files = find_csv_filenames(file_path)
    dataset = load_dataset(format=format, data_files=files,
                           delimiter=delimiter, features=features)

    # Define a function to check for missing values in any column
    def filter_missing_values(example):
        # Check for None or other forms you consider 'missing', like empty strings
        return all(value is not None and value != '' for value in example.values())

    # Apply the filter function to remove rows with missing values
    filtered_dataset = dataset['train'].filter(filter_missing_values)
    dataset['train'] = filtered_dataset

    return dataset

def get_dataframe (file_path, delimiter=","):
    csv_datasets = find_csv_filenames(file_path)
    dfs = [pd.read_csv(file, sep=delimiter) for file in csv_datasets]
    dfs_combined = pd.concat(dfs, ignore_index=True)
    return dfs_combined

def remove_section(text, section_name):
    pattern = re.compile(rf'## {section_name}:(.*?)(!!\n|$)', re.DOTALL)
    return re.sub(pattern, '', text) 

def split_feedback_into_dict(feedback, remove_content=None):
    """
    Split the feedback into blocks starting with "## Error type:
    """
    
    if remove_content is None:
        remove_content = []
    blocks = re.split(r'(?=## Error type:)', feedback)

    # Initialize a dictionary to store feedback
    feedback_dict = {}

    # Iterate over each block
    for block in blocks:
        if block.strip():  # Ensure the block is not empty
            # Find the first occurrence of "## Error type:"
            start_index = block.find('## Error type:')
            # Extract the error type
            end_index = block.find('!!', start_index)
            error_type_line = block[start_index:end_index]
            error_type = error_type_line.replace('## Error type:', '').strip()

            # Everything after the error type to the end of the block is the content
            content = block[end_index:].strip()

            for section_name in remove_content:
                content = remove_section(content, section_name)

            # Ensure content ends with the delimiter for compatibility
            if not content.endswith('!!'):
                content += ' !!'

            # Store content under the corresponding error type
            feedback_dict[error_type] = content

    return feedback_dict


def assemble_feedback_from_dict(feedback_dict):
    feedback_text = ""
    for error_type, content in feedback_dict.items():
        # Add the error type header
        feedback_text += f"## Error type: {error_type} !!\n"
        # Add the content and ensure it ends with '!!'
        if not content.strip().endswith('!!'):
            content = content.strip() + ' !!'
        # Adding a newline for separation between entries
        feedback_text += content + "\n\n"
    return feedback_text


def controller_generate_feedback(dataset, feedback_protocol, scoring_type, ensemble_feedback, feedback_discussion):
    """_summary_

    Args:
        dataset (_type_): _description_
    """
    # Initialize the FeedbackSystem and RefinementSystem
    feedback_system = FeedbackSystem(feedback_protocol=feedback_protocol,
                                     ensemble=ensemble_feedback,
                                     scoring_type=scoring_type,
                                     feedback_discussion=feedback_discussion
                                     )
        
    # Generate feedback for the entire dataset
    feedback_complete = feedback_system.generate_feedback_dataset(
        dataset['train'])

    feedback_df = feedback_system.collect_feedback_to_df(feedback_complete)
    # logging.info(feedback_df.head())
    feedback_df.to_csv(f"{OUTPUT_PATH}/{OUTPUT_FILE_NAME}", index=False)

def controller(in_document='',
               out_document='',
               generate_feedback=True,
               generate_refinement=True,
               feedback_protocol={"location": False, "reasoning": True, "correction": False, "CoT": False},
               scoring_type='existence',
               ensemble_feedback=False,
               feedback_discussion=False,
               transition_protocol='feedback',
               feedback_path=f"{OUTPUT_PATH}/"):
    """
    Controller function for the feedback system multi-run.

    Args:
        in_document (str): Path to the input document.
        out_document (str): Path to save the output document.
        generate_feedback (bool): Flag indicating whether to generate feedback.
        generate_refinement (bool): Flag indicating whether to generate refinement.
        feedback_protocol (dict): Dictionary specifying the feedback protocol.
        scoring_type (str): Type of scoring to be used.
        ensemble_feedback (bool): Flag indicating whether to use ensemble feedback.
        feedback_discussion (bool): Flag indicating whether to enable feedback discussion.
        transition_protocol (str): Protocol for transitioning between feedback rounds.
        feedback_path (str): Path to the feedback file.

    Returns:
        None
    """

    if generate_feedback:
        # Load the dataset
        features = Features({
            'Input': Value('string'),
            'Predicted': Value('string'),
            'Gold': Value('string'),
        })
        dataset = get_dataset(DATASET_PATH, in_document, features)
        controller_generate_feedback(dataset, feedback_protocol, scoring_type, ensemble_feedback, feedback_discussion)

    # Refine the summaries for the entire dataset
    if generate_refinement:
        feedback_df = get_dataframe(feedback_path)

        refinement_system = RefinementSystem(feedback_protocol=feedback_protocol, transition_protocol=transition_protocol)
        refinement_system.refine_summaries_dataset(feedback_df)
        logging.info(feedback_df.head())
        feedback_df.to_csv(f"{OUTPUT_PATH}/{out_document}", index=False)



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    client, MODEL_NAME = init_model(CONFIG_PATH)
        
    working_path = ''
    starting_document = ''

    for looper in range(0, 1):
        working_document = f'Run-{looper}_1.csv'

        logging.info("******************************************** FEEDBACK ********************************************")
        controller(in_document= starting_document, out_document=working_document, generate_feedback=True, generate_refinement=False, feedback_protocol={"reasoning": True, "correction": False, "CoT": True}, scoring_type='existence', ensemble_feedback=True, feedback_path=working_path)
        starting_document = working_document
        working_document = f'Run-{looper}_2.csv'

        logging.info("******************************************** REFINEMENT ********************************************")
        controller(in_document= starting_document, out_document=working_document, generate_feedback=False, generate_refinement=True, feedback_protocol={'transcript': False, 'CoT': True, 'correction': False}, scoring_type="", ensemble_feedback=True, transition_protocol='feedback', feedback_path=working_path)
        starting_document = working_document