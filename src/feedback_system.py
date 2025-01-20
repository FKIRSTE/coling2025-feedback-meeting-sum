import os
import logging
from typing import List, Dict, Any
import pandas as pd

from backbone_model import BackboneModel
from error_type_loader import build_from_files
import dictionary_errors as errors
from utils import split_feedback_into_dict


class FeedbackSystem(BackboneModel):
    """
    A system to generate LLM-based feedback on meeting summaries.
    Inherits from BackboneModel for consistent LLM calls.
    """

    def __init__(
        self,
        feedback_protocol: Dict[str, bool],
        config: Dict[str, Any],
        client_type: str = "openai",
        ensemble: bool = True,
        scoring_type: str = "existence",
    ):
        super().__init__(config, client_type)
        self.feedback_protocol = feedback_protocol
        self.ensemble = ensemble
        self.scoring_type = scoring_type

        # Loading error definitions from files
        self.FEEDBACK_PROMPTS = build_from_files(os.getenv("ERROR_TYPES_PATH", "src/error_types"))

        self.FEEDBACK_VERBOSITY_PROMPTS = {
            "location": "Please provide feedback on the location of the error. Where in the text does the error occur? Is it at the beginning, middle, or end of the text?",
            "reasoning": "Please provide reasoning why you think this passage contains an error? What makes you think this is an error?",
            "correction": "Please provide feedback on how to correct the error. What changes would you suggest to improve the passage?",
            "CoT": "Let's think step by step and describe every step you consider which leads you to the result that an error occurs or not."
        }
        self.FEEDBACK_SCORING_PROMPTS = {
            "existence": "Please provide feedback on the existence of the error. Does this passage contain an error? Answer 'yes' or 'no'. Then provide feedback on the quality of the passage. On a scale of 1 to 5, how much is the summary quality impacted by this error? 1 being the not impacted much and 5 being highly impacted. Assign a score of 0 if the error is not observed.",
        }

    def example_builder(self, example: Dict[str, Any]) -> (str, str):
        """
        Build a prompt snippet from a sample demonstrating an error.
        """
        transcript = example.get("transcript", "")
        summary = example.get("summary")
        if summary is None:
            raise errors.MissingExampleSummaryError(f"Summary for example '{example}' is missing.")
        score = example.get('score')
        if score is None:
            raise errors.MissingExampleScoreError(f"Score for example '{example}' is missing.")
        explanation = example.get('explanation')
        if explanation is None:
            raise errors.MissingExampleExplanationError(f"Explanation for example '{example}' is missing.")

        prompt = (
            f"Transcript: >>{transcript}<<\n"
            f"Predicted Summary: >>{summary}<<\n"
            f"Assigned Score: {score}\n"
            f"Explanation: {explanation}\n"
            f"---\n"
        )

        return prompt, transcript

    def expert_prompt_builder(self) -> str:
        return (
            "You are an experienced linguist and you will be given one summary for a meeting."
            "Your task is to rate the summary based on the existence of the below provided error type."
            "Please make sure you read and understand these instructions carefully."
            "Please keep this document open while reviewing, and refer to it as needed."
            "Please do not be too harsh and only point out errors if they are really an issue. Otherwise be more of a friendly reviewer."
            "Following is the error type(s) you should look for: \n"
        )

    def fewshot_prompt_builder(self, error_type: str) -> (str, str, str):
        """
        Build a portion of the system prompt that includes
        - the definition for this error type,
        - two examples (low and high severity),
        - the transcript for the high example (if needed).
        """
        definition_prompt = self.FEEDBACK_PROMPTS.get(error_type, {}).get("definition")
        if definition_prompt is None:
            raise errors.MissingDefinitionError(f"Definition for error type '{error_type}' is missing.")

        examples_dict = self.FEEDBACK_PROMPTS[error_type].get("example", {})
        low_example = examples_dict.get("low")
        high_example = examples_dict.get("high")
        if not low_example or not high_example:
            raise errors.MissingExampleError(f"Missing 'low' or 'high' example for error type {error_type}")

        low_prompt, _ = self.example_builder(low_example)
        high_prompt, example_transcript = self.example_builder(high_example)

        examples_prompt = (
            f"Below are two examples demonstrating the different impact levels of the previously described error type."
            f"Please learn from these examples the concept and how the rating works. \n"
            f"---\n"
            "Example 1:\n"
            f"{low_prompt}\n"
            "Example 2:\n"
            f"{high_prompt}\n"
        )

        return definition_prompt, examples_prompt, example_transcript

    def task_prompt_builder(self, scoring_type: str) -> (str, str):
        """
        Build the 'secondary tasks' prompt (verbosity) plus the 'primary task' prompt (scoring).
        """
        verbosity_prompt = ""
        for verbosity_type in self.FEEDBACK_VERBOSITY_PROMPTS:
            verbosity_prompt += f"{self.FEEDBACK_VERBOSITY_PROMPTS[verbosity_type]}\n" if self.feedback_protocol.get(verbosity_type) else ""

        scoring_prompt = self.FEEDBACK_SCORING_PROMPTS.get(scoring_type)
        return verbosity_prompt, scoring_prompt

    def evaluation_steps_prompt_builder(self) -> str:
        prompt = (
            "Evaluation steps: \n"
            "1. Read the transcript, if available, carefully and identify main topic and key points. \n"
            "2. Read the predicted summary and compare if it contains instances of the described error type. Note every instance you observe that is part of the error type. Only consider the error type and no other mistakes else. \n"
            "3. Rate the summary based on the existence of the error type with yes when at least one instance of the error type is found or no if the summary does not exhibit the error type. (primary task). \n"
            "4. You may be given secondary tasks, such as thinking step by step, explaining your decision, or pointing out the locations of each individual instance of the error type. These secondary tasks are designed to help you become more certain about your decision. \n"
            "5. Provide your findings in the desired format, so that your final output is a report on the existence of the error type in the given summary. \n"
            "Tip: Consider the whole input, i.e., the transcript and the predicted summary, provided in the user's prompt to make a good decision that humans will agree on. \n"
            "Please do not be too harsh. See it as if the summary is by a student you want to pass the exam. \n"
        )
        return prompt

    def system_prompt_builder(self, error_type: str, scoring_type: str) -> (str, str):
        """
        Build the entire system prompt for a specific error type, returning
        (system_prompt, example_transcript).
        """
        expert_prompt = self.expert_prompt_builder()
        definition_prompt, examples_prompt, example_transcript = self.fewshot_prompt_builder(error_type)
        verbosity_prompt, scoring_prompt = self.task_prompt_builder(scoring_type)
        evaluation_steps_prompt = self.evaluation_steps_prompt_builder()

        system_prompt = (
            f"{expert_prompt}\n\n"
            f"Definition: \n"
            f"\"{definition_prompt}\"\n"
            f"{evaluation_steps_prompt}\n\n"
            f"{examples_prompt}\n\n"
            f"Your secondary task: {verbosity_prompt}\n\n"
            f"Your primary task: {scoring_prompt}"
        )

        return system_prompt, example_transcript

    def user_prompt_builder(self, predicted_summary: str, transcript: str, example_transcript: bool = True) -> str:
        """
        Build a user portion prompt that includes the summary to analyze, optional transcript,
        and required output structure.
        """
        transcript_prompt = (
                f"If required, you can use the original transcript for look up: \n"
                f"\"{transcript}\"\n"
                if example_transcript else ""
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

        if self.feedback_protocol.get("location"):
            formatting_prompt += "## Location: <location> !!\n"
        if self.feedback_protocol.get("reasoning"):
            formatting_prompt += "## Explanation: <reasoning> !!\n"
        if self.feedback_protocol.get("correction"):
            formatting_prompt += "## Correction: <correction> !!\n"
        if self.feedback_protocol.get("CoT"):
            formatting_prompt += "## Chain-of-Thought: <CoT> !! \n"

        formatting_prompt += "## Existence result: <primary task: existence result> !!\n"
        formatting_prompt += "## Likert score result: <primary task: impact on quality result with (1) minor and (5) severe> !!\n"

        user_prompt = (
            f"{resources_prompt}\n\n"
            f"{formatting_prompt}\n\n"
        )

        return user_prompt

    def prompt_builder(
        self, error_type: str, scoring_type: str, predicted_summary: str, transcript: str
    ) -> list:
        """
        Assemble a final prompt array for the chat model, combining system and user messages.
        """
        logging.info("SYSTEM: >> building prompts")
        system_prompt, example_transcript = self.system_prompt_builder(error_type, scoring_type)
        user_prompt = self.user_prompt_builder(predicted_summary, transcript, example_transcript)

        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return prompt

    def parse_feedback(self, feedback: str) -> Dict[str, str]:
        """
        Splits at " !!\n" lines, returning a dict of the relevant pieces.
        """
        parts = feedback.split(" !!\n")
        feedback_dict = {}
        for part in parts:
            if ": " in part:
                key, value = part.split(": ", 1)
                feedback_dict[key.strip("# ").strip()] = value.strip()
        return feedback_dict

    def collect_feedback_to_df(self, dataset: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert the feedback-containing list of dicts into a Pandas DataFrame.
        Each example may have multiple errors; flatten them out as columns.
        """
        rows = []
        for example in dataset:
            row = {
                "Input": example.get('input', 'Transcript could not be retrieved!'),
                "Predicted": example.get('predicted', 'Predicted summary could not be retrieved!'),
                "Gold": example.get('gold', 'Gold summary could not be retrieved!'),
            }
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
                    logging.error("FEEDBACK MISTAKE!")
            rows.append(row)
        return pd.DataFrame(rows)

    def generate_feedback_multi(self, predicted_summary: str, transcript: str) -> Dict[str, Any]:
        """
        For each error type in FEEDBACK_PROMPTS, build a prompt and get feedback from the LLM.
        Return a dictionary of structured feedback for each error type plus the combined 'full_feedback'.
        """
        logging.info("SYSTEM: >> generate feedback")
        feedback_entries = {}
        full_feedback = ""

        for error, prompt in self.FEEDBACK_PROMPTS.items():
            prompt = self.prompt_builder(error, self.scoring_type, predicted_summary, transcript)
            feedback = self.safe_model_call(prompt, max_tokens=self.max_tokens_feedback)
            structured_feedback = self.parse_feedback(feedback)
            feedback_entries[error] = structured_feedback
            full_feedback += f"\n\n {feedback}"

        feedback_entries['full_feedback'] = full_feedback
        return feedback_entries

    def generate_feedback_single(self, predicted_summary: str, transcript: str) -> Dict[str, Any]:
        """
        If you want to combine all error types in one prompt. Currently not used if ensemble=True.
        """
        expert_prompt = self.expert_prompt_builder()

        error_prompts_total = ""
        for error, prompt in self.FEEDBACK_PROMPTS.items():
            definition_prompt, examples_prompt, _ = self.fewshot_prompt_builder(error)
            error_prompt = (
                f"Definition: \n"
                f"\"{definition_prompt}\"\n"
                f"{examples_prompt}\n\n"
                f"------------------------\n\n"
            )
            error_prompts_total += error_prompt

        verbosity_prompt, scoring_prompt = self.task_prompt_builder(self.scoring_type)

        system_prompt = (
            f"{expert_prompt}\n\n"
            f"{error_prompts_total}\n\n"
            f"Your secondary task: {verbosity_prompt}\n\n"
            f"Your primary task: {scoring_prompt} for all previously defined error types. Start each error part with '$$<ERROR TYPE>"
        )
        user_prompt = self.user_prompt_builder(predicted_summary, transcript, True)

        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        feedback_entries = {}
        full_feedback = ""
        full_feedback = self.safe_model_call(prompt, max_tokens=self.max_tokens_feedback)
        feedback_splitted = self.split_feedback_into_dict(full_feedback)

        for error_type, feedback_string in feedback_splitted.items():
            structured_feedback = self.parse_feedback(feedback_string)
            feedback_entries[error_type] = structured_feedback

        feedback_entries['full_feedback'] = full_feedback
        return feedback_entries

    def generate_feedback_dataset(self, dataset: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Iterates a list of examples (with 'Input','Predicted','Gold') and
        calls either generate_feedback_multi or generate_feedback_single for each.
        """
        logging.info("SYSTEM: >> generate dataset")
        updated_dataset = []
        generation_function = self.generate_feedback_multi if self.ensemble else self.generate_feedback_single

        for example in dataset:
            predicted_summary = example["Predicted"]
            transcript = example["Input"]
            feedback = generation_function(predicted_summary, transcript)
            feedback['input'] = transcript
            feedback['predicted'] = predicted_summary
            feedback['gold'] = example["Gold"]
            updated_dataset.append(feedback)
                
        return updated_dataset
