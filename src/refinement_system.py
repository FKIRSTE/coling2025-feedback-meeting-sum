import logging
from typing import Dict, Any
import pandas as pd

from backbone_model import BackboneModel
from utils import split_feedback_into_dict, assemble_feedback_from_dict


class RefinementSystem(BackboneModel):
    """
    A system to refine a predicted summary using a feedback report.
    """

    def __init__(
        self,
        feedback_protocol: Dict[str, bool],
        config: Dict[str, Any],
        transition_protocol: str = "feedback"
    ):
        super().__init__(config)
        self.feedback_protocol = feedback_protocol
        self.transition_protocol = transition_protocol

    def pre_process_feedback(self, feedback: str) -> Dict[str, str]:
        """
        Splits feedback into dictionary blocks. Optionally removes sections
        not relevant if 'CoT' or 'correction' are False.
        """
        sections_to_ignore = []
        if not self.feedback_protocol.get("CoT", False):
            sections_to_ignore.append("Chain-of-Thought")
        if not self.feedback_protocol.get("correction", False):
            sections_to_ignore.append("Correction")

        feedback_splits = split_feedback_into_dict(feedback, sections_to_ignore)

        # If the transition protocol is 'feedback', we just return everything
        if self.transition_protocol == "feedback":
            return feedback_splits


        # If the transition protocol is 'consolidate', let's do something else
        if self.transition_protocol == "consolidate":
            # Filter out negative/no errors
            considered_splits = {
                error_type: content
                for error_type, content in feedback_splits.items()
                if "N/A" not in content and "Overall result: No" not in content and "Overall result: 0" not in content
            }

            positive_feedback = assemble_feedback_from_dict(considered_splits)



        considered_splits = {
            error_type: content
            for error_type, content in feedback_splits.items()
            if "N/A" not in content and "Overall result: No" not in content and "Overall result: 0" not in content
        }

        feedback_splits = considered_splits

        if self.transition_protocol == "consolidate":
            positive_feedback = assemble_feedback_from_dict(feedback_splits)
            prompt = [
                {"role": "system", "content": "You are a professional feedback summarizer, that provides a comprehensive, direct version of a feedback report. You condensed version should be usable for someone to improve their previous summary effectively. So you are allowed to structure it in the most effective way to address the feedback. The refinement should be successful purely from your feedback and the previous summary so include all relevant details given in the report."},
                {"role": "user", "content": f"Please consolidate the following feedback into a plan and provide usable feedback: {positive_feedback}. Use the output format 'Add: <Add the information of ...> \n Remove: <Remove the information of ...> \n Rephrase: <Rephrase the information of ...> \n Simplify: <Shorten the summary regarding ...> \n Keep: <Keep the summary unchanged at ...>'. Include all details from the feedback. Make your answer super precise like you are micro-managing a really limited user and need to provide every little detail and plan how to resolve the issues. As the refiner does not know about the original meeting transcript, you need to show clear examples for each error, why it is a mistake."},
            ]

            consolidated_feedback = self.safe_model_call(prompt, self.max_tokens_feedback)

            return consolidated_feedback

    def refine_summary(self, predicted_summary, feedback):
        system_message = (
            "You are an expert in refining and improving summaries."
            "Your task is to improve the summaries of conversations based on a given feedback report."
            "All the content to improve the original summary and make it the very best is provided in the review, as the reviewer provides all details."
        )

        user_message = (
            f"Please improve this summary: \n"
            f"'{predicted_summary}' \n"
            f"considering this review: \n"
            f"'{feedback}' \n"
        )

        prompt = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        refined_summary = self.safe_model_call(prompt, self.max_tokens_refinement)
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

    def refine_summaries_dataset(self, dataset):
        refine_function = self.refine_summary_full
        # name = f"refined_summary-{self.transition_protocol}"

        dataset['predicted_original'] = dataset['Predicted']
        
        dataset['Predicted'] = dataset.apply(lambda row: refine_function(
            row['predicted_original'], self.pre_process_feedback(row['full_feedback']), row['Input']), axis=1)
