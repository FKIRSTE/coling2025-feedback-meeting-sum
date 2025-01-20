import os
import logging
import argparse
import json
from datasets import Features, Value
from config_loader import load_config
from feedback_system import FeedbackSystem
from refinement_system import RefinementSystem
from utils import get_dataset, get_dataframe

OUTPUT_PATH = os.getenv("WORKING_PATH", "dataset/cache")
CONFIG_PATH = os.getenv("CONFIG_PATH")


def run_single_cycle(in_document='', out_document='', generate_feedback=True, generate_refinement=True,
                     feedback_protocol={"location": False, "reasoning": True, "correction": False, "CoT": False},
                     scoring_type='existence', ensemble_feedback=False,
                     transition_protocol='feedback', feedback_path=f"{OUTPUT_PATH}/"):
    
    config = load_config(CONFIG_PATH)
    
    if generate_feedback:
        features = Features({
            'Input': Value('string'),
            'Predicted': Value('string'),
            'Gold': Value('string'),
        })
        dataset = get_dataset(in_document,  features)
        feedback_system = FeedbackSystem(feedback_protocol=feedback_protocol,
                                         config=config,
                                         ensemble=ensemble_feedback,
                                         scoring_type=scoring_type)
        feedback_complete = feedback_system.generate_feedback_dataset(dataset['train'])
        feedback_df = feedback_system.collect_feedback_to_df(feedback_complete)
        feedback_df.to_csv(f"{OUTPUT_PATH}/{out_document}", index=False)
        
    if generate_refinement:
        feedback_df = get_dataframe(feedback_path)
        refinement_system = RefinementSystem(feedback_protocol=feedback_protocol, config=config, transition_protocol=transition_protocol)
        refinement_system.refine_summaries_dataset(feedback_df)
        feedback_df.to_csv(f"{OUTPUT_PATH}/{out_document}", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run single feedback-refinement cycle.')
    parser.add_argument('--in_document', type=str, required=False, default='', help='Input document path')
    parser.add_argument('--out_document', type=str, required=True, help='Output document path')
    parser.add_argument('--generate_feedback', type=bool, required=False, default=True, help='Generate feedback flag')
    parser.add_argument('--generate_refinement', type=bool, required=False, default=True, help='Generate refinement flag')
    parser.add_argument('--feedback_protocol', type=str, required=False, default='{"reasoning": true, "correction": false, "CoT": false, "transcript": false}', help='Feedback protocol')
    parser.add_argument('--scoring_type', type=str, required=False, default='existence', help='Scoring type')
    parser.add_argument('--ensemble_feedback', type=bool, required=False, default=False, help='Ensemble feedback flag')
    parser.add_argument('--transition_protocol', type=str, required=False, default='feedback', help='Transition protocol')
    parser.add_argument('--feedback_path', type=str, required=False, default=f"{OUTPUT_PATH}/", help='Feedback path')
    
    args = parser.parse_args()
    feedback_protocol = json.loads(args.feedback_protocol)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    run_single_cycle(args.in_document, args.out_document, args.generate_feedback, args.generate_refinement,
                     feedback_protocol, args.scoring_type, args.ensemble_feedback,
                     args.transition_protocol, args.feedback_path)
