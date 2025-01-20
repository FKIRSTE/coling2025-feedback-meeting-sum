import logging
import os
from dotenv import load_dotenv
from single_run import run_single_cycle
from config_loader import load_config

def configure_logging(level=logging.INFO):
    """
    Configure logging format and level application-wide.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    )
    logging.info("Logging is configured.")


def main():
    # Load environment variables from .env
    load_dotenv()

    # Configure logging
    configure_logging()


    WORKING_PATH = os.getenv("WORKING_PATH", "dataset/cache")
    STARTING_DOCUMENT = os.getenv("STARTING_DOCUMENT", "dataset/input/GPT4-Turbo_Summaries.csv")
    iterations = os.getenv("ITERATIONS", 1)

    START_INDEX = os.getenv("START_INDEX", 0)
    STOP_INDEX = os.getenv("STOP_INDEX", iterations)

    config = load_config(os.getenv("RUN_CONFIG"))
    FB_REASONING = config.get("fb_reasoning", True)
    FB_CORRECTION = config.get("fb_correction", False)
    FB_COT = config.get("fb_cot", True)

    TRANSITION_PROTOCOL = config.get("transition_protocol", "feedback")
    RF_TRANSCRIPT = config.get("rf_transcript", False)
    RF_COT = config.get("rf_cot", True)
    RF_CORRECTION = config.get("rf_correction", False)

    for looper in range(START_INDEX, STOP_INDEX):
        working_document = f"Run-{looper}_1.csv"
        logging.info("\n\n*************** FEEDBACK PHASE ***************\n")

        run_single_cycle(
            in_document=STARTING_DOCUMENT,
            out_document=working_document,
            generate_feedback=True,
            generate_refinement=False,
            feedback_protocol={"reasoning": FB_REASONING, "correction": FB_CORRECTION, "CoT": FB_COT},
            scoring_type="existence",
            ensemble_feedback=True,
            feedback_path=WORKING_PATH
        )

        # Update to newly generated file
        STARTING_DOCUMENT = f"{WORKING_PATH}/{working_document}"

        working_document = f"Run-{looper}_2.csv"
        logging.info("\n\n*************** REFINEMENT PHASE ***************\n")

        run_single_cycle(
            in_document=STARTING_DOCUMENT,
            out_document=working_document,
            generate_feedback=False,
            generate_refinement=True,
            feedback_protocol={"transcript": RF_TRANSCRIPT, "CoT": RF_COT, "correction": RF_CORRECTION},
            scoring_type="",
            ensemble_feedback=True,
            transition_protocol=TRANSITION_PROTOCOL,
            feedback_path=WORKING_PATH
        )

        STARTING_DOCUMENT = f"{WORKING_PATH}/{working_document}"


if __name__ == "__main__":
    main()
