import logging
import random
import time
from typing import Any, Dict
from openai import AzureOpenAI

class BackboneModel:
    """
    Base class that initializes the AzureOpenAI client and
    provides helper methods for safe calls with rate-limit handling.
    """

    def __init__(self, config: Dict[str, Any], client_type: str = "openai"):
        self.model_name = config["model"]
        self.max_tokens_feedback = config.get("max_tokens_feedback", 4000)
        self.max_tokens_refinement = config.get("max_tokens_refinement", 200)
        self.client = self.init_model(config, client_type)

    def init_model(self, config: Dict[str, Any], client_type: str) -> AzureOpenAI:
        """
        Initializes the Azure OpenAI client from config.
        """
        api_key = config["api_key"]
        api_version = config["api_version"]
        endpoint = config["endpoint"]
        logging.info("Initializing AzureOpenAI client...")

        if client_type == "openai":
            print(api_key, api_version, endpoint)
            client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=endpoint
            )
            return client
        else:
            raise ValueError(f"Unknown client type: {client_type}")

    def model_call(self, messages: list, max_tokens: int) -> str:
        """
        Wrapper for chat.completions.create. Raises an exception on failure.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
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
            logging.error("API call failed: %s", str(e))
            raise

    def safe_model_call(self, messages: list, max_tokens: int, max_attempts: int = 6, base_delay: float = 3.0) -> str:
        """
        Retries model calls with exponential backoff if a 429 is encountered.
        """
        attempt = 0
        while attempt < max_attempts:
            try:
                return self.model_call(messages, max_tokens)
            except Exception as e:
                if "429" in str(e):
                    sleep_time = (2 ** (attempt + 1)) + (random.randint(0, 1000) / 1000.0)
                    logging.warning("Rate limit hit, backing off for %.2f seconds.", sleep_time)
                    time.sleep(sleep_time)
                    attempt += 1
                else:
                    logging.error("Error encountered: %s", str(e))
                    break
            finally:
                time.sleep(base_delay)

        raise RuntimeError("Max attempts for safe_model_call exceeded.")
