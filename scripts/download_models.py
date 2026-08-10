"""Pre-download all Hugging Face models so the first request in the demo
doesn't trigger a multi-hundred-megabyte download in front of the examiner.

Run this once after `pip install -r requirements.txt`:

    python -m scripts.download_models
"""
from __future__ import annotations

import logging

from moodsync.config import MODEL_CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    log.info("Downloading vision model: %s", MODEL_CONFIG.vision_model)
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    AutoImageProcessor.from_pretrained(MODEL_CONFIG.vision_model)
    AutoModelForImageClassification.from_pretrained(MODEL_CONFIG.vision_model)

    log.info("Downloading text model: %s", MODEL_CONFIG.text_model)
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    AutoTokenizer.from_pretrained(MODEL_CONFIG.text_model)
    AutoModelForSequenceClassification.from_pretrained(MODEL_CONFIG.text_model)

    log.info("Downloading audio emotion model: %s", MODEL_CONFIG.audio_emotion_model)
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    AutoFeatureExtractor.from_pretrained(MODEL_CONFIG.audio_emotion_model)
    AutoModelForAudioClassification.from_pretrained(MODEL_CONFIG.audio_emotion_model)

    log.info("Downloading Whisper: %s", MODEL_CONFIG.asr_model)
    import whisper
    whisper.load_model(MODEL_CONFIG.asr_model)

    log.info("Downloading generator: %s", MODEL_CONFIG.generator_model)
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer as Tok
    Tok.from_pretrained(MODEL_CONFIG.generator_model)
    AutoModelForSeq2SeqLM.from_pretrained(MODEL_CONFIG.generator_model)

    log.info("✅ All models cached.")


if __name__ == "__main__":
    main()
