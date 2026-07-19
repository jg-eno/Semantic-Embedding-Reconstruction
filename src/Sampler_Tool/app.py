"""
Usage:
    python app.py
    open http://localhost:5000
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, render_template, request, jsonify
from config import (
    CONFIG, DECODER_BASE_NAME, ENCODER_MODEL_NAME,
    MAX_NEW_TOKENS, SAMPLER_DEFAULTS, SAMPLE_PARAGRAPH,
)
from sampler import Sampler

app = Flask(__name__)

# Single shared sampler instance — reloads models when config changes.
_sampler: Sampler | None = None
_sampler_cfg_key: tuple | None = None  # tracks which config is loaded


def _get_sampler(cfg: dict, base_model: str, **kwargs) -> Sampler:
    global _sampler, _sampler_cfg_key
    key = (cfg["repo"], cfg["filename"], cfg["prefix_len"], base_model)
    if _sampler is None or _sampler_cfg_key != key:
        if _sampler is not None:
            _sampler.unload()
        from factory import InverterFactory
        from models.encoder import Encoder
        # Pass base_model override via a custom factory subclass isn't needed —
        # we just swap the config and let Sampler lazy-load on first .sample().
        _sampler = Sampler(cfg=cfg, **kwargs)
        _sampler_cfg_key = key
    else:
        # Update sampling params without reloading models.
        for k, v in kwargs.items():
            setattr(_sampler, k, v)
    return _sampler


@app.route("/")
def index():
    defaults = {
        "repo": CONFIG["repo"],
        "filename": CONFIG["filename"],
        "prefix_len": CONFIG["prefix_len"],
        "encoder_model": ENCODER_MODEL_NAME,
        "base_model": DECODER_BASE_NAME,
        "sentence": SAMPLE_PARAGRAPH,
        "n": 5,
        "max_new_tokens": MAX_NEW_TOKENS,
        **SAMPLER_DEFAULTS,
    }
    return render_template("index.html", defaults=defaults)


@app.route("/sample", methods=["POST"])
def sample():
    data = request.get_json()

    cfg = {
        "repo": data["repo"],
        "filename": data["filename"],
        "prefix_len": int(data["prefix_len"]),
    }
    kwargs = {
        "max_new_tokens": int(data["max_new_tokens"]),
        "temperature": float(data["temperature"]),
        "top_p": float(data["top_p"]),
        "top_k": int(data["top_k"]),
        "repetition_penalty": float(data["repetition_penalty"]),
        "temperature_step": float(data["temperature_step"]),
    }

    try:
        sampler = _get_sampler(cfg, data["base_model"], **kwargs)
        result = sampler.sample(data["sentence"], n=int(data["n"]))
        return jsonify({"status": "ok", "samples": result.samples})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
