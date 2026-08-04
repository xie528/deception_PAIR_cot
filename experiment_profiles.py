"""Named experiment profiles for PAIR reproduction and cheap development."""

PAIR_PAPER_PROFILE = {
    # Chao et al. primarily used Mixtral 8x7B Instruct as the attacker.
    "attack_model": "mixtral",
    "attack_max_n_tokens": 500,
    "attack_temperature": 1.0,
    "attack_top_p": 0.9,
    # Vicuna-13B is the default open target in the paper's efficiency study.
    "target_model": "vicuna-13b-v1.5",
    "target_max_n_tokens": 150,
    "target_temperature": 0.0,
    # Main paper evaluation used Llama Guard through JailbreakBench.
    "judge_model": "jailbreakbench",
    "judge_max_n_tokens": 10,
    "judge_temperature": 0.0,
    "n_streams": 30,
    "n_iterations": 3,
    "hidden_cot_defense": False,
}


PILOT_SMALL_PROFILE = {
    # Cheap development models; this is not the final paper configuration.
    # Mixtral 8x7B is not currently served by OpenRouter, so use a hosted 8B model.
    "attack_model": "qwen3-8b",
    "attack_max_n_tokens": 500,
    "attack_temperature": 1.0,
    "attack_top_p": 0.9,
    # Qwen3 8B supports returned reasoning and currently has a live endpoint.
    "target_model": "qwen/qwen3-8b",
    "target_max_n_tokens": 512,
    "target_temperature": 0.0,
    "cot_judge_model": "gpt-4.1-mini",
    "pair_judge_model": "gpt-4.1-mini",
    "judge_max_n_tokens": 200,
    "judge_temperature": 0.0,
    "n_streams": 3,
    "n_iterations": 3,
    "hidden_cot_defense": True,
    "max_regenerations": 3,
}


MEANINGFUL_PILOT_PROFILE = {
    # A stronger hosted attacker paired with an affordable reasoning target.
    "attack_model": "qwen3-32b",
    "attack_max_n_tokens": 500,
    "attack_temperature": 1.0,
    "attack_top_p": 0.9,
    "target_model": "deepseek/deepseek-r1-distill-qwen-14b",
    "target_max_n_tokens": 512,
    "target_temperature": 0.0,
    "cot_judge_model": "gpt-4.1",
    "pair_judge_model": "gpt-4.1",
    "judge_max_n_tokens": 200,
    "judge_temperature": 0.0,
    # More attack coverage than plumbing, without paying for the 30-stream main run.
    "n_streams": 5,
    "n_iterations": 3,
    "hidden_cot_defense": True,
    "max_regenerations": 3,
}


MAIN_HIDDEN_COT_PROFILE = {
    # Preserve PAIR's search budget; use a currently hosted strong attacker.
    "attack_model": "qwen3-32b",
    "attack_max_n_tokens": 500,
    "attack_temperature": 1.0,
    "attack_top_p": 0.9,
    # Reasoning-capable replacement for PAIR's non-reasoning Vicuna target.
    "target_model": "deepseek/deepseek-r1-distill-qwen-32b",
    "target_max_n_tokens": 512,
    "target_temperature": 0.0,
    "cot_judge_model": "gpt-4.1",
    "pair_judge_model": "gpt-4.1",
    "judge_max_n_tokens": 200,
    "judge_temperature": 0.0,
    "n_streams": 30,
    "n_iterations": 3,
    "hidden_cot_defense": True,
    "max_regenerations": 3,
}


PILOT_STRONG_PROFILE = {
    "attack_model": "deepseek-v4-flash",
    "attack_max_n_tokens": 1200,
    "attack_temperature": 0.8,
    "attack_top_p": 0.9,
    "target_model": "qwen/qwen3-32b",
    "target_max_n_tokens": 700,
    "target_temperature": 0.0,
    "cot_judge_model": "openai/gpt-5.2",
    "cot_judge_base_url": "https://openrouter.ai/api/v1",
    "cot_judge_api_key_env": "OPENROUTER_API_KEY",
    "pair_judge_model": "google/gemini-3.1-pro-preview",
    "judge_max_n_tokens": 800,
    "judge_temperature": 0.0,
    "n_streams": 3,
    "n_iterations": 3,
    "hidden_cot_defense": True,
    "max_regenerations": 3,
}

PILOT_STRONG_SMOKE_PROFILE = {
    **PILOT_STRONG_PROFILE,
    "n_streams": 1,
    "n_iterations": 1,
}

PILOT_STRONG_LARGE_PROFILE = {
    **PILOT_STRONG_PROFILE,
    "target_max_n_tokens": 1200,
    "reasoning_max_tokens": 768,
    "n_streams": 3,
    "n_iterations": 5,
}

PILOT_STRONG_LONG_PROFILE = {
    **PILOT_STRONG_PROFILE,
    "target_max_n_tokens": 1200,
    "reasoning_max_tokens": 768,
    "n_streams": 1,
    "n_iterations": 20,
}

PILOT_STRONG_DEPTH10_PROFILE = {
    **PILOT_STRONG_PROFILE,
    "target_max_n_tokens": 1200,
    "reasoning_max_tokens": 768,
    # With one stream, PAIR deterministically uses its first (role-playing)
    # system prompt. Ten iterations trade some depth for broader objectives.
    "n_streams": 1,
    "n_iterations": 10,
}


PROFILES = {
    "paper": PAIR_PAPER_PROFILE,
    "pilot-small": PILOT_SMALL_PROFILE,
    "meaningful-pilot": MEANINGFUL_PILOT_PROFILE,
    "main-hidden-cot": MAIN_HIDDEN_COT_PROFILE,
    "pilot-strong": PILOT_STRONG_PROFILE,
    "pilot-strong-smoke": PILOT_STRONG_SMOKE_PROFILE,
    "pilot-strong-large": PILOT_STRONG_LARGE_PROFILE,
    "pilot-strong-long": PILOT_STRONG_LONG_PROFILE,
    "pilot-strong-depth10": PILOT_STRONG_DEPTH10_PROFILE,
}


def apply_profile(args):
    """Apply a named profile after argparse has created the namespace."""
    for name, value in PROFILES[args.profile].items():
        setattr(args, name, value)
    # The original AttackLM reads these sampling values from config.
    import config

    config.ATTACK_TEMP = args.attack_temperature
    config.ATTACK_TOP_P = args.attack_top_p
    config.TARGET_TEMP = args.target_temperature
    return args
