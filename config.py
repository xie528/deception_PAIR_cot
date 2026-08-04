from enum import Enum
VICUNA_PATH = "/home/pchao/vicuna-13b-v1.5"
LLAMA_PATH = "/home/pchao/Llama-2-7b-chat-hf"

ATTACK_TEMP = 1
TARGET_TEMP = 0
ATTACK_TOP_P = 0.9
TARGET_TOP_P = 1


## MODEL PARAMETERS ##
class Model(Enum):
    vicuna = "vicuna-13b-v1.5"
    llama_2 = "llama-2-7b-chat-hf"
    gpt_3_5 = "gpt-3.5-turbo-1106"
    gpt_4 = "gpt-4-0125-preview"
    claude_1 = "claude-instant-1.2"
    claude_2 = "claude-2.1"
    gemini = "gemini-pro"
    mixtral = "mixtral"
    qwen3_8b = "qwen3-8b"
    qwen3_32b = "qwen3-32b"
    deepseek_v4_flash = "deepseek-v4-flash"
    gpt_4_1_nano = "gpt-4.1-nano"
    gpt_4_1_mini = "gpt-4.1-mini"
    gpt_4_1 = "gpt-4.1"

MODEL_NAMES = [model.value for model in Model]


HF_MODEL_NAMES: dict[Model, str] = {
    Model.llama_2: "meta-llama/Llama-2-7b-chat-hf",
    Model.vicuna: "lmsys/vicuna-13b-v1.5",
    Model.mixtral: "mistralai/Mixtral-8x7B-Instruct-v0.1"
}

TOGETHER_MODEL_NAMES: dict[Model, str] = {
    Model.llama_2: "together_ai/togethercomputer/llama-2-7b-chat",
    Model.vicuna: "together_ai/lmsys/vicuna-13b-v1.5",
    Model.mixtral: "together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1"
}

OPENROUTER_MODEL_NAMES: dict[Model, str] = {
    Model.qwen3_8b: "openrouter/qwen/qwen3-8b",
    Model.qwen3_32b: "openrouter/qwen/qwen3-32b",
    Model.deepseek_v4_flash: "openrouter/deepseek/deepseek-v4-flash",
}

FASTCHAT_TEMPLATE_NAMES: dict[Model, str] = {
    Model.gpt_3_5: "gpt-3.5-turbo",
    Model.gpt_4: "gpt-4",
    Model.claude_1: "claude-instant-1.2",
    Model.claude_2: "claude-2.1",
    Model.gemini: "gemini-pro",
    Model.vicuna: "vicuna_v1.1",
    Model.llama_2: "llama-2-7b-chat-hf",
    Model.mixtral: "mixtral",
    Model.qwen3_8b: "gpt-4",
    Model.qwen3_32b: "gpt-4",
    Model.deepseek_v4_flash: "gpt-4",
    Model.gpt_4_1_nano: "gpt-4",
    Model.gpt_4_1_mini: "gpt-4",
    Model.gpt_4_1: "gpt-4",
}

API_KEY_NAMES: dict[Model, str] = {
    Model.gpt_3_5:  "OPENAI_API_KEY",
    Model.gpt_4:    "OPENAI_API_KEY",
    Model.claude_1: "ANTHROPIC_API_KEY",
    Model.claude_2: "ANTHROPIC_API_KEY",
    Model.gemini:   "GEMINI_API_KEY",
    Model.vicuna:   "TOGETHER_API_KEY",
    Model.llama_2:  "TOGETHER_API_KEY",
    Model.mixtral:  "TOGETHER_API_KEY",
    Model.qwen3_8b: "OPENROUTER_API_KEY",
    Model.qwen3_32b: "OPENROUTER_API_KEY",
    Model.deepseek_v4_flash: "OPENROUTER_API_KEY",
    Model.gpt_4_1_nano: "OPENAI_API_KEY",
    Model.gpt_4_1_mini: "OPENAI_API_KEY",
    Model.gpt_4_1: "OPENAI_API_KEY",
}

LITELLM_TEMPLATES: dict[Model, dict] = {
    Model.vicuna: {"roles":{
                    "system": {"pre_message": "", "post_message": " "},
                    "user": {"pre_message": "USER: ", "post_message": " ASSISTANT:"},
                    "assistant": {
                        "pre_message": "",
                        "post_message": "",
                    },
                },
                "post_message":"</s>",
                "initial_prompt_value" : "",
                "eos_tokens": ["</s>"]         
                },
    Model.llama_2: {"roles":{
                    "system": {"pre_message": "[INST] <<SYS>>\n", "post_message": "\n<</SYS>>\n\n"},
                    "user": {"pre_message": "", "post_message": " [/INST]"},
                    "assistant": {"pre_message": "", "post_message": ""},
                },
                "post_message" : " </s><s>",
                "initial_prompt_value" : "",
                "eos_tokens" :  ["</s>", "[/INST]"]  
            },
    Model.mixtral: {"roles":{
                    "system": {
                        "pre_message": "[INST] ",
                        "post_message": " [/INST]"
                    },
                    "user": { 
                        "pre_message": "[INST] ",
                        "post_message": " [/INST]"
                    }, 
                    "assistant": {
                        "pre_message": " ",
                        "post_message": "",
                    }
                },
                "post_message": "</s>",
                "initial_prompt_value" : "<s>",
                "eos_tokens": ["</s>", "[/INST]"]
    }
}
