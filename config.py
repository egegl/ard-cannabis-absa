"""Study constants, prompts and shared helpers."""

import hashlib
import json
import os

SYNTH_MODEL = "gpt-5.4-mini-2026-03-17"
SYNTH_REASONING_EFFORT = "medium"
MAX_OUTPUT_TOKENS = 25_000
POSTS_PER_CALL = 5

POSTS_SCHEMA = {
    "type": "object",
    "properties": {
        "posts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "post_text": {"type": "string"},
                    "aspect": {"type": "string"},
                },
                "required": ["post_text", "aspect"],
                "additionalProperties": False,
            },
            "minItems": POSTS_PER_CALL,
            "maxItems": POSTS_PER_CALL,
        }
    },
    "required": ["posts"],
    "additionalProperties": False,
}

TEXT_FORMAT = {
    "format": {
        "type": "json_schema",
        "name": "synthetic_reddit_posts",
        "strict": True,
        "schema": POSTS_SCHEMA,
    }
}

SOURCE_CSV = "final_sentiment_dataset.csv"
N_ROWS = 479
N_POSTS = 374

TASKS = ("absa", "trad")
LABELS = (0, 1, 2)
LABEL_TO_SENTIMENT = {
    0: "neutral (no strong opinion)",
    1: "positive (supportive/favorable)",
    2: "negative (critical/unfavorable)",
}

SPLIT_SEED = 42
DEMO_MASTER_SEED = 42
K_DEMONSTRATIONS = 9
MODEL_SEEDS = (42, 7, 123, 2024, 99)  # model init only

FOLDS = ("train", "val", "test")
FOLD_FRACTIONS = {"train": 0.70, "val": 0.10, "test": 0.20}
SPLIT_MAX_PASSES = 20

GEN_TARGETS = (100, 200, 300, 400, 500)

# acceptance checks for generated posts
ASPECT_MAX_CHARS = 50
POST_MIN_CHARS = 10
POST_MAX_CHARS = 4000
NEAR_DUP_NGRAM = 5
NEAR_DUP_JACCARD = 0.90

MODELS = {
    "trad": ("microsoft/deberta-v3-base", "8ccc9b6f36199bec6961081d44eb72fb3f7353f3"),
    "absa": ("yangheng/deberta-v3-base-absa-v1.1", "10c9dff335a44073e1352360c3a7bc54dc58eb01"),
}

HPARAMS = {
    "learning_rate": 2e-5,
    "effective_batch_size": 16,
    "num_epochs": 10,
    "early_stopping_patience": 3,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "max_length": 510,
    "max_grad_norm": 1.0,
    "label_smoothing": 0.0,
}

TIE_TOLERANCE = 0.005
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 42

CONDITIONS = (
    "real_unweighted",
    "real_weighted",
    "augmented_unweighted",
    "augmented_weighted",
)

# --- generation prompts ---

_ABSA_BEFORE = (
    "You are generating a synthetic Reddit post about autoimmune rheumatic diseases (ARD) from a real Reddit user.\n"
    "The posts come from ARD related subreddits.\n"
    "All of the following example posts express a similar sentiment towards marijuana/cannabis/cbd.\n"
    "Your job is to generate a new Reddit post that could plausibly come from the same kinds of users "
    "and subreddits, with the same sentiment polarity towards marijuana/cannabis/cbd as the examples, "
    "varied in writing style, tone, punctuation, and grammar.\n\n"
    "These posts will be used as synthetic data for Aspect-Based Sentiment Analysis (ABSA). "
    "Each post discusses a specific cannabis-related aspect (like 'cbd', 'cannabis', 'marijuana', 'thc', "
    "'weed', 'medical marijuana', etc.). "
    "The sentiment analysis will be performed specifically around the aspect mentioned in each post.\n\n"
    "Here are example posts from a single sentiment class, with their corresponding aspects:\n\n"
)

_ABSA_AFTER = (
    "\nAll of these examples belong to the same sentiment class toward marijuana/cannabis "
    "(label = <<LABEL>>, meaning <<SENTIMENT>>). "
    "Keep the overall sentiment towards marijuana/cannabis consistent with these examples.\n\n"
    "Now generate 5 NEW, distinct and varying Reddit posts.\n"
    "Each post should:\n"
    "- Be plausible as a real Reddit post from ARD related subreddits.\n"
    "- Match the style, tone, and topic distribution of the examples.\n"
    "- Have the SAME overall sentiment towards marijuana/cannabis as the examples.\n"
    "- Not copy the examples (use different examples and wording; be creative).\n"
    "- NOT reuse structure, phrases, or narrative patterns from the examples. Each post must feel "
    "uniquely written by a different person.\n"
    "- Contain at most 500 tokens of model output per post (keep each post reasonably short).\n"
    "- Vary in length. It can be short (one or a few sentences/words), or long (a couple paragraphs), "
    "as long as it stays under the 500-token limit.\n"
    "- Focus on a specific cannabis-related aspect (choose from: 'cbd', 'cannabis', 'marijuana', 'thc', "
    "'weed', 'medical marijuana', 'pot' or similar terms).\n"
    "- The aspect MUST be mentioned verbatim in the post text itself, since sentiment analysis will be "
    "performed around this specific aspect.\n"
    "Format each post so that the sentiment is clearly directed toward the chosen cannabis-related "
    "aspect, not just general emotions. Return your answer as strict JSON following this schema:\n"
    "{\n"
    "   \"posts\": [\n"
    "       {\n"
    "           \"post_text\": \"string - the full text of one Reddit post\",\n"
    "           \"aspect\": \"string - the cannabis-related aspect discussed in this post\"\n"
    "       },\n"
    "       ... (4 more posts)\n"
    "   ]\n"
    "}\n"
)

_TRAD_BEFORE = (
    "You are generating a synthetic Reddit post about autoimmune rheumatic diseases (ARD) from a real Reddit user.\n"
    "The posts come from ARD related subreddits.\n"
    "All of the following example posts share a similar OVERALL sentiment -- the general tone/mood of "
    "the post as a whole.\n"
    "Your job is to generate a new Reddit post that could plausibly come from the same kinds of users "
    "and subreddits, with the same OVERALL sentiment as the examples, varied in writing style, tone, "
    "punctuation, and grammar.\n\n"
    "These posts will be used as synthetic data for traditional (whole-post) sentiment analysis. "
    "Each post mentions a specific cannabis-related aspect (like 'cbd', 'cannabis', 'marijuana', 'thc', "
    "'weed', 'medical marijuana', etc.), but the label reflects the sentiment of the post as a whole -- "
    "NOT sentiment directed specifically at that cannabis aspect.\n\n"
    "Here are example posts from a single sentiment class, with their corresponding aspects:\n\n"
)

_TRAD_AFTER = (
    "\nAll of these examples belong to the same OVERALL sentiment class "
    "(label = <<LABEL>>, meaning <<SENTIMENT>>). "
    "Keep the overall tone/mood of each generated post consistent with these examples.\n\n"
    "Now generate 5 NEW, distinct and varying Reddit posts.\n"
    "Each post should:\n"
    "- Be plausible as a real Reddit post from ARD related subreddits.\n"
    "- Match the style, tone, and topic distribution of the examples.\n"
    "- Have the SAME OVERALL sentiment (whole-post tone) as the examples.\n"
    "- Not copy the examples (use different examples and wording; be creative).\n"
    "- NOT reuse structure, phrases, or narrative patterns from the examples. Each post must feel "
    "uniquely written by a different person.\n"
    "- Contain at most 500 tokens of model output per post (keep each post reasonably short).\n"
    "- Vary in length. It can be short (one or a few sentences/words), or long (a couple paragraphs), "
    "as long as it stays under the 500-token limit.\n"
    "- Mention a specific cannabis-related aspect (choose from: 'cbd', 'cannabis', 'marijuana', 'thc', "
    "'weed', 'medical marijuana', 'pot' or similar terms) somewhere in the post.\n"
    "- The aspect MUST be mentioned verbatim in the post text itself.\n"
    "Return your answer as strict JSON following this schema:\n"
    "{\n"
    "   \"posts\": [\n"
    "       {\n"
    "           \"post_text\": \"string - the full text of one Reddit post\",\n"
    "           \"aspect\": \"string - the cannabis-related aspect discussed in this post\"\n"
    "       },\n"
    "       ... (4 more posts)\n"
    "   ]\n"
    "}\n"
)

PROMPT_TEMPLATES = {
    "absa": {"before": _ABSA_BEFORE, "after": _ABSA_AFTER},
    "trad": {"before": _TRAD_BEFORE, "after": _TRAD_AFTER},
}


def build_prompt(task, label, demonstrations):
    tpl = PROMPT_TEMPLATES[task]
    example_lines = [
        f"Example {i} (aspect: '{d['aspect']}'): {d['input_text']}\n"
        for i, d in enumerate(demonstrations, start=1)
    ]
    after = tpl["after"].replace("<<LABEL>>", str(int(label))).replace(
        "<<SENTIMENT>>", LABEL_TO_SENTIMENT[int(label)]
    )
    return tpl["before"] + "\n".join(example_lines) + after


def request_body(prompt):
    return {
        "model": SYNTH_MODEL,
        "input": prompt,
        "reasoning": {"effort": SYNTH_REASONING_EFFORT},
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "text": TEXT_FORMAT,
    }


# --- ids ---

def row_id(i):
    return f"row_{i:04d}"


def real_instance_id(source_row_id):
    return f"real::{source_row_id}"


def synth_instance_id(task, label, call_index, item_index):
    return f"synth::{task}::{int(label)}::{call_index:04d}::{item_index}"


def synth_post_id(task, label, call_index, item_index):
    return f"synthpost::{task}::{int(label)}::{call_index:04d}::{item_index}"


def select_demonstrations(task, label, call_index, pool_row_ids, k=K_DEMONSTRATIONS):
    """k training rows chosen by seeded sha256 ranking, varies per call index"""
    def key(rid):
        material = f"{DEMO_MASTER_SEED}|{task}|{int(label)}|{call_index}|{rid}"
        return hashlib.sha256(material.encode()).hexdigest()

    return sorted(sorted(pool_row_ids), key=key)[:k]


# --- files ---

DATASET_COLUMNS = [
    "instance_id", "source_row_id", "post_id",
    "label", "aspect", "input_text", "synthetic",
]


def open_write(path, mode="w"):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if "b" in mode:
        return open(path, mode)
    return open(path, mode, encoding="utf-8", newline="")


def write_text(path, text):
    tmp = path + ".tmp"
    with open_write(tmp) as f:
        f.write(text)
    os.replace(tmp, path)


def dump_json(obj):
    return json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=True) + "\n"


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class Paths:
    def __init__(self, base="artifacts"):
        self.base = base
        self.manifests = os.path.join(base, "manifests")
        self.pools = os.path.join(base, "pools")
        self.datasets = os.path.join(base, "datasets")
        self.results = os.path.join(base, "results")
        self.results_val = os.path.join(base, "results", "val")
        self.results_test = os.path.join(base, "results", "test")
        self.predictions = os.path.join(base, "results", "predictions")
        self.logs = os.path.join(base, "logs")
        self.split_manifest = os.path.join(self.manifests, "split.json")
        self.class_weights_manifest = os.path.join(self.manifests, "class_weights.json")

    def selection_manifest(self, task):
        return os.path.join(self.manifests, f"selection_{task}.json")

    def dataset_csv(self, task, name):
        return os.path.join(self.datasets, task, name)
