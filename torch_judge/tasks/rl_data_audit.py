"""Audit the row-level integrity of an offline RL dataset."""

from ._interview import interview
from ._rl_extension import DATASETS, case, code_source, task

TASK = task(
    "RL Dataset Integrity Audit", "Medium", "rl_data_audit",
    """Implement `rl_data_audit(train, eval_rows) -> dict` for two lists of record dictionaries.

Each record must have nonempty string `sample_id`, `prompt`, and `response` fields. Return exactly four keys: `missing_train` and `missing_eval` are ascending zero-based row indices with any missing/empty required field; `duplicate_ids` is a sorted list of IDs appearing more than once within either split; `leakage_ids` is a sorted list of IDs present in both splits. Count IDs even on rows with a missing prompt or response, but ignore missing/empty IDs in duplicate and leakage checks. Never mutate input records. This audit detects identity leakage, not paraphrase or semantic similarity.""",
    ["verl_dataproto_filter_chunk"],
    ("Which fields make a record usable? Should a row with a valid ID but no response still participate in leakage checks?",
     "Scan each split once for invalid row indices and ID counts. Build the cross-split intersection from nonempty IDs; sort the final ID lists for deterministic output."),
    [code_source(DATASETS, "Split-wise unique sample IDs are the basis for exact-ID overlap checks.", "Adds per-split counting and required-field validation to the split-wise unique operation; plain dictionaries replace Arrow datasets.")],
    [
        case("Visible missing, duplicate, and leakage", """
train=[{'sample_id':'a','prompt':'p','response':'r'},{'sample_id':'a','prompt':'p','response':''},{'sample_id':'b','prompt':'p','response':'r'}]
ev=[{'sample_id':'b','prompt':'p','response':'r'},{'sample_id':'c','prompt':'p','response':'r'}]
assert {fn}(train,ev)=={'missing_train':[1],'missing_eval':[],'duplicate_ids':['a'],'leakage_ids':['b']}
""", "protocol.validation"),
        case("Independent count oracle and no mutation", """
from collections import Counter
import copy, random
random.seed(41)
for _ in range(5):
    rows=[[{'sample_id':random.choice(['a','b','c','d','']), 'prompt':random.choice(['q','']), 'response':random.choice(['r',''])} for _ in range(11)] for _ in range(2)]
    before=copy.deepcopy(rows)
    valid=lambda x:isinstance(x,str) and bool(x)
    ids=[[r['sample_id'] for r in split if valid(r['sample_id'])] for split in rows]
    counts=[Counter(part) for part in ids]
    expected={'missing_train':[i for i,r in enumerate(rows[0]) if not all(valid(r[k]) for k in ('sample_id','prompt','response'))], 'missing_eval':[i for i,r in enumerate(rows[1]) if not all(valid(r[k]) for k in ('sample_id','prompt','response'))], 'duplicate_ids':sorted(k for k in counts[0].keys() | counts[1].keys() if counts[0][k]>1 or counts[1][k]>1), 'leakage_ids':sorted(set(ids[0]) & set(ids[1]))}
    assert {fn}(*rows)==expected
    assert rows==before
""", "protocol.validation", unshown=True),
        case("Absent keys, empty IDs, and within-split duplicates", """
train=[{}, {'sample_id':'', 'prompt':'p','response':'r'}, {'sample_id':'x','prompt':'p','response':'r'}, {'sample_id':'x','prompt':'p','response':'r'}]
ev=[{'sample_id':'', 'prompt':'p','response':'r'}, {'sample_id':'z','prompt':'p','response':'r'}]
assert {fn}(train,ev)=={'missing_train':[0,1],'missing_eval':[0],'duplicate_ids':['x'],'leakage_ids':[]}
""", "edge.empty_or_boundary", unshown=True),
    ],
    '''from collections import Counter

def rl_data_audit(train, eval_rows):
    def inspect(rows):
        missing = []
        ids = []
        for index, row in enumerate(rows):
            if any(not isinstance(row.get(key), str) or not row.get(key) for key in ("sample_id", "prompt", "response")):
                missing.append(index)
            sample_id = row.get("sample_id")
            if isinstance(sample_id, str) and sample_id:
                ids.append(sample_id)
        return missing, Counter(ids)

    missing_train, train_ids = inspect(train)
    missing_eval, eval_ids = inspect(eval_rows)
    all_ids = train_ids.keys() | eval_ids.keys()
    return {
        "missing_train": missing_train,
        "missing_eval": missing_eval,
        "duplicate_ids": sorted(key for key in all_ids if train_ids[key] > 1 or eval_ids[key] > 1),
        "leakage_ids": sorted(train_ids.keys() & eval_ids.keys()),
    }
''',
    model_connections=["Hugging Face Datasets' DatasetDict.unique yields unique IDs per split; this task additionally counts duplicates and intersects the train/eval ID sets."],
    pros=["Exact-ID checks quickly expose duplicate rows and train/eval leakage before RL training."],
    cons=["Paraphrases or semantically identical prompts with different IDs are not detected."],
    interview_questions=interview(
        concept=[
            'What integrity problems in an offline RL or SFT dataset would you check before training?',
            'Why does train and eval ID overlap matter?',
        ],
        deep_dive=[
            'How do you report missing fields, duplicates and leakage without mutating input rows?',
            'Why ignore empty IDs in duplicate and leakage checks?',
            'What data structures keep this linear in the number of rows?',
        ],
        tradeoffs=[
            'ID-based leakage detection misses paraphrases. What would you add to catch near-duplicates?',
            'What is the cost of false positives in a leakage check?',
        ],
    ),
)
