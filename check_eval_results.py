import argparse
import glob
import json
import os
import re
from collections import defaultdict

from config.global_name import ROMA_BENCH_DIR, ROMA_BENCH_RESULTS_DIR


# ==========================================================
# Reference list
#
# A list loaded from some previous eval result json (each element has at
# least id / level / qtype). When not empty, only the samples that appear in
# reference_list are counted (matched by id+level; if the same id+level has a
# different qtype in reference_list and in the current data, the two refer to
# different QA data and an error is raised).
# e.g. reference_list = json.load(open("/path/to/some_eval_result.json"))
# ==========================================================

reference_list = None
# reference_list = json.load(open("selected_qa.json"))


def filter_by_reference(data, reference_list):
    """
    When reference_list is not empty, keep only the samples whose (id, level) appears in it;
    raise if the same (id, level) has a different qtype in reference_list and in the current data.
    """
    if not reference_list:
        return data

    reference_qtype = {}
    for ref in reference_list:
        key = (ref.get("id"), ref.get("level"))
        reference_qtype[key] = ref.get("qtype")

    filtered = []
    for item in data:
        key = (item.get("id"), item.get("level"))
        if key not in reference_qtype:
            continue
        ref_qtype = reference_qtype[key]
        cur_qtype = item.get("qtype")
        if ref_qtype != cur_qtype:
            raise Exception(
                f"reference_list mismatch for id={item.get('id')} level={item.get('level')}: "
                f"qtype in reference_list={ref_qtype!r} vs current item={cur_qtype!r}"
            )
        filtered.append(item)
    return filtered


# ==========================================================
# Normalize
# ==========================================================

def normalize(text):
    """Remove spaces and lowercase."""
    return text.replace(" ", "").lower()


def normalize_answer_gt(text):
    """
    Extract the option letter(s) from the GT.

    e.g.
        D. none -> d
        A. xxx  -> a
        AB      -> ab
        C       -> c
    """
    text = text.strip()

    if "." in text:
        text = text.split(".")[0]

    return text.replace(" ", "").lower()


# ==========================================================
# Extract Answer (fallback; our results usually already have answer_final)
# ==========================================================

def extract_answer(answer_llm, gt, question_text):
    """
    Extract the answer from the LLM output, as a fallback when answer_final is missing.

    Supports:
        [A]
        [AB]
        [none]

    and:
        The answer is A.
        answer: B
        answer = C
    """

    text = answer_llm.lower()

    # ======================================================
    # 1. Match [A] / [AB] / [none] first
    # ======================================================

    match = re.search(
        r"\[([a-zA-Z]+)\]",
        text
    )

    if match:

        answer = match.group(1).lower()

        if answer == "none":

            gt_match = re.search(
                r"([a-zA-Z])\s*\.\s*none\b",
                gt.lower()
            )

            if gt_match:
                return gt_match.group(1).lower()

        return answer

    # ======================================================
    # 2. Without [], match:
    #
    # The answer is A
    # answer: B
    # answer = C
    # ======================================================

    patterns = [
        r"answer\s*(?:is|:)\s*([a-zA-Z]+)",
        r"answer\s*=\s*([a-zA-Z]+)",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text
        )

        if match:

            answer = match.group(1).lower()

            if answer == "none":

                gt_match = re.search(
                    r"([a-zA-Z])\s*\.\s*none\b",
                    gt.lower()
                )

                if gt_match:
                    return gt_match.group(1).lower()

            return answer

    return None


# ==========================================================
# Judge Correctness
# ==========================================================

def is_correct(pred, gt, question_text):
    """
    Judge whether the answer is correct.

    Supports:
        single choice
        multiple choice
        ordering
    """

    if pred is None:
        return False

    pred = normalize(pred)
    gt = normalize_answer_gt(gt)

    # ======================================================
    # Ordering
    # ======================================================

    if "order" in question_text.lower():
        return pred == gt

    # ======================================================
    # Single choice
    # ======================================================

    if len(gt) == 1:
        return pred == gt

    # ======================================================
    # Multiple choice
    # Order is ignored
    # ======================================================

    return set(pred) == set(gt)


# ==========================================================
# Get plain-text question
#
# In our results the question is a role/content message dict;
# question_text is used first, and rebuilt from question.content if missing
# ==========================================================

def get_question_text(item):

    question_text = item.get("question_text")

    if question_text:
        return question_text

    question = item.get("question", {})

    parts = []

    for block in question.get("content", []):

        if block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif block.get("type") == "image":
            parts.append("[IMAGE]")
        elif block.get("type") == "audio":
            parts.append("[AUDIO]")
        elif block.get("type") == "tactile":
            parts.append("[TACTILE]")

    return "".join(parts)


# ==========================================================
# Type -> Multiple Categories
# ==========================================================

def get_categories(qtype):
    """
    A type can belong to several categories at the same time.

    Rules:

    Force / Weight:
        force
        heaviest
        lightest

    Inside / Empty:
        inside
        empty

    Texture:
        texture

    Roughness:
        roughness
        roughest
        smoothest

    Hardness:
        hardness
        hardest
        softest

    Material:
        material

    Note:
        No early return.
        Every matching category is added.
    """

    qtype_lower = qtype.lower()

    categories = []

    # ======================================================
    # Force / Weight
    # ======================================================

    if (
        "force" in qtype_lower
        or "heaviest" in qtype_lower
        or "lightest" in qtype_lower
    ):
        categories.append(
            "Force / Weight"
        )

    # ======================================================
    # Inside / Empty
    # ======================================================

    if (
        "inside" in qtype_lower
        or "empty" in qtype_lower
    ):
        categories.append(
            "Inside / Empty"
        )

    # ======================================================
    # Texture
    # ======================================================

    if "texture" in qtype_lower:
        categories.append(
            "Texture"
        )

    # ======================================================
    # Roughness
    # ======================================================

    if (
        "roughness" in qtype_lower
        or "roughest" in qtype_lower
        or "smoothest" in qtype_lower
    ):
        categories.append(
            "Roughness"
        )

    # ======================================================
    # Hardness
    # ======================================================

    if (
        "hardness" in qtype_lower
        or "hardest" in qtype_lower
        or "softest" in qtype_lower
    ):
        categories.append(
            "Hardness"
        )

    # ======================================================
    # Material
    # ======================================================

    if (
        "material" in qtype_lower
    ):
        categories.append(
            "Material"
        )

    # ======================================================
    # Other
    # ======================================================

    if not categories:
        categories.append(
            "Other"
        )

    return categories


def get_effective_level(item):
    """
    Some types are nominally level1 (e.g. the compound_* tasks: a single object,
    but two properties are asked at once, such as compound_hardness_roughness).
    Their qtype spans two categories, so they are counted as level2 instead of
    using the level field stored in the data.
    """
    level = item.get("level", "unknown")
    qtype = item.get("qtype", "") or ""
    if len(get_categories(qtype)) >= 2:
        return "level2"
    return level


# ==========================================================
# Interaction Statistics (only over successful samples,
# i.e. answer_final is not None)
# ==========================================================

NON_GRASP_ACTIONS = ["lift", "squeeze", "shake", "rotate", "collide", "press"]


def is_sample_successful(item):
    """Same as the is_error check in evaluate(): a sample is successful iff answer_final is not None."""
    return item.get("answer_final") is not None


def is_sample_non_error(item):
    """A sample without an error field (or with error=None) is non-error. answer_final does not have
    to be parsable, only the inference itself must not have raised (looser than is_sample_successful)."""
    return item.get("error") is None


def is_sample_any(item):
    """No filtering at all, error samples included. The scene object count only depends on the
    question image path, not on whether the model produced a result, so it can cover every sample."""
    return True


def collect_interaction_stats(data, sample_filter):
    """
    Interaction statistics over the samples where sample_filter(item) is true, grouped by level,
    plus an "ALL" entry over every level.

    Stats of each level:
        num_success   number of counted samples (tasks)
        num_objects   number of grasps, i.e. grasped object instances (one grasp = one object)
        grasp_count   total number of grasps (same value as num_objects, kept separate for clarity)
        action_counts {action_name: count}  total count of each of the 6 non-grasp actions
    """

    def new_stats():
        return {
            "num_success": 0,
            "num_objects": 0,
            "grasp_count": 0,
            "action_counts": defaultdict(int),
        }

    stats_by_level = defaultdict(new_stats)

    for item in data:

        if not sample_filter(item):
            continue

        level = get_effective_level(item)

        for level_key in (level, "ALL"):
            s = stats_by_level[level_key]
            s["num_success"] += 1

            for event in item.get("action_results", []) or []:
                action = event.get("action")
                if action == "grasp":
                    s["grasp_count"] += 1
                    s["num_objects"] += 1
                elif action in NON_GRASP_ACTIONS:
                    s["action_counts"][action] += 1

    return stats_by_level


def print_interaction_stats(data, sample_filter, title):

    stats_by_level = collect_interaction_stats(data, sample_filter)

    print("\n")
    print("=" * 100)
    print(title)
    print("=" * 100)

    if "ALL" not in stats_by_level or stats_by_level["ALL"]["num_success"] == 0:
        print("No matching samples to compute interaction statistics.")
        print("=" * 100)
        return

    levels = [lv for lv in stats_by_level.keys() if lv != "ALL"]

    try:
        levels.sort(key=lambda x: int(re.sub(r"[^0-9]", "", str(x)) or 0))
    except (ValueError, TypeError):
        levels.sort(key=lambda x: str(x))

    for level in levels + ["ALL"]:

        s = stats_by_level[level]
        n = s["num_success"]
        n_obj = s["num_objects"]

        label = "TOTAL (ALL levels)" if level == "ALL" else f"LEVEL {level}"

        print("\n")
        print("-" * 100)
        print(f"{label}  |  samples = {n}  |  grasped objects = {n_obj}")
        print("-" * 100)

        avg_grasp = (s["grasp_count"] / n) if n > 0 else 0.0
        print(f"Avg grasp count per task: {avg_grasp:.3f}")

        print(
            f"{'Action':<15}"
            f"{'Count':>10}"
            f"{'Avg/Task':>15}"
            f"{'Avg/Object':>15}"
        )
        print("-" * 55)

        total_6 = 0
        for action in NON_GRASP_ACTIONS:
            cnt = s["action_counts"].get(action, 0)
            total_6 += cnt
            avg_task = (cnt / n) if n > 0 else 0.0
            avg_obj = (cnt / n_obj) if n_obj > 0 else 0.0
            print(
                f"{action:<15}"
                f"{cnt:>10}"
                f"{avg_task:>15.3f}"
                f"{avg_obj:>15.3f}"
            )

        print("-" * 55)
        avg_task_total = (total_6 / n) if n > 0 else 0.0
        # n_obj is only incremented on "grasp" (see collect_interaction_stats), so this
        # averages over the objects that were actually grasped.
        avg_obj_total = (total_6 / n_obj) if n_obj > 0 else 0.0
        print(
            f"{'TOTAL (6)':<15}"
            f"{total_6:>10}"
            f"{avg_task_total:>15.3f}"
            f"{avg_obj_total:>15.3f}"
        )

        print(f"\nAvg interactions per object (excl. grasp): {avg_obj_total:.3f}")

    print("=" * 100)


# ==========================================================
# Scene Object Count Statistics
#
# The scene folder is the folder of the first image in the question
# (e.g. ROMA-Bench/b1/Initial_180.png -> ROMA-Bench/b1, same as
# load_scene_data in eval_roma_7b.py), and the number of objects in the
# scene is simply len() of the save.json in that folder.
# ==========================================================

BENCH_DIR = ROMA_BENCH_DIR

_scene_object_count_cache = {}


def get_scene_object_count(item):
    """
    Derive the scene folder from the first image path in item["question"]["content"] and use
    the length of its save.json as the number of objects in the scene. A scene is shared by
    many QA, so the result is cached. Returns None if the image path / save.json is not found.
    """
    question = item.get("question", {})
    image_path = None
    for block in question.get("content", []):
        if block.get("type") == "image":
            image_path = block.get("image")
            break

    if not image_path:
        return None

    scene_folder = os.path.join(BENCH_DIR, os.path.dirname(image_path))

    if scene_folder in _scene_object_count_cache:
        return _scene_object_count_cache[scene_folder]

    save_path = os.path.join(scene_folder, "save.json")
    count = None
    try:
        with open(save_path, "r", encoding="utf-8") as f:
            save_data = json.load(f)
        count = len(save_data)
    except Exception:
        count = None

    _scene_object_count_cache[scene_folder] = count
    return count


def print_scene_object_count_stats(data, sample_filter, title):

    level_totals = defaultdict(lambda: {"sum": 0, "count": 0})

    for item in data:

        if not sample_filter(item):
            continue

        count = get_scene_object_count(item)
        if count is None:
            continue

        level = get_effective_level(item)

        for level_key in (level, "ALL"):
            level_totals[level_key]["sum"] += count
            level_totals[level_key]["count"] += 1

    print("\n")
    print("=" * 100)
    print(title)
    print("=" * 100)

    if "ALL" not in level_totals or level_totals["ALL"]["count"] == 0:
        print("No matching samples with a resolvable scene to compute this statistic.")
        print("=" * 100)
        return

    levels = [lv for lv in level_totals.keys() if lv != "ALL"]

    try:
        levels.sort(key=lambda x: int(re.sub(r"[^0-9]", "", str(x)) or 0))
    except (ValueError, TypeError):
        levels.sort(key=lambda x: str(x))

    print(
        f"{'Level':<20}"
        f"{'Samples':>12}"
        f"{'Avg objects/scene':>22}"
    )
    print("-" * 60)

    for level in levels + ["ALL"]:
        s = level_totals[level]
        avg = (s["sum"] / s["count"]) if s["count"] > 0 else 0.0
        label = "TOTAL (ALL levels)" if level == "ALL" else f"LEVEL {level}"
        print(
            f"{label:<20}"
            f"{s['count']:>12}"
            f"{avg:>22.3f}"
        )

    print("=" * 100)


# ==========================================================
# Final Summary Table (Level x Category, 1 decimal place)
# ==========================================================

# H(Hardness) / R(Roughness) / T(Texture) / I(Inside) / M(Material) / W(Weight),
# in that fixed order. "Inside"/"Weight" map to this file's own Category names
# ("Inside / Empty" / "Force / Weight").
SUMMARY_CATEGORY_COLUMNS = [
    ("H", "Hardness"),
    ("R", "Roughness"),
    ("T", "Texture"),
    ("I", "Inside / Empty"),
    ("M", "Material"),
    ("W", "Force / Weight"),
]

SUMMARY_LEVELS = ["level1", "level2", "level3"]


def _fmt_pct(correct, total, width):
    text = "N/A" if total == 0 else f"{correct / total * 100:.1f}%"
    return f"{text:>{width}}"


def print_final_summary_table(level_stats, level_overall_stats):
    """
    Print one accuracy table of level1/level2/level3 (plus an ALL row) x category.
    Categories are shown in the fixed H/R/T/I/M/W order; percentages have one decimal place.
    """

    col_w = 9
    overall_w = 10
    level_w = 10

    print("\n")
    print("=" * 100)
    print("Final Summary: Accuracy by Level x Category")
    print("=" * 100)

    header = f"{'Level':<{level_w}}"
    for abbr, _ in SUMMARY_CATEGORY_COLUMNS:
        header += f"{abbr:>{col_w}}"
    header += f"{'Overall':>{overall_w}}"
    print(header)
    print("-" * len(header))

    col_totals = [[0, 0] for _ in SUMMARY_CATEGORY_COLUMNS]
    overall_correct_all = 0
    overall_total_all = 0

    for level in SUMMARY_LEVELS:

        row = f"{level:<{level_w}}"

        for col_idx, (_, category) in enumerate(SUMMARY_CATEGORY_COLUMNS):
            qtype_stats = level_stats.get(level, {}).get(category, {})
            c = sum(s["correct"] for s in qtype_stats.values())
            t = sum(s["total"] for s in qtype_stats.values())
            col_totals[col_idx][0] += c
            col_totals[col_idx][1] += t
            row += _fmt_pct(c, t, col_w)

        overall = level_overall_stats.get(level, {})
        oc, ot = overall.get("correct", 0), overall.get("total", 0)
        overall_correct_all += oc
        overall_total_all += ot
        row += _fmt_pct(oc, ot, overall_w)

        print(row)

    row = f"{'ALL':<{level_w}}"
    for c, t in col_totals:
        row += _fmt_pct(c, t, col_w)
    row += _fmt_pct(overall_correct_all, overall_total_all, overall_w)
    print(row)

    print("=" * 100)


# ==========================================================
# Evaluate
# ==========================================================

def evaluate(data):

    # ======================================================
    # Overall
    # ======================================================

    overall_correct = 0
    overall_total = len(data)

    # ======================================================
    # Error samples (pred is None -> generation / parsing failed)
    # ======================================================

    error_ids = []

    # ======================================================
    # Level + Category + Type
    #
    # level_stats[level][category][type]
    # ======================================================

    level_stats = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: {
                    "correct": 0,
                    "total": 0
                }
            )
        )
    )

    # ======================================================
    # Overall Type
    # ======================================================

    type_stats = defaultdict(
        lambda: {
            "correct": 0,
            "total": 0
        }
    )

    # ======================================================
    # Overall Level
    # ======================================================

    level_overall_stats = defaultdict(
        lambda: {
            "correct": 0,
            "total": 0
        }
    )

    # ======================================================
    # Evaluate every sample
    # ======================================================

    for item in data:

        question_text = get_question_text(item)
        answer_llm = item.get("answer_llm", "")
        gt = item["answer_gt"]

        qtype = item.get(
            "qtype",
            "unknown"
        )

        level = get_effective_level(item)

        # --------------------------------------------------
        # A type can have several categories
        # --------------------------------------------------

        categories = get_categories(qtype)

        # --------------------------------------------------
        # Extract answer
        #
        # Our results already contain answer_final;
        # the regex fallback is only for when it is missing
        # --------------------------------------------------

        pred = item.get("answer_final")

        # if not pred:
        #     pred = extract_answer(
        #         answer_llm,
        #         gt,
        #         question_text
        #     )

        # --------------------------------------------------
        # pred is still None
        # -> the model gave no parsable answer, so something
        #    went wrong (generation exception / truncation /
        #    parsing failure ...). It is forced to be wrong,
        #    without falling back to the correct field.
        # --------------------------------------------------

        is_error = pred is None

        if is_error:

            ok = False

            error_ids.append(
                f"{item.get('level', 'unknown')}_{item.get('id', 'unknown')}"
            )

        else:

            # ----------------------------------------------
            # Judge
            # ----------------------------------------------

            # ok = item.get("correct")

            # if ok is None:
            #     ok = is_correct(
            #         pred,
            #         gt,
            #         question_text
            #     )
            pred = normalize(pred)
            gt = normalize_answer_gt(gt)
            ok = is_correct(
                pred,
                gt,
                question_text
            )

        # ==================================================
        # Overall
        # ==================================================

        if ok:
            overall_correct += 1

        # ==================================================
        # Overall Type
        #
        # Each sample is counted once per type
        # ==================================================

        type_stats[qtype]["total"] += 1

        if ok:
            type_stats[qtype]["correct"] += 1

        # ==================================================
        # Overall Level
        #
        # Each sample is counted once per level
        # ==================================================

        level_overall_stats[level]["total"] += 1

        if ok:
            level_overall_stats[level]["correct"] += 1

        # ==================================================
        # Level + Category + Type
        #
        # Note:
        # a sample can be counted in several categories here
        # ==================================================

        for category in categories:

            level_stats[level][category][qtype]["total"] += 1

            if ok:
                level_stats[level][category][qtype]["correct"] += 1

        # ==================================================
        # Print wrong samples
        # ==================================================

        if not ok:

            qid = item.get(
                "id",
                "unknown"
            )

            # last_line = (
            #     answer_llm.strip().splitlines()[-1]
            #     if answer_llm.strip()
            #     else ""
            # )

            n_actions = len(
                item.get("action_results", [])
            )

            mark = "‼ ERROR (no pred)" if is_error else "✗"

            print(
                f"{str(qid):>4} | "
                f"Level={str(level):<7} | "
                f"Category={','.join(categories):<35} | "
                f"{qtype:<35} | "
                f"pred={str(pred):<6} | "
                f"gt={gt:<8} | "
                f"actions={n_actions:<3} | {mark}"
            )

            # print(
            #     f"Last line : {last_line}"
            # )

            print(
                f"Question  : {question_text}"
            )

            print(
                f"LLM Answer: {answer_llm}"
            )

            print(
                "-" * 150
            )

    # ======================================================
    # Overall
    # ======================================================

    print("\n")
    print("=" * 100)
    print("Overall")
    print("=" * 100)

    print(
        f"Correct : {overall_correct}"
    )

    print(
        f"Total   : {overall_total}"
    )

    if overall_total > 0:

        print(
            f"Accuracy: "
            f"{overall_correct / overall_total:.2%}"
        )

    else:

        print(
            "Accuracy: N/A"
        )

    print(
        f"Errors  : {len(error_ids)} "
        f"(pred=None, forced wrong) -> ids: {error_ids}"
    )

    # ======================================================
    # Overall Accuracy by Level
    # ======================================================

    print("\n")
    print("=" * 100)
    print("Overall Accuracy by Level")
    print("=" * 100)

    print(
        f"{'Level':<15}"
        f"{'Correct':>12}"
        f"{'Total':>12}"
        f"{'Accuracy':>15}"
    )

    print("-" * 60)

    levels = list(
        level_overall_stats.keys()
    )

    try:

        levels.sort(
            key=lambda x: int(
                re.sub(r"[^0-9]", "", str(x)) or 0
            )
        )

    except (ValueError, TypeError):

        levels.sort(
            key=lambda x: str(x)
        )

    for level in levels:

        stats = level_overall_stats[level]

        c = stats["correct"]
        t = stats["total"]

        acc = (
            c / t
            if t > 0
            else 0
        )

        print(
            f"{str(level):<15}"
            f"{c:>12}"
            f"{t:>12}"
            f"{acc:>14.2%}"
        )

    print("=" * 100)

    # ======================================================
    # Category order
    # ======================================================

    category_order = [
        "Force / Weight",
        "Inside / Empty",
        "Texture",
        "Roughness",
        "Hardness",
        "Material",
        "Other"
    ]

    # ======================================================
    # Detailed Level Statistics
    # ======================================================

    print("\n")
    print("=" * 130)
    print("Detailed Accuracy by Level / Category / Type")
    print("=" * 130)

    for level in levels:

        # ==================================================
        # Level Header
        # ==================================================

        print("\n")
        print("#" * 130)
        print(
            f"LEVEL {level}"
        )
        print("#" * 130)

        level_total = (
            level_overall_stats[level]["total"]
        )

        level_correct = (
            level_overall_stats[level]["correct"]
        )

        level_acc = (
            level_correct / level_total
            if level_total > 0
            else 0
        )

        print(
            f"\nLevel {level} Overall: "
            f"{level_correct}/{level_total} "
            f"({level_acc:.2%})"
        )

        # ==================================================
        # Each Category
        # ==================================================

        for category in category_order:

            if category not in level_stats[level]:
                continue

            category_types = (
                level_stats[level][category]
            )

            # ------------------------------------------------
            # Category total
            # ------------------------------------------------

            category_correct = 0
            category_total = 0

            for stats in category_types.values():

                category_correct += (
                    stats["correct"]
                )

                category_total += (
                    stats["total"]
                )

            category_acc = (
                category_correct / category_total
                if category_total > 0
                else 0
            )

            # ------------------------------------------------
            # Category Header
            # ------------------------------------------------

            print("\n")
            print("-" * 115)

            print(
                f"Category: {category}"
            )

            print("-" * 115)

            # ------------------------------------------------
            # Table Header
            # ------------------------------------------------

            print(
                f"{'Type':<70}"
                f"{'Correct':>10}"
                f"{'Total':>10}"
                f"{'Accuracy':>12}"
            )

            print("-" * 115)

            # ------------------------------------------------
            # Sort types by Accuracy
            # ------------------------------------------------

            type_results = []

            for qtype, stats in category_types.items():

                c = stats["correct"]
                t = stats["total"]

                acc = (
                    c / t
                    if t > 0
                    else 0
                )

                type_results.append(
                    (
                        qtype,
                        c,
                        t,
                        acc
                    )
                )

            type_results.sort(
                key=lambda x: x[3]
            )

            # ------------------------------------------------
            # Print each type
            # ------------------------------------------------

            for qtype, c, t, acc in type_results:

                print(
                    f"{qtype:<70}"
                    f"{c:>10}"
                    f"{t:>10}"
                    f"{acc:>11.2%}"
                )

            # ------------------------------------------------
            # Category Overall
            # ------------------------------------------------

            print("-" * 115)

            print(
                f"{'CATEGORY TOTAL':<70}"
                f"{category_correct:>10}"
                f"{category_total:>10}"
                f"{category_acc:>11.2%}"
            )

            print("-" * 115)

    # ======================================================
    # Full Type Accuracy
    # ======================================================

    print("\n")
    print("=" * 100)
    print("Full Type Accuracy")
    print("=" * 100)

    print(
        f"{'Type':<70}"
        f"{'Correct':>10}"
        f"{'Total':>10}"
        f"{'Accuracy':>12}"
    )

    print("-" * 100)

    full_type_results = []

    for qtype, stats in type_stats.items():

        c = stats["correct"]
        t = stats["total"]

        acc = (
            c / t
            if t > 0
            else 0
        )

        full_type_results.append(
            (
                qtype,
                c,
                t,
                acc
            )
        )

    full_type_results.sort(
        key=lambda x: x[3]
    )

    for qtype, c, t, acc in full_type_results:

        print(
            f"{qtype:<70}"
            f"{c:>10}"
            f"{t:>10}"
            f"{acc:>11.2%}"
        )

    print("=" * 100)

    # ======================================================
    # Interaction Statistics
    # ======================================================

    print_interaction_stats(
        data, is_sample_successful,
        "Interaction Statistics (successful samples only)"
    )
    print_interaction_stats(
        data, is_sample_non_error,
        "Interaction Statistics (non-error samples)"
    )

    # ======================================================
    # Scene Object Count Statistics
    # ======================================================

    print_scene_object_count_stats(
        data, is_sample_successful,
        "Scene Object Count Statistics (successful samples only)"
    )
    print_scene_object_count_stats(
        data, is_sample_non_error,
        "Scene Object Count Statistics (non-error samples)"
    )
    print_scene_object_count_stats(
        data, is_sample_any,
        "Scene Object Count Statistics (all samples, including errors)"
    )

    # ======================================================
    # Final Summary Table
    # ======================================================

    print_final_summary_table(level_stats, level_overall_stats)


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Compute ROMA-Bench accuracy from the eval_roma_7b.py results")
    parser.add_argument("--results", default=os.path.join(ROMA_BENCH_RESULTS_DIR, "eval_roma_bench_*.json"),
                        help="Glob pattern of the per-shard result json files.")
    parser.add_argument("--bench_dir", default=ROMA_BENCH_DIR, help="ROMA-Bench scene directory (for the scene object counts).")
    args = parser.parse_args()
    BENCH_DIR = args.bench_dir

    # ======================================================
    # Load JSON (every shard matched by --results)
    # ======================================================

    json_paths = sorted(glob.glob(args.results))
    if not json_paths:
        raise SystemExit(f"No result files match {args.results}")

    data = []
    for json_path in json_paths:
        with open(json_path, "r", encoding="utf-8") as f:
            data.extend(json.load(f))

    print(f"Loaded {len(data)} samples from {len(json_paths)} file(s).")

    # ======================================================
    # Select QA
    # ======================================================

    target_qa = []

    for item in data:

        # Filter here if needed, e.g.
        #
        # if item.get("level") == "level1":
        #     target_qa.append(item)
        #
        # or:
        #
        # if "empty" in item["qtype"]:
        #     target_qa.append(item)

        target_qa.append(item)

    target_qa = filter_by_reference(target_qa, reference_list)
    if reference_list:
        print(f"[REFERENCE] {len(target_qa)} samples remain after reference_list filtering.")

    print(
        f"Evaluating "
        f"{len(target_qa)} samples."
    )

    # ======================================================
    # Evaluate
    # ======================================================

    evaluate(target_qa)
