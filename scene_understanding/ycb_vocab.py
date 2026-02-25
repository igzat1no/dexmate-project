"""
YCB dataset vocabulary + common tabletop objects for open-vocabulary detection.

The vocabulary is used as Grounding DINO text prompts.  It covers:
  1. YCB-M objects (the 20 objects in the dataset) — always included.
  2. Additional YCB objects that are visually distinct.
  3. Extra tabletop objects for generalization (no generic hypernyms).

Design notes:
  - Removed generic hypernyms ("bottle", "can", "box", "ball") that
    compete with specific YCB names in Grounding DINO.
  - Removed near-duplicates ("cups" vs "mug", "windex bottle" vs
    "mustard bottle") that cause cross-class confusion.
  - Recommend box_threshold >= 0.3 when using this vocab.
"""

# ── YCB-M core (20 objects that appear in the YCB-M dataset) ───────────
YCB_M_OBJECTS = {
    "002": "master chef can",
    "003": "cracker box",
    "004": "sugar box",
    "005": "tomato soup can",
    "006": "mustard bottle",
    "007": "tuna fish can",
    "008": "pudding box",
    "009": "gelatin box",
    "010": "potted meat can",
    "011": "banana",
    "019": "pitcher",
    "021": "bleach cleanser",
    "024": "bowl",
    "025": "mug",
    "035": "power drill",
    "036": "wood block",
    "037": "scissors",
    "040": "marker",
    "051": "large clamp",
    "052": "extra large clamp",
    "061": "foam brick",
}

# ── Additional YCB objects (visually distinct from core set) ───────────
YCB_EXTENDED = {
    "013": "apple",
    "017": "orange",
    "026": "sponge",
    "027": "skillet",
    "029": "plate",
    "030": "fork",
    "031": "spoon",
    "032": "knife",
    "033": "spatula",
    "038": "flat screwdriver",
    "042": "adjustable wrench",
    "043": "phillips screwdriver",
    "044": "medium clamp",
    "048": "hammer",
    "053": "mini soccer ball",
    "055": "baseball",
    "056": "tennis ball",
    "062": "dice",
    "077": "rubiks cube",
}

# ── Extra tabletop objects (no generic hypernyms) ──────────────────────
EXTRA_TABLETOP_OBJECTS = [

    "jar",
    "pen",
    "pencil",
    "stapler",
    "remote control",
    "phone",
    "book",
    "notebook",
    "flashlight",
    "key",
    "watch",
]

# Merged full ID→name mapping
YCB_ID_TO_NAME = {**YCB_M_OBJECTS, **YCB_EXTENDED}

# Combined vocabulary for Grounding DINO prompts (deduplicated, order-preserved)
YCB_VOCAB = list(dict.fromkeys(
    list(YCB_M_OBJECTS.values())
    + list(YCB_EXTENDED.values())
    + EXTRA_TABLETOP_OBJECTS
))

# Reverse mapping: natural name → YCB ID (only for YCB objects)
YCB_NAME_TO_ID = {v: k for k, v in YCB_ID_TO_NAME.items()}

# Original dataset class names with numeric prefix (YCB-M convention)
YCB_FULL_CLASS_NAMES = {
    "002": "002_master_chef_can",
    "003": "003_cracker_box",
    "004": "004_sugar_box",
    "005": "005_tomato_soup_can",
    "006": "006_mustard_bottle",
    "007": "007_tuna_fish_can",
    "008": "008_pudding_box",
    "009": "009_gelatin_box",
    "010": "010_potted_meat_can",
    "011": "011_banana",
    "019": "019_pitcher_base",
    "021": "021_bleach_cleanser",
    "024": "024_bowl",
    "025": "025_mug",
    "035": "035_power_drill",
    "036": "036_wood_block",
    "037": "037_scissors",
    "040": "040_large_marker",
    "051": "051_large_clamp",
    "052": "052_extra_large_clamp",
    "061": "061_foam_brick",
}
