"""Central configuration: paths, seeds, target names and domain constants."""
from pathlib import Path

RANDOM_STATE = 42
TEST_SIZE = 0.2

# ── Regression ────────────────────────────────────────────────────────────────
REG_DATA_PATH = Path("data/train_data.csv")
REG_SAVE_DIR = Path("saved_models")
REG_TARGET = "RecommendationCount"

# ── Classification ────────────────────────────────────────────────────────────
CLS_DATA_PATH = Path("data/train_data2.csv")
CLS_SAVE_DIR = Path("saved_models_cls")
CLS_TARGET = "GamePopularity"

# ── Preprocessing ─────────────────────────────────────────────────────────────
# Raw identifier / free-text columns removed once their features are extracted.
COLUMNS_TO_DROP = [
    "QueryID", "ResponseID", "QueryName", "ResponseName",
    "PCRecReqsText", "LinuxRecReqsText", "MacRecReqsText",
    "PCMinReqsText", "LinuxMinReqsText", "MacMinReqsText",
    "SupportedLanguages", "ReleaseDate",
]

# Columns converted to 0/1 depending on whether they hold any content.
BINARY_TEXT_COLUMNS = [
    "SupportURL", "SupportEmail", "Website", "ExtUserAcctNotice", "DRMNotice",
    "Background", "HeaderImage", "LegalNotice", "IsFree", "FreeVerAvail",
    "PlatformMac", "PlatformLinux", "PlatformWindows",
    "GenreIsMassivelyMultiplayer", "GenreIsRacing", "GenreIsSports",
    "GenreIsFreeToPlay", "GenreIsEarlyAccess", "GenreIsSimulation", "GenreIsRPG",
    "GenreIsStrategy", "GenreIsCasual", "GenreIsAdventure", "GenreIsAction",
    "GenreIsIndie", "GenreIsNonGame",
    "CategoryMultiplayer", "CategoryCoop", "CategoryMMO",
]

TEXT_LENGTH_COLUMNS = ["DetailedDescrip", "AboutText", "ShortDescrip"]

POSITIVE_WORDS = ["great", "amazing", "fun", "recommend", "excellent",
                  "love", "best", "fantastic", "perfect", "good"]
NEGATIVE_WORDS = ["bad", "terrible", "boring", "waste", "buggy",
                  "broken", "awful", "worst", "crash", "disappointment"]

COMMON_LANGUAGES = [
    "English", "French", "German", "Italian", "Spanish", "Korean", "Japanese",
    "Russian", "Turkish", "Thai", "Portuguese", "Polish", "Dutch", "Arabic",
    "Simplified Chinese", "Traditional Chinese", "Czech", "Hungarian", "Romanian",
]
ASIAN_LANGUAGES = ["Korean", "Japanese", "Simplified Chinese", "Traditional Chinese"]

# Minimum-requirement text columns parsed into numeric hardware features.
REQUIREMENT_SOURCES = {
    "Linux": "LinuxMinReqsText",
    "Mac": "MacMinReqsText",
    "PC": "PCMinReqsText",
}
REQUIREMENT_FIELDS = ["RAM_GB", "Storage_GB", "CPU_GHz", "OpenGL"]
