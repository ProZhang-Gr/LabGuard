"""LabGuard-Grounder: natural-language laboratory rules -> LabGuard-IR (Sec. 3.4).

Heavy backends (torch / transformers / peft / openai) are imported lazily by
the individual grounders.
"""

from labguard.grounder.base import EMPTY_IR, Grounder, OracleGrounder, clean_ir
from labguard.grounder.constrained import IRGrammar, IRLogitsProcessor
from labguard.grounder.factory import build_grounder, parse_spec
from labguard.grounder.hybrid import HybridGrounder
from labguard.grounder.prompts import chat_messages, sft_prompt, sft_target, system_prompt
from labguard.grounder.regex_grounder import RegexGrounder, extract_substances
from labguard.grounder.retrieval import KNNGrounder

__all__ = [
    "EMPTY_IR", "Grounder", "OracleGrounder", "clean_ir", "IRGrammar", "IRLogitsProcessor", "build_grounder",
    "parse_spec", "HybridGrounder", "chat_messages", "sft_prompt", "sft_target", "system_prompt", "RegexGrounder",
    "extract_substances", "KNNGrounder",
]
