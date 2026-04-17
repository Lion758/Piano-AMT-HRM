"""
MIDI Analysis Module

A comprehensive toolkit for analyzing MIDI performances with educational focus.
Provides tools for parsing, time alignment, phrase segmentation, error analysis,
and generating GPT-ready summaries for piano pedagogy.
"""

from importlib import import_module

__version__ = "0.1.0"

__all__ = [
    "MIDIParser",
    "TimeAlignment",
    "PaperBestTimeAlignment",
    "align_midi_files_paper_best",
    "PhraseSegmentation",
    "ErrorAnalysis",
    "JSONSummarization",
    "MIDIAnalyzer",
    "quick_analyze",
    "compare_performance",
    "GPTTutor",
    "create_tutor_feedback",
]

package_info = {
    "name": "midi-analysis-module",
    "version": __version__,
    "description": "Educational MIDI analysis tool for piano pedagogy",
    "modules": [
        "midi_parser",
        "time_alignment",
        "paper_time_alignment",
        "phrase_segmentation",
        "error_analysis",
        "json_summarization",
        "analyzer",
        "gpt_tutor",
    ],
}

_LAZY_EXPORTS = {
    "MIDIParser": (".midi_parser", "MIDIParser"),
    "TimeAlignment": (".time_alignment", "TimeAlignment"),
    "PaperBestTimeAlignment": (".paper_time_alignment", "PaperBestTimeAlignment"),
    "align_midi_files_paper_best": (".paper_time_alignment", "align_midi_files_paper_best"),
    "PhraseSegmentation": (".phrase_segmentation", "PhraseSegmentation"),
    "ErrorAnalysis": (".error_analysis", "ErrorAnalysis"),
    "JSONSummarization": (".json_summarization", "JSONSummarization"),
    "MIDIAnalyzer": (".analyzer", "MIDIAnalyzer"),
    "quick_analyze": (".analyzer", "quick_analyze"),
    "compare_performance": (".analyzer", "compare_performance"),
    "GPTTutor": (".gpt_tutor", "GPTTutor"),
    "create_tutor_feedback": (".gpt_tutor", "create_tutor_feedback"),
}


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name, __name__)
    return getattr(module, attr_name)


def __dir__():
    return sorted(set(globals()) | set(__all__))
