"""
gpt_tutor.py

GPT tutor layer for turning MIDI analysis summaries into interactive, constructive feedback.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from openai import (
    APIConnectionError,
    APITimeoutError,
    APIStatusError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)


SummarySource = Union[str, Path, Dict[str, Any]]


DEFAULT_SYSTEM_PROMPT = (
    "You are a supportive piano tutor. Use only the provided analysis data. "
    "Give specific, actionable feedback with this structure: "
    "1) overall assessment 2) strengths 3) improvement areas 4) concrete practice plan. "
    "When possible, mention exact metric names and values from the analysis."
)


class GPTTutor:
    """Interactive tutor wrapper around the OpenAI Responses API."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5-mini")
        try:
            self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize OpenAI client. Set OPENAI_API_KEY or pass api_key explicitly."
            ) from exc
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.previous_response_id: Optional[str] = None
        self.summary_data: Dict[str, Any] = {}
        self.summary_path: Optional[str] = None

    def start_session(
        self,
        summary: SummarySource,
        user_prompt: str = "",
        student_question: str = "",
        max_output_tokens: int = 900,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Start a new tutoring session from gpt_summary data."""
        self.summary_data, self.summary_path = self._load_summary(summary)

        instructions = self._build_instructions(user_prompt)
        response = self._safe_create_response(
            model=self.model,
            instructions=instructions,
            input=self._build_initial_input(student_question),
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )

        self.previous_response_id = response.id
        return {
            "response_id": response.id,
            "model": getattr(response, "model", self.model),
            "text": self._extract_output_text(response),
        }

    def ask(
        self,
        user_question: str,
        max_output_tokens: int = 700,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Continue an active tutoring session with a follow-up question."""
        if not self.previous_response_id:
            raise ValueError("No active tutor session. Call start_session() first.")

        response = self._safe_create_response(
            model=self.model,
            previous_response_id=self.previous_response_id,
            input=user_question,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )

        self.previous_response_id = response.id
        return {
            "response_id": response.id,
            "model": getattr(response, "model", self.model),
            "text": self._extract_output_text(response),
        }

    def save_state(self, state_path: Union[str, Path]) -> Dict[str, Any]:
        """Persist session state so follow-up Q&A can resume later."""
        payload = {
            "model": self.model,
            "previous_response_id": self.previous_response_id,
            "summary_path": self.summary_path,
            "system_prompt": self.system_prompt,
        }
        path = Path(state_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    def load_state(self, state_path: Union[str, Path]) -> Dict[str, Any]:
        """Load previously saved tutor state."""
        path = Path(state_path)
        if not path.exists():
            raise FileNotFoundError(f"State file not found: {path}")

        payload = json.loads(path.read_text(encoding="utf-8"))
        self.model = payload.get("model", self.model)
        self.previous_response_id = payload.get("previous_response_id")
        self.summary_path = payload.get("summary_path")
        self.system_prompt = payload.get("system_prompt", self.system_prompt)

        if self.summary_path:
            summary_path = Path(self.summary_path)
            if summary_path.exists():
                self.summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
        return payload

    def _load_summary(self, summary: SummarySource) -> Tuple[Dict[str, Any], Optional[str]]:
        if isinstance(summary, dict):
            return summary, None

        path = Path(summary)
        if not path.exists():
            raise FileNotFoundError(f"Summary file not found: {path}")
        return json.loads(path.read_text(encoding="utf-8")), str(path)

    def _build_instructions(self, user_prompt: str) -> str:
        context = self.summary_data.get("gpt_prompt_context", {})
        instruction_context = context.get("instruction_context", {})
        response_format = context.get("response_formatting", {})

        role = instruction_context.get("role", "You are an experienced piano teacher.")
        tone = instruction_context.get("tone", "Constructive and specific.")
        feedback_format = instruction_context.get(
            "format",
            "1. Overall assessment 2. Strengths 3. Areas for improvement 4. Practice suggestions",
        )
        detail_level = instruction_context.get("detail_level", "Be specific with examples.")
        max_length = response_format.get("max_length", "500-700 words")

        parts = [
            self.system_prompt,
            role,
            f"Tone: {tone}",
            f"Response format: {feedback_format}",
            f"Detail level: {detail_level}",
            f"Target length: {max_length}",
        ]
        if user_prompt.strip():
            parts.append(f"Additional user prompt: {user_prompt.strip()}")
        return "\n".join(parts)

    def _build_initial_input(self, student_question: str) -> str:
        question = student_question.strip() or "Please provide full constructive feedback for this performance."
        payload = {
            "student_question": question,
            "analysis_summary": self.summary_data,
        }
        return json.dumps(payload, ensure_ascii=False)

    def _extract_output_text(self, response: Any) -> str:
        text = getattr(response, "output_text", None)
        if text:
            return text.strip()

        outputs = getattr(response, "output", []) or []
        chunks = []
        for item in outputs:
            for content in getattr(item, "content", []) or []:
                value = getattr(content, "text", None)
                if value:
                    chunks.append(value)
        return "\n".join(chunks).strip()

    def _safe_create_response(self, **kwargs: Any) -> Any:
        """
        Call Responses API with compatibility fallback for models that
        do not accept temperature.
        """
        clean_kwargs = dict(kwargs)
        if clean_kwargs.get("temperature", None) is None:
            clean_kwargs.pop("temperature", None)

        try:
            return self.client.responses.create(**clean_kwargs)
        except BadRequestError as exc:
            msg = str(exc)
            if "Unsupported parameter: 'temperature'" in msg and "temperature" in clean_kwargs:
                clean_kwargs.pop("temperature", None)
                return self.client.responses.create(**clean_kwargs)
            raise


def create_tutor_feedback(
    summary_path: Union[str, Path],
    user_prompt: str = "",
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """One-shot helper for generating initial tutor feedback."""
    tutor = GPTTutor(model=model)
    return tutor.start_session(summary=summary_path, user_prompt=user_prompt)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive GPT tutor for MIDI analysis summaries.")
    parser.add_argument(
        "--summary",
        default="analysis_results/gpt_summary.json",
        help="Path to gpt_summary.json",
    )
    parser.add_argument(
        "--state",
        default="analysis_results/tutor_session.json",
        help="Path to save/load tutor session state",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
        help="Model name for Responses API",
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="Additional prompt to steer tutor behavior",
    )
    parser.add_argument(
        "--question",
        default="",
        help="Initial question (new session) or follow-up question (--resume)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from --state instead of starting a new session",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Open an interactive Q&A loop after first response",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=900,
        help="Maximum output tokens per response",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional temperature. Omit for model default.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    try:
        tutor = GPTTutor(model=args.model)

        if args.resume:
            tutor.load_state(args.state)
            if args.question.strip():
                follow_up = tutor.ask(args.question.strip(), max_output_tokens=args.max_output_tokens)
                print(follow_up["text"])
            else:
                print("Session resumed. Use --question or --interactive.")
        else:
            first = tutor.start_session(
                summary=args.summary,
                user_prompt=args.prompt,
                student_question=args.question,
                max_output_tokens=args.max_output_tokens,
                temperature=args.temperature,
            )
            print(first["text"])

        if args.interactive:
            while True:
                question = input("\nYou: ").strip()
                if question.lower() in {"quit", "exit", "q"}:
                    break
                if not question:
                    continue
                answer = tutor.ask(
                    question,
                    max_output_tokens=args.max_output_tokens,
                    temperature=args.temperature,
                )
                print(f"\nTutor:\n{answer['text']}")

        tutor.save_state(args.state)

    except FileNotFoundError as exc:
        print(f"[GPT Tutor] File error: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except RuntimeError as exc:
        print(f"[GPT Tutor] Configuration error: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except AuthenticationError:
        print(
            "[GPT Tutor] Authentication failed (invalid API key). "
            "Set a valid OPENAI_API_KEY and try again.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    except RateLimitError:
        print(
            "[GPT Tutor] Rate limit or quota reached. Check billing/quota and retry.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    except BadRequestError as exc:
        print(
            f"[GPT Tutor] Bad request: {exc}. "
            "Check model name, prompt size, and request parameters.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    except (APIConnectionError, APITimeoutError):
        print(
            "[GPT Tutor] Network error while contacting the API. Check internet and retry.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    except APIStatusError as exc:
        print(f"[GPT Tutor] API error (status {exc.status_code}): {exc}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
