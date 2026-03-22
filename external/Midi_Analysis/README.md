# MIDI Analysis Module

MIDI performance analysis pipeline with optional GPT tutor feedback.

Core flow:
1. Parse reference and performance MIDI.
2. Align notes (timing-aware).
3. Compute error metrics and recommendations.
4. Export machine-readable reports.
5. Feed `gpt_summary.json` into `gpt_tutor.py` for interactive feedback.

## Requirements

- Python 3.10+
- `pip`
- OpenAI API key (for GPT tutor only)

## Project Layout

- `sample_files/reference.mid`
- `sample_files/performance.mid`
- `src/analyzer.py`
- `src/gpt_tutor.py`
- `analysis_results/` (generated outputs)

## Setup

### Windows (PowerShell)

```powershell
cd midi-analysis-module
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### macOS/Linux

```bash
cd midi-analysis-module
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Configure OpenAI API Key (Tutor Only)

Use placeholders in docs and code. Never commit real keys.

### Windows (PowerShell)

```powershell
# Current terminal only
$env:OPENAI_API_KEY="your_openai_api_key_here"

# Optional default model
$env:OPENAI_MODEL="gpt-5-mini"

# Optional persist for new terminals
setx OPENAI_API_KEY "your_openai_api_key_here"
setx OPENAI_MODEL "gpt-5-mini"
```

### macOS/Linux

```bash
export OPENAI_API_KEY="your_openai_api_key_here"
export OPENAI_MODEL="gpt-5-mini"
```

Verify key shape without printing full secret:

```powershell
python -c "import os; k=os.getenv('OPENAI_API_KEY',''); print('set=',bool(k), 'len=',len(k), 'prefix=',k[:8] if k else '')"
```

## Run the Analysis Pipeline

### 1) Reference vs Performance Analysis

```bash
python -c "from src.analyzer import compare_performance; compare_performance('sample_files/reference.mid','sample_files/performance.mid','analysis_results')"
```

Paper-based note matcher (ASAP/parangonar, best automatic model from the paper: `hDTW+sym`):

```bash
python -c "from src.analyzer import compare_performance; compare_performance('sample_files/reference.mid','sample_files/performance.mid','analysis_results', alignment_backend='paper_best', alignment_model='automatic_hdtw_sym')"
```

Generates:

- `analysis_results/full_analysis.json`
- `analysis_results/gpt_summary.json`
- `analysis_results/alignment_report.json`
- `analysis_results/error_details.json`

### 2) Solo Performance Analysis

```bash
python -c "from src.analyzer import quick_analyze; import json; r=quick_analyze('sample_files/performance.mid'); print(json.dumps(r.get('metrics', {}), indent=2))"
```

### 3) Parse-Only Inspection

```bash
python -c "from src.analyzer import MIDIAnalyzer; MIDIAnalyzer().print_parsed_data('sample_files/performance.mid')"
```

### 4) Check Output Files

```powershell
Get-ChildItem analysis_results
```

## Run GPT Tutor

### One-Shot Feedback

```bash
python -m src.gpt_tutor --summary analysis_results/gpt_summary.json --question "Give me constructive feedback."
```

### Add Prompt Steering

```bash
python -m src.gpt_tutor --summary analysis_results/gpt_summary.json --prompt "Focus on timing and dynamics." --question "What should I practice first?"
```

### Interactive Session

```bash
python -m src.gpt_tutor --summary analysis_results/gpt_summary.json --interactive
```

Type follow-up questions, then `exit` or `q` to end.

### Resume Previous Tutor Session

```bash
python -m src.gpt_tutor --resume --state analysis_results/tutor_session.json --question "Where exactly am I rushing?"
```

### Tune Output Length/Style

```bash
python -m src.gpt_tutor --summary analysis_results/gpt_summary.json --question "Give concise feedback." --max-output-tokens 500
```

```bash
python -m src.gpt_tutor --summary analysis_results/gpt_summary.json --question "Be detailed." --temperature 0.2
```

### CLI Help

```bash
python -m src.gpt_tutor --help
```

## API Smoke Test (Before Tutor)

Use this if tutor errors and you need to isolate API connectivity:

```bash
python -c "from openai import OpenAI; c=OpenAI(); r=c.responses.create(model='gpt-5-mini', input='ping'); print(r.output_text)"
```

## Troubleshooting

### `401 invalid_api_key`

Cause:
- Placeholder key used.
- Quotes accidentally included in key value.
- Revoked/expired key.

Fix:

```powershell
Remove-Item Env:OPENAI_API_KEY -ErrorAction SilentlyContinue
$env:OPENAI_API_KEY = (Get-Clipboard).Trim().Trim('"').Trim("'")
python -c "import os; k=os.getenv('OPENAI_API_KEY',''); print('len=',len(k), 'first=',repr(k[:1]), 'last=',repr(k[-1:]))"
```

### `429 insufficient_quota`

Cause:
- Billing/quota exhausted for API project.

Fix:
- Add credits / enable billing in OpenAI platform.

### PowerShell shows `>>`

Cause:
- Incomplete command (continuation mode).

Fix:
- Press `Ctrl+C`, then re-run the command in one line.

### `Summary file not found`

Cause:
- `analysis_results/gpt_summary.json` not generated yet.

Fix:
- Run compare command first:
  `python -c "from src.analyzer import compare_performance; compare_performance('sample_files/reference.mid','sample_files/performance.mid','analysis_results')"`

## Optional Legacy Script

There is also a CLI smoke-test script:

```bash
python test_midi_analysis.py list
python test_midi_analysis.py solo --midi sample_files/performance.mid
python test_midi_analysis.py compare --reference sample_files/reference.mid --performance sample_files/performance.mid
```

Note:
- It uses relative paths by default and supports explicit path overrides.
- For production usage, prefer the direct commands above.
