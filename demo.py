#!/usr/bin/env python3
"""
Vietnamese AMR Parser – Gradio Demo
Model: MTUP2 (Qwen2.5-7B-Instruct + LoRA)
AI Evaluator: Claude API (claude-sonnet-4-6)

Usage:
    python demo.py --adapter checkpoints/mtup_2tasks/MTUP2/s42/final
"""

import re
import sys
import argparse
import textwrap
from pathlib import Path

import torch
import gradio as gr
import anthropic
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).parent / "src"))
from postprocess import postprocess_amr

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

SYSTEM_PROMPT = """Bạn là chuyên gia phân tích AMR (Abstract Meaning Representation) cho tiếng Việt.

QUY TẮC QUAN TRỌNG:
1. Chỉ dùng từ tiếng Việt trong câu gốc làm concept. KHÔNG tự tạo từ mới.
2. Concept AMR chuẩn được phép dùng (tiếng Anh): person, name, now, after, before, and, or, contrast-01, obligate-01, possible-01, rate-entity-91, temporal-quantity, percentage-entity, multi-sentence, have-rel-role-91, have-org-role-91, amr-unknown, province, city, mountain, country, company, ethnic-group, date-entity, date-interval, monetary-quantity, about
3. :classifier cho lượng từ: người, anh, chị, cái, con, chiếc, đứa, ngôi, căn, cùi, ngọn, chú, cánh
4. :polarity - (dấu trừ literal) cho phủ định
5. :modality + cho được, :modality - cho bị
6. multi-sentence cho câu có nhiều mệnh đề độc lập (phân tách bởi dấu phẩy)
7. :wiki - cho tên riêng người/địa danh (dấu trừ literal)
8. :pivot cho trạng thái/tính từ, :agent cho hành động
9. "hiện nay", "đến nay", "nay" → (n / now)
10. Từ ghép giữ nguyên: "làm việc", "chịu thua", "thoát nghèo", "vận động"
11. Đảm bảo dấu ngoặc mở và đóng cân bằng
12. Mỗi biến (chữ cái + số tùy chọn) phải DUY NHẤT trong toàn bộ AMR
13. :prep cho giới từ: về, từ, ở, vào, trong, cho, tại, bởi, vì, để, với
14. "theo X, ..." → root là (t / theo :source(X) :topic(mệnh đề chính))
15. "thưa X, ..." → thêm :polite+(wz / sir) vào root verb của câu chính
16. Động từ chuyển động sau main verb (đi, về, lại, tiếp, ra, vào) → :manner(đ / đi), KHÔNG dùng :compound
17. :domain cho tính từ/trạng thái khi subject không chủ động hành động
18. "X nói/khẳng định/cho biết: ..." → (s / say-01 :agent X :topic(mệnh đề))"""

CLAUDE_EVAL_PROMPT = """\
Bạn là chuyên gia đánh giá AMR (Abstract Meaning Representation) tiếng Việt.

Câu gốc: {sentence}

AMR được sinh ra:
{amr}

Hãy đánh giá AMR trên theo 4 tiêu chí. Trả lời ngắn gọn, rõ ràng:

1. **Tính đúng đắn cú pháp** (Syntactic validity): AMR có cân bằng ngoặc không? Variables có duy nhất không? (Đúng / Sai – giải thích nếu Sai)

2. **Tính đúng đắn ngữ nghĩa** (Semantic correctness): Root concept có phản ánh đúng vị ngữ chính của câu không? Các relation (:agent, :theme, :pivot…) có gán đúng không? (Tốt / Chấp nhận được / Có lỗi – nêu lỗi cụ thể nếu có)

3. **Độ phủ** (Coverage): Có từ/ý nào quan trọng trong câu bị bỏ sót không? (Đầy đủ / Thiếu – nêu cụ thể nếu thiếu)

4. **Nhận xét tổng thể** (Overall): Đánh giá ngắn 1–2 câu và cho điểm từ 1–5 (1=rất tệ, 5=xuất sắc).
"""

# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

_model = None
_tokenizer = None


def load_model(adapter_path: str):
    global _model, _tokenizer
    if _model is not None:
        return

    print(f"Loading tokenizer from {BASE_MODEL}...")
    _tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, padding_side="left"
    )
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    print(f"Loading base model {BASE_MODEL}...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter from {adapter_path}...")
    _model = PeftModel.from_pretrained(base, adapter_path)
    _model.eval()
    print("Model ready.")


# ─────────────────────────────────────────────────────────────────────────────
# AMR parsing
# ─────────────────────────────────────────────────────────────────────────────

def extract_balanced_amr(text: str, start_pos: int):
    paren_start = text.find("(", start_pos)
    if paren_start == -1:
        return None
    depth = 0
    for i in range(paren_start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[paren_start : i + 1]
    last_close = text.rfind(")", paren_start)
    if last_close != -1:
        fragment = text[paren_start : last_close + 1]
        opens = fragment.count("(")
        closes = fragment.count(")")
        if opens > closes:
            fragment += ")" * (opens - closes)
        return fragment
    return None


def parse_sentence(sentence: str) -> tuple[str, str]:
    """Returns (amr_flat, raw_model_output)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": sentence.strip()},
    ]
    text = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].to(_model.device)

    with torch.no_grad():
        output_ids = _model.generate(
            input_ids,
            max_new_tokens=1536,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=_tokenizer.pad_token_id,
            eos_token_id=_tokenizer.eos_token_id,
        )

    new_ids = output_ids[0][input_ids.shape[1] :]
    response = _tokenizer.decode(new_ids, skip_special_tokens=True)

    # Extract [STEP 2: AMR_WITH_VAR]
    marker = response.rfind("[STEP 2: AMR_WITH_VAR]")
    if marker == -1:
        marker = response.rfind("[STEP 1: AMR_NO_VAR]")
    if marker == -1:
        return "(a / amr-unknown)", response

    amr = extract_balanced_amr(response, marker) or "(a / amr-unknown)"
    amr = postprocess_amr(amr)
    amr_flat = re.sub(r"\s+", " ", amr).strip()
    return amr_flat, response


def format_amr_pretty(amr_flat: str) -> str:
    """Indent AMR for display."""
    result = []
    indent = 0
    i = 0
    while i < len(amr_flat):
        c = amr_flat[i]
        if c == "(":
            result.append("\n" + "    " * indent + "(")
            indent += 1
        elif c == ")":
            indent = max(0, indent - 1)
            result.append(")")
        elif c == ":" and i > 0 and amr_flat[i - 1] == " ":
            result.append("\n" + "    " * indent + ":")
            i += 1
            continue
        else:
            result.append(c)
        i += 1
    return "".join(result).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Claude evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_with_claude(sentence: str, amr: str, api_key: str) -> str:
    if not api_key.strip():
        return "⚠️ Chưa nhập Anthropic API key."
    if amr == "(a / amr-unknown)":
        return "⚠️ AMR không hợp lệ, không thể đánh giá."

    client = anthropic.Anthropic(api_key=api_key.strip())
    prompt = CLAUDE_EVAL_PROMPT.format(sentence=sentence, amr=amr)

    try:
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    except Exception as e:
        return f"❌ Lỗi khi gọi Claude API: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────────────────────

EXAMPLES = [
    "Tôi yêu Hà Nội.",
    "Anh ấy không đi làm hôm nay.",
    "Cô giáo dạy học sinh viết văn.",
    "Hiện nay xã có 68 tổ nhân dân.",
    "Chủ trương tốt nhưng dân không hiểu.",
]


def run_parse_and_eval(sentence: str, api_key: str):
    if not sentence.strip():
        return "", "", "", "⚠️ Vui lòng nhập câu."

    amr_flat, raw_output = parse_sentence(sentence)
    amr_pretty = format_amr_pretty(amr_flat)
    evaluation = evaluate_with_claude(sentence, amr_flat, api_key)
    return amr_flat, amr_pretty, raw_output, evaluation


def build_ui():
    with gr.Blocks(title="Vietnamese AMR Parser", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
# Vietnamese AMR Parser
**Model:** MTUP2 — Qwen2.5-7B-Instruct + LoRA (Smatch F1 = 0.646)
**AI Evaluator:** Claude (claude-sonnet-4-6)
"""
        )

        with gr.Row():
            with gr.Column(scale=3):
                sentence_input = gr.Textbox(
                    label="Câu tiếng Việt",
                    placeholder="Nhập câu cần parse...",
                    lines=2,
                )
                api_key_input = gr.Textbox(
                    label="Anthropic API Key",
                    placeholder="sk-ant-...",
                    type="password",
                    lines=1,
                )
                parse_btn = gr.Button("Parse & Evaluate", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("**Ví dụ:**")
                for ex in EXAMPLES:
                    gr.Button(ex, size="sm").click(
                        fn=lambda s=ex: s,
                        outputs=sentence_input,
                    )

        with gr.Row():
            with gr.Column():
                amr_flat_out = gr.Textbox(
                    label="AMR (single line – dùng cho eval)",
                    lines=3,
                    show_copy_button=True,
                )
            with gr.Column():
                amr_pretty_out = gr.Textbox(
                    label="AMR (indented – dễ đọc)",
                    lines=10,
                    show_copy_button=True,
                )

        eval_out = gr.Markdown(label="Claude Evaluation")

        with gr.Accordion("Raw model output (debug)", open=False):
            raw_out = gr.Textbox(label="Full model response", lines=15)

        parse_btn.click(
            fn=run_parse_and_eval,
            inputs=[sentence_input, api_key_input],
            outputs=[amr_flat_out, amr_pretty_out, raw_out, eval_out],
        )

        sentence_input.submit(
            fn=run_parse_and_eval,
            inputs=[sentence_input, api_key_input],
            outputs=[amr_flat_out, amr_pretty_out, raw_out, eval_out],
        )

    return demo


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter",
        default="checkpoints/mtup_2tasks/MTUP2/s42/final",
        help="Path to LoRA adapter directory",
    )
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    args = parser.parse_args()

    adapter_path = Path(args.adapter)
    if not adapter_path.exists():
        print(f"ERROR: Adapter not found at {adapter_path}")
        print("Tải checkpoint từ server về trước rồi chạy lại.")
        sys.exit(1)

    load_model(str(adapter_path))

    ui = build_ui()
    ui.launch(server_port=args.port, share=args.share)
