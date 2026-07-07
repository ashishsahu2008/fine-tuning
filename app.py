"""
Text-to-SQL demo — fine-tuned Qwen2.5-3B (QLoRA).
Deploy on a Hugging Face Space. See README for hardware notes.
"""

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---- change this to YOUR pushed model repo ----
MODEL_ID = "ashishsahu2008/qwen2.5-3b-text2sql"

SYSTEM = ("You are a SQL expert. Given a database schema and a question, "
          "output only the SQL query that answers it.")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",     # fp16 on GPU, fp32 on CPU
    device_map="auto",
)
model.eval()


def generate_sql(schema, question):
    if not schema.strip() or not question.strip():
        return "-- Please provide both a schema and a question."
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",
         "content": f"Schema:\n{schema}\n\nQuestion: {question}"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            input_ids=inputs,
            max_new_tokens=128,
            do_sample=False,          # greedy = deterministic, matches eval
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
    return text.strip()


EXAMPLES = [
    ["CREATE TABLE head (age INTEGER)",
     "How many heads of the departments are older than 56?"],
    ["CREATE TABLE table_11803648_17 (nationality VARCHAR, player VARCHAR)",
     "Where is Andre Petersson from?"],
    ["CREATE TABLE employees (name VARCHAR, salary INTEGER, department VARCHAR)",
     "List the names of employees in the Sales department earning over 50000."],
]

with gr.Blocks(title="Text-to-SQL") as demo:
    gr.Markdown(
        "# Natural language → SQL\n"
        "Fine-tuned **Qwen2.5-3B** (QLoRA) on `b-mc2/sql-create-context`. "
        "Paste a `CREATE TABLE` schema and ask a question in plain English."
    )
    schema = gr.Textbox(
        label="Schema (CREATE TABLE ...)",
        lines=4,
        placeholder="CREATE TABLE employees (name VARCHAR, salary INTEGER)",
    )
    question = gr.Textbox(
        label="Question",
        lines=2,
        placeholder="Who earns the most?",
    )
    btn = gr.Button("Generate SQL", variant="primary")
    output = gr.Code(label="Generated SQL", language="sql")

    btn.click(generate_sql, inputs=[schema, question], outputs=output)
    gr.Examples(examples=EXAMPLES, inputs=[schema, question])

if __name__ == "__main__":
    demo.launch()
