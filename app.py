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
    torch_dtype=torch.bfloat16,   # ~6GB instead of ~12GB at fp32
    low_cpu_mem_usage=True,
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
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,             # returns {input_ids, attention_mask}
    ).to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs,                 # passes input_ids AND attention_mask
            max_new_tokens=128,
            do_sample=False,          # greedy = deterministic, matches eval
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
    return text.strip()


EXAMPLES = [
    # simple COUNT with a numeric filter
    ["CREATE TABLE head (age INTEGER)",
     "How many heads of the departments are older than 56?"],
    # lookup by name
    ["CREATE TABLE table_11803648_17 (nationality VARCHAR, player VARCHAR)",
     "Where is Andre Petersson from?"],
    # filter with two conditions
    ["CREATE TABLE employees (name VARCHAR, salary INTEGER, department VARCHAR)",
     "List the names of employees in the Sales department earning over 50000."],
    # MAX aggregate
    ["CREATE TABLE products (name VARCHAR, price INTEGER, category VARCHAR)",
     "What is the most expensive product?"],
    # MIN aggregate
    ["CREATE TABLE flights (flight_no VARCHAR, duration INTEGER, airline VARCHAR)",
     "Which flight has the shortest duration?"],
    # AVG aggregate
    ["CREATE TABLE students (name VARCHAR, grade INTEGER, class VARCHAR)",
     "What is the average grade of students in class A?"],
    # SUM aggregate
    ["CREATE TABLE orders (order_id INTEGER, amount INTEGER, customer VARCHAR)",
     "What is the total amount spent by customer John Smith?"],
    # COUNT of everything
    ["CREATE TABLE movies (title VARCHAR, year INTEGER, genre VARCHAR)",
     "How many movies were released in 2020?"],
    # ORDER BY / top result
    ["CREATE TABLE cities (name VARCHAR, population INTEGER, country VARCHAR)",
     "List the top 5 cities by population."],
    # DISTINCT
    ["CREATE TABLE sales (region VARCHAR, product VARCHAR, revenue INTEGER)",
     "What are the distinct regions where products were sold?"],
    # string / partial match
    ["CREATE TABLE books (title VARCHAR, author VARCHAR, pages INTEGER)",
     "Find all books written by an author whose name contains 'King'."],
    # numeric range
    ["CREATE TABLE cars (model VARCHAR, year INTEGER, mileage INTEGER)",
     "Show cars made between 2015 and 2020."],
    # GROUP BY with count
    ["CREATE TABLE table_2891_4 (team VARCHAR, wins INTEGER, season VARCHAR)",
     "How many wins does each team have?"],
    # ordering ascending
    ["CREATE TABLE marathon (runner VARCHAR, finish_time INTEGER, country VARCHAR)",
     "Who had the fastest finish time?"],
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
