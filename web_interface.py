from ollama import  generate
import gradio as gr
import time
import re

def get_llm_response(input_text, task='revision'):
    """
    Parameters:
        model_name: str
            Name of the model to be used for the task
        input_text: str
            Input text from whisper transcribed result
        task: str
            Task to be performed by the model. 'revision' means to correct the input text but do not change the content.
            'response' means to generate a response to the input text.
    Returns:
        json format file for 'ie' task. 
        conversation text format file for 'ca' task.
    """    
    revision_prompt = """
    あなたは法律文書の専門家です。以下の指示に従って文書を処理してください：

    基本方針：
    - 原文をできるだけ尊重する
    - 明らかな誤りのみを修正する
    - 法的に問題のある箇所のみを修正する
    - 文体や表現の好みによる変更は避ける
    - 修正理由や説明は省略する

    修正対象：
    - 法律用語の明らかな誤用
    - 法令の引用の誤り
    - 重大な文法的誤り
    - 明らかな事実誤認
    - 法的な論理の矛盾

    出力形式：
    [修正後の文書のみを出力]
    """
    response_prompt = "あなたは専門的な日本法律顧問です。質問されたことに対して専門的な法律の見地から日本語で答えてください。重複内容を出力しないでください"

    model_name = 'llama3.3_jp_keiyaku_1221_Q4KM'

    generation_params = {
        #"do_sample": True,
        "temperature": 0.6,
        "top_p": 0.9,
        "top_k": 40,
        "num_predict": 1024,
        "num_ctx": 2048,
        "repeat_penalty": 1.2,
    }

    if task == 'revision':

        system_prompt = revision_prompt

        res = generate(
            model=model_name,
            prompt=input_text,
            system=system_prompt,
            options=generation_params
        )
    elif task == 'response':

        system_prompt = response_prompt

        res = generate(
            model=model_name,
            prompt=input_text,
            system=system_prompt,
            options=generation_params
        )
    else:
        print("Invalid task. Please choose either 'revision' or 'response'.")

    response = res['response']
    for i in range(len(response)):
        yield response[:i+1]
        time.sleep(0.01)

    # return res['response']


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    # タイトル
    gr.Markdown("# 契約書内容処理アプリ")
    gr.Markdown("このアプリでは、契約書内容の修正や質問への回答が可能です。")
    
    # 入力テキストエリア
    with gr.Row():
        input_text = gr.Textbox(
            label="契約書内容を入力してください",
            placeholder="ここに処理したいテキストを入力してください...",
            lines=10
        )
    
    # タスク選択エリア
    with gr.Row():
        task_radio = gr.Radio(
            choices=[("回答する", "response"), ("修正する", "revision")],
            label="タスクを選択してください",
            value="response"
        )
    
    # 送信ボタン
    with gr.Row():
        submit_btn = gr.Button("送信", variant="primary")
    
    # 出力エリア
    with gr.Row():
        output_text = gr.Textbox(
            label="処理結果",
            lines=20
        )
    
    # クリックイベントの設定
    submit_btn.click(
        fn=get_llm_response,
        inputs=[input_text, task_radio],
        outputs=output_text,
        api_name="stream_output"
    )

if __name__ == "__main__":

    demo.queue()
    demo.launch()