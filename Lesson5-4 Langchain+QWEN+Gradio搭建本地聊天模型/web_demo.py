import gradio as gr
# 为上述“结合Langchain调用Qwen” 中的 class Qwen2 
from Qwen2 import Qwen2
# 模型所在的地址
model_path = "/mnt/diskb5/zhangbo_private/llm_model/qwen/Qwen2___5-7B-Instruct"
# model_path = "/mnt/diskb5/LLM/Qwen/Qwen2-72B-Instruct-AWQ"
# 载入模型
llm = Qwen2()
llm.load_model(model_path)

# 定义一个简单的predict
def predict(query,history):
    response = llm(query)
    # 将LLM中的history替换为只要query，避免上下文太长导致模型无法理解
    llm.query_only(query)

    return response
# 调用 Gradio 中的预制前端界面
gr.ChatInterface(predict).launch(share=True)