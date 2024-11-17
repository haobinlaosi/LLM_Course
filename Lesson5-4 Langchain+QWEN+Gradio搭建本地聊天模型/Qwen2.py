### 文件名为Qwen2.py，自定义文件名也可以，只要 from xxx import xxx 对应即可
# 一个简单的对话应用，能够保存对话历史
from langchain.llms.base import LLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional
import torch

class Qwen2(LLM):
    
    # 模型参数
    max_new_tokens: int = 1920
    temperature: float  = 0.9
    top_p: float = 0.8
    tokenizer: object = None
    model: object = None
    history: List = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self,max_new_tokens = 1920,
                      temperature = 0.9,
                      top_p = 0.8):
        super().__init__()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    @property
    def _llm_type(self) -> str:
        return "Qwen2"
        
    # 载入模型，max_memory代表在载入模型阶段该显卡最多使用显存的大小，AWQ量化版本不支持模型载入到CPU内存中
    def load_model(self, model_name_or_path=None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
        self.model =AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype="auto",
            # torch_dtype="torch.float16",
            device_map="auto",       #sequential/auto/balanced_low_0
            max_memory={0:"40GB",1: "40GB", 2: "8GB", 3: "8GB"}
        )
        
        # 模型主要的chat功能实现
    def chat_stream(self, model, tokenizer, query:str, history:list):
        with torch.no_grad():    
            # 历史整理
            messages = [
                {'role': 'system', 'content': '###角色\n你是一位临床心血管医生，你在心血管领域有非常深入的知识。你非常擅长使用通俗易懂的方式去准确回答心血管问题。 ###目标\n我希望你根据用户提出的心血管相关临床问题，提供准确、专业且易于理解的回答。\n用户将提供问题相关的文档，这些文档按与问题的相关性从高到低排列。你需要结合这些文档内容以及你本身的医学知识，正确回答题目并提供详细解释。\n你的所有回答都应有医学依据，确保回答的准确性和可靠性。'},
            ]
            # 将之前的history内容重新组合
            for item in history:
                if item['role'] == 'user':
                    if item.get('content'):
                        messages.append({'role': 'user', 'content': item['content']})
                if item['role'] == 'assistant':
                    if item.get('content'):
                        messages.append({'role': 'assistant', 'content': item['content']})
            # 最新的用户问题            
            messages.append({'role': 'user', 'content': query})
            # 模型推理
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
                        
            model_inputs = tokenizer([text], return_tensors="pt").to(self.device)
            
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                top_p=self.top_p,
                temperature=self.temperature
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
                        # 模型根据messages的内容后的输出
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]  
                        # 将模型输出组合到messages中
            messages.append({'role': 'assistant', 'content': response})

        return response ,messages

        # Langchain调用
    def _call(self, prompt: str ,stop: Optional[List[str]] = ["<|user|>"]):
        # 主要调用chat_stream实现
        response, self.history = self.chat_stream(self.model, self.tokenizer, prompt, self.history)

        return response    
        
    # 当使用RAG技术时会出现用户输入存在大量的参考资料，导致模型难以理解整体上下文内容。当LLM生成回复后，使用该函数可将history的用户输入转换为不含参考资料的内容
    def query_only(self, query):
        if self.history[-2]['role'] == 'user':
            self.history[-2]['content'] = query
            
    # 返回模型history
    def get_history(self) -> List:
        return self.history
    
    # 删除模型所有history
    def delete_history(self):
        del self.history 
        self.history = []