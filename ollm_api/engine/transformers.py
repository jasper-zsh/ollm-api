from transformers import AutoModelForCausalLM, AutoTokenizer
from ..api.openai.chat import ChatCompletionRequest, ChatCompletionChoice, Usage, ChatCompletionChunkChoice, Message
from typing import Generator

class TransformersEngine:
    def load_model(self, model_path, adapter):
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, resume_download=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, resume_download=True)
        self.adapter = adapter
    
    def encode(self, text):
        return self.tokenizer.encode(text)
    
    def chat(self, request: ChatCompletionRequest) -> (ChatCompletionChoice, Usage):
        kwargs, prompt_tokens = self.adapter.parse_request(request)
        response, _ = self.model.chat(self.tokenizer, **kwargs)
        # print(f"<chat>\n{kwargs['history']}\n{kwargs['query']}\n<!-- *** -->\n{response}\n</chat>")
        choice, completion_tokens = self.adapter.parse_response(response)
        return choice, Usage(completion_tokens=completion_tokens, prompt_tokens=prompt_tokens, total_tokens=completion_tokens+prompt_tokens)

    def chat_stream(self, request: ChatCompletionRequest) -> Generator[ChatCompletionChunkChoice, None, None]:
        kwargs, prompt_tokens = self.adapter.parse_request(request)
        yield ChatCompletionChunkChoice(index=0, delta=Message(role='assistant'))

        res_gen = self.model.chat_stream(self.tokenizer, **kwargs)
        for choice in self.adapter.parse_stream(res_gen):
            yield choice
        
        yield ChatCompletionChunkChoice(index=0, delta=Message(), finish_reason='stop')
            