from ..engine.transformers import TransformersEngine
from ..api.openai.chat import ChatCompletionRequest, Message, ChatCompletionChoice, ChatCompletionChunkChoice, FunctionCall, Usage
import copy
import json
from typing import Generator

TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters}"""


REACT_INSTRUCTION = """Answer the following questions as best you can. You have access to the following APIs:

{tools_text}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tools_name_text}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!"""

_TEXT_COMPLETION_CMD = object()

class QwenModel:
    def __init__(self, engine: TransformersEngine) -> None:
        self.engine = engine
    
    def parse_request(self, request: ChatCompletionRequest) -> (dict, int):
        ret = {}
        if request.temperature is not None:
            if request.temperature < 0.01:
                ret['top_k'] = 1
            else:
                ret['temperature'] = request.temperature
        if request.top_p is not None:
            ret['top_p'] = request.top_p
        
        # TODO: Stop Words in request
        stop_words = []
        if request.functions:
            if 'Observation:' not in stop_words:
                stop_words.append('Observation:')
        
        messages = copy.deepcopy(request.messages)
        default_system = 'You are a helpful assistant.'
        system = ''
        if messages[0].role == 'system':
            system = messages.pop(0).content.lstrip('\n').rstrip()
            if system == default_system:
                system = ''
        if request.functions:
            tools_text = []
            tools_name_text = []
            for f in request.functions:
                name = f.name
                desc = f.description
                tool = TOOL_DESC.format(
                    name_for_model=name,
                    name_for_human=name,
                    description_for_model=desc,
                    parameters=json.dumps(f.parameters, ensure_ascii=False)
                )
                tools_text.append(tool)
                tools_name_text.append(name)
            tools_text = '\n\n'.join(tools_text)
            tools_name_text = ', '.join(tools_name_text)
            system += '\n\n' + REACT_INSTRUCTION.format(
                tools_text=tools_text,
                tools_name_text=tools_name_text,
            )
            system = system.lstrip('\n').rstrip()
        
        # There's dummy thought in example code, ignore it temporary
        msgs = []
        for m_idx, m in enumerate(messages):
            role, content, f_call = m.role, m.content, m.function_call
            if content:
                content = content.lstrip('\n').rstrip()
            if role == 'function':
                msgs[-1].content += f"\nObservation: {content}"
                if m_idx == len(messages) - 1:
                    msgs[-1].content += '\nThought:'
            elif role == 'assistant':
                if f_call is not None:
                    f_name, f_args = f_call.name, f_call.arguments
                content = f"\n{content}\nAction: {f_name}\nAction Input: {f_args}"
                if msgs[-1].role == 'user':
                    msgs.append(Message(role='assistant', content=content.lstrip('\n').rstrip()))
                else:
                    msgs[-1].content += content
            elif role == 'user':
                msgs.append(Message(role='user', content=content.lstrip('\n').rstrip()))
        
        query = _TEXT_COMPLETION_CMD
        if msgs[-1].role == 'user':
            query = msgs[-1].content
            msgs = msgs[:-1]
        
        prompt_tokens = 0
        history = []
        for i in range(0, len(msgs), 2):
            if msgs[i].role == 'user' and msgs[i+1].role == 'assistant':
                user_msg = msgs[i].content
                bot_msg = msgs[i+1].content
                if system and (i == len(msgs) - 2):
                    user_msg = f'{system}\n\nQuestion: {user_msg}'
                    system = ''
                prompt_tokens += len(self.engine.encode(user_msg))
                prompt_tokens += len(self.engine.encode(bot_msg))
                history.append([user_msg, bot_msg])
        if system:
            assert query is not _TEXT_COMPLETION_CMD
            query = f'{system}\n\nQuestion: {query}'
        ret['history'] = history
        ret['query'] = query
        prompt_tokens += len(self.engine.encode(query))

        ret['stop_words_ids'] = [self.engine.encode(s) for s in stop_words] if stop_words else None

        return ret, prompt_tokens

    def parse_response(self, response: str) -> (ChatCompletionChoice, int):
        completion_tokens = len(self.engine.encode(response))
        ret = ChatCompletionChoice(
            index=0,
            message=Message(
                role='assistant'
            ),
            finish_reason='stop',
        )
        f_name, f_args = '', ''
        i = response.rfind('\nAction:')
        j = response.rfind('\nAction Input:')
        k = response.rfind('\nObservation:')
        if 0 <= i < j:
            if k < j:
                response = response.rstrip() + '\nObservation:'
            k = response.rfind('\nObservation:')
            f_name = response[i + len('\nAction:') : j].strip()
            f_args = response[j + len('\nAction Input:') : k].strip()
        if f_name:
            ret.message.content=response[:i]
            ret.message.function_call = FunctionCall(name=f_name, arguments=f_args)
            ret.finish_reason = 'function_call'
            return ret, completion_tokens
        z = response.rfind('\nFinal Answer: ')
        if z >= 0:
            response = response[z + len('\nFinal Answer: ') : ]
        ret.message.content = response
        return ret, completion_tokens
    
    def parse_stream(self, generator: Generator[str, None, None]) -> Generator[ChatCompletionChunkChoice, None, None]:
        current_length = 0
        for new_res in generator:
            if len(new_res) == current_length:
                continue

            new_text = new_res[current_length:]
            current_length = len(new_res)

            yield ChatCompletionChunkChoice(index=0, delta=Message(content=new_text))