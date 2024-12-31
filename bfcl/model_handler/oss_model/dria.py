from bfcl.model_handler.oss_model.base_oss_handler import OSSHandler
from bfcl.model_handler.utils import convert_to_function_call
from overrides import override
import json
import re


class DriaHandler(OSSHandler):
    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)
        # Set any specific attributes for your model
        self.stop_token_ids = [151643, 151644, 151645]  # The stop tokens for Qwen
        self.skip_special_tokens = True

    def _convert_json_schema_to_pythonic(self, function: dict) -> str:
        """Convert JSON function schema to Pythonic format."""
        name = function["name"]
        params = function["parameters"]["properties"]
        required = function.get("parameters", {}).get("required", [])

        param_list = []
        for param_name, param_info in params.items():
            param_type = param_info.get("type", "Any")
            if param_type == "integer":
                param_type = "int"
            elif param_type == "string":
                param_type = "str"
            elif param_type == "number":
                param_type = "float"
            elif param_type == "boolean":
                param_type = "bool"

            if param_name not in required:
                param_list.append(f"{param_name}: {param_type} = None")
            else:
                param_list.append(f"{param_name}: {param_type}")

        params_str = ", ".join(param_list)
        description = function.get("description", "")

        return f"def {name}({params_str}):\n    \"{description}\"\n"

    @override
    def _format_prompt(self, messages, function):
        # Convert functions to Pythonic format
        pythonic_functions = []
        if isinstance(function, list):
            for f in function:
                pythonic_functions.append(self._convert_json_schema_to_pythonic(f))
        else:
            pythonic_functions.append(self._convert_json_schema_to_pythonic(function))

        formatted_prompt = "<|im_start|>system\n"
        formatted_prompt += """You are an expert AI assistant that specializes in providing Python code to solve the task/problem at hand provided by the user.

You can use Python code freely, including the following available functions:

<|functions_schema|>
{}
<|end_functions_schema|>

The following dangerous builtins are restricted for security:
- exec
- eval
- execfile
- compile
- importlib
- __import__
- input
- exit

Think step by step and provide your reasoning, outside of the function calls.
You can write Python code and use the available functions. Provide all your python code in a SINGLE markdown code block like the following:

```python
result = example_function(arg1, "string")
result2 = example_function2(result, arg2)
```

DO NOT use print() statements AT ALL. Avoid mutating variables whenever possible.""".format(
            "\n".join(pythonic_functions))

        formatted_prompt += "<|im_end|>\n"

        # Add the messages
        for message in messages:
            formatted_prompt += f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"

        formatted_prompt += "<|im_start|>assistant\n"
        return formatted_prompt

    @override
    def decode_ast(self, result: str, language: str = "Python") -> list:
        """Extract function calls from Python code block."""
        code_match = re.search(r'```python\n(.*?)\n```', result, re.DOTALL)
        if not code_match:
            return []

        code = code_match.group(1)
        function_calls = []
        pattern = r'(\w+)\(((?:[^()]+|(?R))*)\)'
        matches = re.finditer(pattern, code)

        for match in matches:
            func_name = match.group(1)
            args_str = match.group(2)

            args_dict = {}
            if args_str:
                args_parts = re.findall(r'(\w+)\s*=\s*([^,]+)(?:,|$)', args_str)
                for arg_name, arg_value in args_parts:
                    arg_value = arg_value.strip()
                    try:
                        arg_value = eval(arg_value)
                    except:
                        pass
                    args_dict[arg_name] = arg_value

            function_calls.append({func_name: args_dict})

        return function_calls

    @override
    def decode_execute(self, result: str) -> list:
        """Convert the decoded function calls to execution format."""
        function_calls = self.decode_ast(result)
        execution_list = []

        for func_call in function_calls:
            for func_name, args in func_call.items():
                args_str = ", ".join([f"{k}={repr(v)}" for k, v in args.items()])
                execution_list.append(f"{func_name}({args_str})")

        return execution_list