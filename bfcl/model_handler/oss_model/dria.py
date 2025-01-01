from bfcl.model_handler.oss_model.base_oss_handler import OSSHandler
from overrides import override
import re
import ast
import json
import inspect


class DriaHandler(OSSHandler):
    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)
        self.stop_token_ids = [151643, 151644, 151645]
        self.skip_special_tokens = True

    def _convert_json_to_python_schema(self, function: dict) -> str:
        """Convert a JSON function schema to a Python function definition with docstring."""
        name = function["name"]
        description = function.get("description", "")
        params = function["parameters"]["properties"]
        required = function["parameters"].get("required", [])

        # Build parameter list with type hints
        param_list = []
        for param_name, param_info in params.items():
            param_type = param_info.get("type", "Any")
            # Convert JSON types to Python types
            if param_type == "integer":
                param_type = "int"
            elif param_type == "string":
                param_type = "str"
            elif param_type == "number":
                param_type = "float"
            elif param_type == "boolean":
                param_type = "bool"
            elif param_type == "array":
                item_type = param_info.get("items", {}).get("type", "Any")
                param_type = f"List[{item_type}]"

            # Add default None if parameter is optional
            if param_name not in required:
                param_list.append(f"{param_name}: {param_type} = None")
            else:
                param_list.append(f"{param_name}: {param_type}")

        params_str = ", ".join(param_list)

        # Create function definition with docstring
        func_def = f"def {name}({params_str}):\n"
        func_def += f'    """{description}\n\n'

        # Add parameter descriptions to docstring
        for param_name, param_info in params.items():
            param_desc = param_info.get("description", "")
            func_def += f"    Args:\n        {param_name}: {param_desc}\n"

        func_def += '    """\n'

        return func_def

    @override
    def _format_prompt(self, messages, function):
        # Convert function to appropriate format if needed

        if isinstance(function, list):
            python_functions = [self._convert_json_to_python_schema(f) for f in function]
            formatted_functions = "\n\n".join(python_functions)
        else:
            formatted_functions = self._convert_json_to_python_schema(function)

        formatted_prompt = inspect.cleandoc(
            """<|im_start|>system
            You are an expert AI assistant that specializes in providing Python code to solve the task/problem at hand provided by the user.

            You can use Python code freely, including the following available functions:

            <|functions_schema|>
            {functions}
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

            DO NOT use print() statements AT ALL. Avoid mutating variables whenever possible.
            <|im_end|>
            """
        )

        # Format the prompt with the functions
        formatted_prompt = formatted_prompt.format(
            functions=formatted_functions
        )

        # Add conversation messages
        for message in messages:
            formatted_prompt += f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"

        # Add final assistant token
        formatted_prompt += "<|im_start|>assistant\n"

        return formatted_prompt

    def _parse_value(self, value_str: str):
        """Helper to parse argument values."""
        value_str = value_str.strip()

        # Remove quotes if present
        if (value_str.startswith('"') and value_str.endswith('"')) or \
                (value_str.startswith("'") and value_str.endswith("'")):
            value_str = value_str[1:-1]

        # Handle boolean values
        if value_str.lower() == 'true':
            return True
        if value_str.lower() == 'false':
            return False

        # Try to evaluate as Python literal
        try:
            return ast.literal_eval(value_str)
        except:
            # If evaluation fails, return as string
            return value_str

    @override
    def decode_ast(self, result, language="Python"):
        try:
            block = re.search(r'```python\s*(.*?)\s*```', result, re.DOTALL)
            if not block: return []
            code = block.group(1)
            calls = re.findall(r'([\w\.]+)\((.*?)\)', code)
            results = []
            for func, args in calls:
                parts, cur, bc, q = [], [], 0, None
                for c in args:
                    if c in '([{':
                        bc += 1
                    elif c in ')]}':
                        bc -= 1
                    elif c in "\"'":
                        q = None if q == c else (c if not q else q)
                    if c == ',' and bc == 0 and not q:
                        parts.append(''.join(cur).strip())
                        cur = []
                    else:
                        cur.append(c)
                if cur: parts.append(''.join(cur).strip())
                d = {}
                for p in parts:
                    if '=' in p:
                        k, v = p.split('=', 1)
                        k, v = k.strip(), v.strip()
                        try:
                            v = ast.literal_eval(v)
                        except:
                            v = v.strip("\"'")
                        d[k] = v
                results.append({func: d})
            # print(results)
            return results
        except:
            print(f"Warning, couldn't parse {result}")
            return []

    @override
    def decode_execute(self, result):
        """Convert to execution format."""
        if not isinstance(result, str):
            return []

        function_calls = self.decode_ast(result)
        execution_list = []

        for func_call in function_calls:
            for func_name, args in func_call.items():
                args_str = ", ".join([f"{k}={repr(v)}" for k, v in args.items()])
                execution_list.append(f"{func_name}({args_str})")

        return execution_list