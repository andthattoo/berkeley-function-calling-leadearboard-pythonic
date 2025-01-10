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
    @staticmethod
    def _convert_json_to_python_schema(function: dict) -> str:
        """
        Convert a JSON function schema to a Python function definition with a ReST-style docstring.
        If a parameter is optional (not in 'required'), assign a default value based on its type hint
        or the first enum value, if 'enum' is present.
        """
        name = function["name"]
        description = function.get("description", "")
        params = function["parameters"]["properties"]
        required = function["parameters"].get("required", [])

        # Build parameter list with type hints
        param_list = []
        for param_name, param_info in params.items():
            json_type = param_info.get("type", "Any")
            enum_values = param_info.get("enum")

            # If 'enum' exists, use Literal
            if enum_values:
                param_type = f"Literal[{', '.join(repr(v) for v in enum_values)}]"
            elif json_type == "integer":
                param_type = "int"
            elif json_type == "string":
                param_type = "str"
            elif json_type == "number":
                param_type = "float"
            elif json_type == "boolean":
                param_type = "bool"
            elif json_type == "array":
                item_type = param_info.get("items", {}).get("type", "Any")
                param_type = f"List[{item_type}]"
            else:
                param_type = "Any"

            if param_name in required:
                param_list.append(f"{param_name}: {param_type}")
            else:
                param_list.append(f"{param_name}: Optional[{param_type}] = {'None'}")

        params_str = ", ".join(param_list)

        # Create function definition
        func_def = f"def {name}({params_str}):\n"

        # Create ReST-style docstring
        docstring = f'    """{description}\n\n'
        for param_name, param_info in params.items():
            param_desc = param_info.get("description", "")
            docstring += f"    :param {param_name}: {param_desc}\n"

        # docstring += "    :return: Describe the return value here.\n"
        if required:
            docstring += "    :raises ValueError: If required parameters are missing.\n"
        docstring += '    """\n'

        return func_def + docstring

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

    @override
    def decode_ast(self, result, language="Python"):
        try:
            calls = re.findall(r'([\w\.]+)\((.*?)\)', result)
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
            print(results)
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

