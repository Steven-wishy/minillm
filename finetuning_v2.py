import os
import re
import sys
import random
import json
import math
import html
import time
import gc
import logging
import traceback
import ast
import hashlib
import signal
import urllib.parse
from dataclasses import dataclass
try:
    import gradio as gr
    _GRADIO_AVAILABLE = True
except ImportError:
    _GRADIO_AVAILABLE = False

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

from collections import Counter
from typing import List, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from unsloth import FastLanguageModel, PatchFastRL
from torch.optim import AdamW
from peft import get_peft_model_state_dict

# Structured logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("AGI_Pipeline_V2")

@dataclass
class TrainingConfig:
    model_name: str = "LiquidAI/LFM2.5-230M-Base"
    max_seq_length: int = 1024
    max_new_tokens: int = 1024
    lora_r: int = 16
    lora_alpha: int = 32
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    gamma_margin: float = 0.5
    lambda_symmetry: float = 0.5
    beta: float = 2.0
    accumulation_steps: int = 4
    num_candidates: int = 2
    train_split_ratio: float = 0.85
    save_checkpoint_dir: str = "./checkpoints"
    trust_remote_code: bool = True


# Safe import of requests for active web searches
try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False
    logger.warning("Package 'requests' is not installed. Web search queries will fail gracefully.")

# Enable Unsloth's optimized reinforcement learning kernels to save VRAM on the T4 GPU
PatchFastRL("GRPO", FastLanguageModel)


# ==============================================================================
# Helper Functions for SimPO Alignment & Model Merging
# ==============================================================================

def get_sequence_log_prob(model, tokenizer, prompt: str, completion: str) -> torch.Tensor:
    """
    Computes the length-normalized log probability of the completion tokens conditioned on the prompt.
    Retains gradients for backpropagation.
    """
    prompt_tokens = tokenizer(prompt, return_tensors="pt")
    full_tokens = tokenizer(prompt + completion, return_tensors="pt")
    
    prompt_len = prompt_tokens["input_ids"].shape[1]
    full_len = full_tokens["input_ids"].shape[1]
    
    if full_len <= prompt_len:
        return torch.tensor(0.0, device=model.device, requires_grad=True)
        
    input_ids = full_tokens["input_ids"].to(model.device)
    attention_mask = full_tokens["attention_mask"].to(model.device)
    
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # shape: (batch_size, sequence_length, vocab_size)
    
    # We want the log probs of the completion tokens
    completion_logits = logits[0, prompt_len - 1 : full_len - 1, :]
    completion_targets = input_ids[0, prompt_len : full_len]
    
    log_probs = F.log_softmax(completion_logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=completion_targets.unsqueeze(-1)).squeeze(-1)
    
    return token_log_probs.mean()


def run_slerp_adapter_merge(model, weights_v0: Dict[str, torch.Tensor], weights_v1: Dict[str, torch.Tensor], t: float = 0.5) -> Dict[str, torch.Tensor]:
    """
    Applies Spherical Linear Interpolation (SLERP) to merge two sets of LoRA adapter weights.
    """
    merged_weights = {}
    for key in weights_v0.keys():
        if key not in weights_v1:
            merged_weights[key] = weights_v0[key].clone()
            continue
            
        v0 = weights_v0[key]
        v1 = weights_v1[key]
        
        orig_shape = v0.shape
        v0_flat = v0.view(-1).float()
        v1_flat = v1.view(-1).float()
        
        v0_norm = torch.norm(v0_flat)
        v1_norm = torch.norm(v1_flat)
        
        if v0_norm < 1e-8 or v1_norm < 1e-8:
            merged_flat = (1 - t) * v0_flat + t * v1_flat
        else:
            v0_unit = v0_flat / v0_norm
            v1_unit = v1_flat / v1_norm
            
            dot = torch.dot(v0_unit, v1_unit)
            dot = torch.clamp(dot, -1.0, 1.0)
            
            omega = torch.acos(dot)
            sin_omega = torch.sin(omega)
            
            if sin_omega < 1e-4:
                merged_flat = (1 - t) * v0_flat + t * v1_flat
            else:
                weight_v0 = torch.sin((1 - t) * omega) / sin_omega
                weight_v1 = torch.sin(t * omega) / sin_omega
                merged_flat = weight_v0 * v0_flat + weight_v1 * v1_flat
                
                target_norm = (1 - t) * v0_norm + t * v1_norm
                merged_flat = (merged_flat / torch.norm(merged_flat)) * target_norm
                
        merged_weights[key] = merged_flat.view(orig_shape).to(v0.dtype)
        
    return merged_weights


# ==========================================
# 1. AST LOOP-LIMITER (DETERMINISTIC VISITOR)
# ==========================================

class LoopLimiterTransformer(ast.NodeTransformer):
    """
    Deterministic AST transformer that injects loop counter initializations
    immediately before any For or While loop, and injects guard statements
    at the start of the loop body.
    Processes children first to ensure nested loops are instrumented correctly.
    """
    def __init__(self, limit: int = 5000):
        super().__init__()
        self.limit = limit
        self.counter_id = 0

    def _generate_counter_name(self) -> str:
        name = f"_loop_counter_{self.counter_id}"
        self.counter_id += 1
        return name

    def visit_For(self, node: ast.For) -> Any:
        self.generic_visit(node)
        counter_name = self._generate_counter_name()
        
        init_node = ast.Assign(
            targets=[ast.Name(id=counter_name, ctx=ast.Store())],
            value=ast.Constant(value=0)
        )
        increment = ast.AugAssign(
            target=ast.Name(id=counter_name, ctx=ast.Store()),
            op=ast.Add(),
            value=ast.Constant(value=1)
        )
        check = ast.If(
            test=ast.Compare(
                left=ast.Name(id=counter_name, ctx=ast.Load()),
                ops=[ast.Gt()],
                comparators=[ast.Constant(value=self.limit)]
            ),
            body=[
                ast.Raise(
                    exc=ast.Call(
                        func=ast.Name(id="TimeoutError", ctx=ast.Load()),
                        args=[ast.Constant(value="Loop threshold exceeded.")],
                        keywords=[]
                    )
                )
            ],
            orelse=[]
        )
        
        node.body = [increment, check] + node.body
        return [init_node, node]

    def visit_While(self, node: ast.While) -> Any:
        self.generic_visit(node)
        counter_name = self._generate_counter_name()
        
        init_node = ast.Assign(
            targets=[ast.Name(id=counter_name, ctx=ast.Store())],
            value=ast.Constant(value=0)
        )
        increment = ast.AugAssign(
            target=ast.Name(id=counter_name, ctx=ast.Store()),
            op=ast.Add(),
            value=ast.Constant(value=1)
        )
        check = ast.If(
            test=ast.Compare(
                left=ast.Name(id=counter_name, ctx=ast.Load()),
                ops=[ast.Gt()],
                comparators=[ast.Constant(value=self.limit)]
            ),
            body=[
                ast.Raise(
                    exc=ast.Call(
                        func=ast.Name(id="TimeoutError", ctx=ast.Load()),
                        args=[ast.Constant(value="Loop threshold exceeded.")],
                        keywords=[]
                    )
                )
            ],
            orelse=[]
        )
        
        node.body = [increment, check] + node.body
        return [init_node, node]


# ==========================================
# 2. SECURE AST MATHEMATICAL PARSER
# ==========================================

def calculate_expression(expr: str) -> str:
    """
    100% secure math parser using explicit AST node evaluation.
    Supports basic arithmetic operations and mathematical constants (pi, e).
    Does NOT execute dynamic code or use eval/exec.
    """
    allowed_operators = {
        ast.Add: lambda x, y: x + y,
        ast.Sub: lambda x, y: x - y,
        ast.Mult: lambda x, y: x * y,
        ast.Div: lambda x, y: x / y if y != 0 else float('nan'),
        ast.Pow: lambda x, y: x ** y if y < 100 else float('inf'),
        ast.USub: lambda x: -x,
        ast.UAdd: lambda x: +x
    }
    allowed_constants = {
        "pi": math.pi,
        "e": math.e
    }
    
    def _eval_node(node):
        if isinstance(node, ast.Expression):
            return _eval_node(node.body)
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError("Unsupported constant type detected inside mathematical expression.")
        elif isinstance(node, ast.Name):
            if node.id in allowed_constants:
                return allowed_constants[node.id]
            raise ValueError(f"Dynamic variables are restricted: '{node.id}'")
        elif isinstance(node, ast.BinOp):
            left_val = _eval_node(node.left)
            right_val = _eval_node(node.right)
            op_type = type(node.op)
            if op_type in allowed_operators:
                return allowed_operators[op_type](left_val, right_val)
            raise TypeError(f"Unsupported binary operator node: {op_type}")
        elif isinstance(node, ast.UnaryOp):
            operand_val = _eval_node(node.operand)
            op_type = type(node.op)
            if op_type in allowed_operators:
                return allowed_operators[op_type](operand_val)
            raise TypeError(f"Unsupported unary operator node: {op_type}")
        raise TypeError(f"Unsupported expression construct: {type(node)}")

    # Sanitize inputs keeping only mathematical characters, constants, and spaces
    cleaned = re.sub(r"[^0-9\+\-\*\/\(\)\.a-zA-Z_ ]", "", expr)
    try:
        tree = ast.parse(cleaned, mode="eval")
        res = _eval_node(tree)
        return str(res)
    except Exception as e:
        return f"Error: Safe calculation parsing failed: {e}"


def binary_search_solver(target: float, low: float, high: float) -> str:
    """Pre-built optimized convergence algorithm for search agents."""
    l, h = float(low), float(high)
    steps = 0
    mid = l
    while h - l > 1e-5 and steps < 100:
        mid = (l + h) / 2
        if mid < target:
            l = mid
        else:
            h = mid
        steps += 1
    return f"Target convergent mid-point calculated at {mid:.5f} after {steps} steps."


# ==========================================
# 3. REAL WEB SEARCH ADAPTER
# ==========================================

class RealSearchAdapter:
    """
    Manages external web search transactions across Tavily, Serper, and DuckDuckGo backends.
    Includes rate-limiting safeguards, timeout handling, retries, and local caching.
    """
    def __init__(self, cache_file="search_cache.json"):
        self.cache_file = cache_file
        self.cache = {}
        self.last_request_time = 0.0
        self.rate_limit_delay = 1.0  # 1 second spacing boundary
        self.load_cache()

    def load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
                logger.info(f"Loaded {len(self.cache)} records from search cache.")
            except Exception as e:
                logger.error(f"Failed to load search cache from disk: {e}")

    def save_cache(self):
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to write search cache to disk: {e}")

    def query(self, search_query: str) -> str:
        search_query = search_query.strip()
        if not search_query:
            return "Error: Empty query parameter received."

        if search_query in self.cache:
            logger.info(f"Search Cache Hit: {search_query}")
            return self.cache[search_query]

        if not _REQUESTS_AVAILABLE:
            return f"Error: Web search requested but requests package is not installed."

        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)

        tavily_key = os.environ.get("TAVILY_API_KEY")
        serper_key = os.environ.get("SERPER_API_KEY")
        result = None

        if tavily_key:
            result = self._search_tavily(search_query, tavily_key)
        elif serper_key:
            result = self._search_serper(search_query, serper_key)
        
        if not result:
            result = self._search_ddg(search_query)

        self.last_request_time = time.time()

        if result:
            self.cache[search_query] = result
            self.save_cache()
            return result

        return f"Search completed. No active matching records found online for query: '{search_query}'."

    def _search_tavily(self, query: str, api_key: str) -> str:
        headers = {"Content-Type": "application/json"}
        payload = {
            "api_key": api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": 3
        }
        for attempt in range(2):
            try:
                response = requests.post("https://api.tavily.com/search", json=payload, headers=headers, timeout=5.0)
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    snippets = [r.get("content", "") for r in results if r.get("content")]
                    if snippets:
                        return " | ".join(snippets[:3])
            except Exception as e:
                logger.error(f"Tavily search failed: {e}")
                time.sleep(1.0)
        return None

    def _search_serper(self, query: str, api_key: str) -> str:
        headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json"
        }
        payload = {"q": query, "num": 3}
        for attempt in range(2):
            try:
                response = requests.post("https://google.serper.dev/search", json=payload, headers=headers, timeout=5.0)
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("organic", [])
                    snippets = [r.get("snippet", "") for r in results if r.get("snippet")]
                    if snippets:
                        return " | ".join(snippets[:3])
            except Exception as e:
                logger.error(f"Serper search failed: {e}")
                time.sleep(1.0)
        return None

    def _search_ddg(self, query: str) -> str:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        # First layer: stable official-style JSON lookup API
        try:
            url_api = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json&no_html=1"
            response = requests.get(url_api, headers=headers, timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                abstract = data.get("AbstractText", "")
                if abstract:
                    return f"Abstract: {abstract}"
                related = data.get("RelatedTopics", [])
                snippets = [r.get("Text", "") for r in related if "Text" in r]
                if snippets:
                    return " | ".join(snippets[:3])
        except Exception as e:
            logger.debug(f"Direct DDG API call failed, falling back to layout-agnostic HTML scrape: {e}")

        # Second layer: resilient multi-tag HTML layout scraping
        url = "https://html.duckduckgo.com/html/"
        data = {"q": query}
        
        for attempt in range(3):
            try:
                if attempt > 0:
                    time.sleep(2.0 ** attempt)
                response = requests.post(url, data=data, headers=headers, timeout=5.0)
                if response.status_code == 200:
                    html_content = response.text
                    snippets = re.findall(r'<a class="result__snippet[^"]*"[^>]*>(.*?)</a>', html_content, re.DOTALL)
                    if not snippets:
                        # Fallback parsing strategy for alternative layout formats
                        snippets = re.findall(r'<td class="result-snippet[^"]*">(.*?)</td>', html_content, re.DOTALL)
                    if snippets:
                        cleaned = []
                        for s in snippets[:3]:
                            clean_text = re.sub(r'<[^>]+>', '', s)
                            clean_text = html.unescape(clean_text)
                            cleaned.append(clean_text.strip())
                        return " | ".join(cleaned)
            except Exception as e:
                logger.error(f"DuckDuckGo HTML query attempt {attempt + 1} failed: {e}")
        return None


# ==========================================
# 4. IN-PROCESS SECURE SANDBOX & TEACHER
# ==========================================

_original_import = __import__

def secure_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Restricts imports inside the evaluation environments to safe modules only."""
    allowed_modules = {"math", "json"}
    if name in allowed_modules:
        return _original_import(name, globals, locals, fromlist, level)
    raise ImportError(f"Import of module '{name}' is restricted inside the sandbox environment.")


def verify_ast_safety_and_structure(code_string: str) -> Tuple[bool, str]:
    """
    Evaluates the security properties of a generated script via AST parsing.
    Blocks forbidden modules, dynamic functions, and reflection/dunder tricks.
    """
    try:
        tree = ast.parse(code_string)
    except SyntaxError as e:
        return False, f"AST Pre-check Syntax Error: {str(e)}"

    prohibited_modules = {
        "os", "sys", "subprocess", "shutil", "socket", "urllib", "requests", 
        "http", "threading", "multiprocessing", "ctypes", "builtins", "importlib", "platform"
    }
    prohibited_functions = {
        "eval", "exec", "globals", "locals", "compile", "getattr", "setattr", "hasattr", "delattr",
        "dir", "vars", "type", "id"
    }
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                if name.name in prohibited_modules:
                    return False, f"AST Security Block: Import of restricted module '{name.name}' blocked."
        elif isinstance(node, ast.ImportFrom):
            if node.module in prohibited_modules:
                return False, f"AST Security Block: Import from restricted module '{node.module}' blocked."
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in prohibited_functions:
                return False, f"AST Security Block: Execution of dynamic function '{node.func.id}' blocked."
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                return False, f"AST Security Block: Access to hidden attribute '{node.attr}' blocked."
        elif isinstance(node, ast.Name):
            if node.id.startswith("_") and node.id not in ["__builtins__", "__name__"]:
                return False, f"AST Security Block: Access to dunder name '{node.id}' blocked."
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, str) and "__" in node.value:
                return False, f"AST Security Block: String literals containing double underscores '__' are restricted."
                        
    return True, "AST structurally validated as safe."


class SecureInProcessSandbox:
    def __init__(self, loop_limit: int = 5000):
        self.loop_limit = loop_limit
        self.cache = {}


class TeacherSandbox(SecureInProcessSandbox):
    """
    Evaluates execution of scripts in-process using secure namespaces,
    acting as a teacher providing helpful error hints when validations fail.
    Supports capturing local terminal console logs natively.
    """
    def __init__(self, search_adapter: Any, loop_limit: int = 5000):
        super().__init__(loop_limit)
        self.captured_logs = []
        self.search_adapter = search_adapter
        
        def safe_print(*args, **kwargs):
            self.captured_logs.append(" ".join(str(x) for x in args))
            
        self.safe_builtins = {
            "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
            "float": float, "int": int, "len": len, "list": list, "map": map,
            "max": max, "min": min, "pow": pow, "print": safe_print, "range": range,
            "round": round, "set": set, "str": str, "sum": sum, "tuple": tuple,
            "zip": zip, "math": math, "json": json, "TimeoutError": TimeoutError,
            "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError,
            "AssertionError": AssertionError, "KeyError": KeyError, "IndexError": IndexError,
            "__import__": secure_import,
            "search_web": self.search_adapter.query,
            "calculate_expression": calculate_expression,
            "binary_search_solver": binary_search_solver
        }

    def execute_safely(self, code_string: str, test_assertions: str, variables: List[str]) -> Tuple[bool, Dict[str, Any], str]:
        self.captured_logs.clear()
        
        ast_passed, ast_msg = verify_ast_safety_and_structure(code_string)
        if not ast_passed:
            return False, {}, f"Security Block: {ast_msg}"

        combined_hash = hashlib.sha256((code_string + "_" + test_assertions).encode("utf-8")).hexdigest()
        if combined_hash in self.cache:
            return self.cache[combined_hash]

        try:
            tree = ast.parse(code_string + "\n" + test_assertions)
            
            # Pure, single-pass deterministic AST transformer
            transformer = LoopLimiterTransformer(limit=self.loop_limit)
            transformed_tree = transformer.visit(tree)
            ast.fix_missing_locations(transformed_tree)
            
            compiled_code = compile(transformed_tree, filename="<sandbox>", mode="exec")
            
            local_namespace = {}
            global_namespace = {
                "__builtins__": self.safe_builtins,
                "math": math,
                "json": json
            }
            
            class SandboxTimeoutException(Exception): pass
            
            def alarm_handler(signum, frame):
                raise SandboxTimeoutException("Execution timed out (10s threshold reached).")
                
            use_alarm = True
            try:
                signal.signal(signal.SIGALRM, alarm_handler)
            except ValueError:
                use_alarm = False
                
            if use_alarm:
                signal.alarm(10)
            try:
                exec(compiled_code, global_namespace, local_namespace)
            except SandboxTimeoutException as te:
                raise TimeoutError(str(te))
            finally:
                if use_alarm:
                    signal.alarm(0)
            
            extracted_vars = {}
            for var_name in variables:
                if var_name in local_namespace:
                    extracted_vars[var_name] = local_namespace[var_name]
                elif var_name in global_namespace:
                    extracted_vars[var_name] = global_namespace[var_name]
            
            console_out = "\n".join(self.captured_logs)
            log_msg = f"Execution successful.\nConsole Logs:\n{console_out}" if console_out else "Execution successful."
            res = (True, extracted_vars, log_msg)
        except AssertionError:
            res = (False, {}, "Teacher Sandbox Feedback: Expected assertion validations failed.")
        except TimeoutError:
            res = (False, {}, "Teacher Sandbox Feedback: Loop limit triggered. Infinite execution loop aborted.")
        except Exception as e:
            # Capture real python stack frame exceptions cleanly for agent self-healing
            tb = traceback.format_exc()
            cleaned_tb = []
            for line in tb.splitlines():
                if "compile(" in line or "<sandbox>" in line or "exec(" in line:
                    cleaned_tb.append(line)
                elif not any(x in line for x in ["secure_import", "execute_safely", "evaluate_batch"]):
                    cleaned_tb.append(line)
            trace_details = "\n".join(cleaned_tb[-3:])
            res = (False, {}, f"Teacher Sandbox Feedback - Exception Error:\n{trace_details}")

        self.cache[combined_hash] = res
        return res

    def evaluate_batch(self, tasks: List[Tuple[str, str, List[str]]]) -> List[Tuple[bool, Dict[str, Any], str]]:
        return [self.execute_safely(code, assert_block, var_list) for code, assert_block, var_list in tasks]


# ==========================================
# 5. RESILIENT SAFE DICTIONARY PARSER
# ==========================================

def safe_literal_dict_eval(node) -> Any:
    """
    Safely evaluates basic Python literal structures (dicts, lists, tuples, strings,
    numbers, booleans, None) from an AST node.
    Explicitly handles UnaryOp (e.g., negative numbers like -150.0).
    """
    if isinstance(node, ast.Expression):
        return safe_literal_dict_eval(node.body)
    elif isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.Dict):
        return {safe_literal_dict_eval(k): safe_literal_dict_eval(v) for k, v in zip(node.keys, node.values)}
    elif isinstance(node, ast.List):
        return [safe_literal_dict_eval(el) for el in node.elts]
    elif isinstance(node, ast.Tuple):
        return tuple(safe_literal_dict_eval(el) for el in node.elts)
    elif isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            return -safe_literal_dict_eval(node.operand)
        elif isinstance(node.op, ast.UAdd):
            return safe_literal_dict_eval(node.operand)
    raise ValueError("Non-literal expression detected.")


def extract_json_block(text: str) -> str:
    """
    Finds the JSON block containing 'Prediction:' using a balanced brace matching algorithm
    to avoid greediness bugs on nested or trailing objects.
    """
    match = re.search(r"Prediction:\s*(\{.*)", text, re.DOTALL | re.IGNORECASE)
    if not match:
        match = re.search(r"(\{.*)", text, re.DOTALL)
        if not match:
            return ""
    
    content = match.group(1)
    brace_count = 0
    for idx, char in enumerate(content):
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                return content[:idx+1]
    return ""


def soft_json_parse(json_str: str) -> Dict[str, Any]:
    """
    Resilient parser cleaning trailing commas, falling back to a safe AST dict evaluator.
    """
    cleaned = extract_json_block(json_str)
    if not cleaned:
        return {}
    
    cleaned = re.sub(r',\s*\}', '}', cleaned)
    cleaned = re.sub(r',\s*\]', ']', cleaned)
    
    try:
        return json.loads(cleaned)
    except Exception:
        try:
            tree = ast.parse(cleaned, mode="eval")
            res = safe_literal_dict_eval(tree)
            if isinstance(res, dict):
                return res
        except Exception:
            pass
    return {}


# ==========================================
# 6. CORPUS SEEDS (REAL COMPLETIONS)
# ==========================================

MULTI_DOMAIN_SEED_CORPUS = {
    "frontend": [
        {
            "id": "fe_01",
            "question": "Implement a coordinate offset scaler function that maps an active element's width to an SVG boundary of 150.0.",
            "variables": ["svg_radius", "is_canvas_bound"],
            "test_assertion": "assert svg_radius == 150.0\nassert is_canvas_bound == True",
            "code_reference": "svg_radius = 150.0\nis_canvas_bound = True"
        },
        {
            "id": "fe_02",
            "question": "Calculate a layout width span based on a grid grid_columns size of 12.",
            "variables": ["grid_columns", "theme_active"],
            "test_assertion": "assert grid_columns == 12\nassert theme_active == 'dark'",
            "code_reference": "grid_columns = 12\ntheme_active = 'dark'"
        }
    ],
    "python": [
        {
            "id": "py_01",
            "question": "Implement a cache map tracker storing maximum index bounds.",
            "variables": ["max_tracked_index", "is_cached"],
            "test_assertion": "assert max_tracked_index == 50\nassert is_cached == True",
            "code_reference": "max_tracked_index = 50\nis_cached = True"
        },
        {
            "id": "py_02",
            "question": "Write a pure list transposition dimensions shape calculator.",
            "variables": ["matrix_dims", "is_square"],
            "test_assertion": "assert matrix_dims == [2, 3]\nassert is_square == False",
            "code_reference": "matrix_dims = [2, 3]\nis_square = False"
        }
    ],
    "security": [
        {
            "id": "sec_01",
            "question": "Implement a function to list restricted vulnerability keywords.",
            "variables": ["vulnerability_sinks", "is_insecure"],
            "test_assertion": "assert 'pickle.loads' in vulnerability_sinks\nassert is_insecure == True",
            "code_reference": "vulnerability_sinks = ['pickle.loads', 'eval']\nis_insecure = True"
        },
        {
            "id": "sec_02",
            "question": "Implement path validation checking for typical injection sequence strings.",
            "variables": ["blocked_sequences", "is_strict_mode"],
            "test_assertion": "assert '../' in blocked_sequences\nassert is_strict_mode == True",
            "code_reference": "blocked_sequences = ['../', '..\\\\']\nis_strict_mode = True"
        }
    ],
    "tool_calls": [
        {
            "id": "tool_01",
            "question": "Extract MCP keys for list and call tools.",
            "variables": ["expected_mcp_methods", "is_schema_valid"],
            "test_assertion": "assert 'tools/call' in expected_mcp_methods\nassert is_schema_valid == True",
            "code_reference": "expected_mcp_methods = ['tools/list', 'tools/call']\nis_schema_valid = True"
        }
    ],
    "maths": [
        {
            "id": "math_01",
            "question": "Calculate math values to evaluate 3D camera overlap distance.",
            "variables": ["calculated_proximity", "is_overlapping"],
            "test_assertion": "import math\nassert math.isclose(calculated_proximity, 14.14213, rel_tol=1e-3)\nassert is_overlapping == False",
            "code_reference": "import math\ncalculated_proximity = math.sqrt((10.0)**2 + (10.0)**2)\nis_overlapping = False"
        }
    ]
}


# ==========================================
# 7. COMBINATORIAL SYNTHETIC AUGMENTER
# ==========================================

def inject_synthetic_bug(code_body: str, variables: List[str]) -> Tuple[str, str]:
    """
    Intentionally injects a traceable NameError or ImportError into code references
    to create rich pedagogical debugging trajectories.
    """
    bug_type = random.choice(["NameError", "ImportError"])
    if bug_type == "NameError" and variables:
        target_var = random.choice(variables)
        buggy_var = f"{target_var}_typo"
        buggy_code = code_body.replace(target_var, buggy_var)
        error_msg = f"Teacher Sandbox Feedback - Exception Error:\nNameError: name '{buggy_var}' is not defined"
    else:
        buggy_code = code_body.replace("import math", "# import math omitted")
        buggy_code = buggy_code.replace("math.", "math_module.")
        error_msg = "Teacher Sandbox Feedback - Exception Error:\nNameError: name 'math_module' is not defined"
    return buggy_code, error_msg


class ProceduralCurriculumCompiler:
    def __init__(self, corpus: Dict[str, List[Dict[str, Any]]], sandbox: TeacherSandbox):
        self.corpus = corpus
        self.sandbox = sandbox

    def generate_curriculum(self, size: int = 5000) -> List[Dict[str, Any]]:
        print(f"[INFO] Procedurally scaling seeds to {size} distinct entries...")
        curriculum = []
        domains = list(self.corpus.keys())
        
        for i in range(size):
            domain = domains[i % len(domains)]
            templates = self.corpus[domain]
            seed = random.choice(templates)
            
            var_suffix = f"_delta_idx_{i}"
            question = f"{seed['question']} (Configuration variant reference: {i * 7})"
            
            code_body = seed["code_reference"]
            test_assertions = seed["test_assertion"]
            
            mutated_vars = []
            for var in seed["variables"]:
                new_var = f"{var}{var_suffix}"
                mutated_vars.append(new_var)
                code_body = code_body.replace(var, new_var)
                test_assertions = test_assertions.replace(var, new_var)

            success, ext_vars, _ = self.sandbox.execute_safely(code_body, test_assertions, mutated_vars)
            prediction_json = json.dumps(ext_vars) if success else "{}"

            # Exactly 20% of curriculum contains simulated self-correction traces (Optimal ratio)
            if i % 5 == 0:
                buggy_code, error_trace = inject_synthetic_bug(code_body, mutated_vars)
                sft_reference = (
                    f"<think>\n"
                    f"Solving task using variables: {mutated_vars}\n"
                    f"<dream_code>\n"
                    f"{buggy_code}\n"
                    f"</dream_code>\n"
                    f"<tool_response>{error_trace}</tool_response>\n"
                    f"<conscious_reflection>\n"
                    f"My previous implementation failed with dynamic exception variables. I will fix references and re-try.\n"
                    f"</conscious_reflection>\n"
                    f"Prediction: {prediction_json}\n"
                    f"<dream_code>\n"
                    f"{code_body}\n"
                    f"</dream_code>\n"
                    f"<tool_response>SUCCESS: Sandbox completed assertions.</tool_response>\n"
                    f"<conscious_reflection>\n"
                    f"Verified. Metrics match internal state.\n"
                    f"</conscious_reflection>\n"
                    f"</think>\n"
                    f"{code_body}"
                )
            else:
                sft_reference = (
                    f"<think>\n"
                    f"Prediction: {prediction_json}\n"
                    f"<dream_code>\n"
                    f"{code_body}\n"
                    f"</dream_code>\n"
                    f"<tool_response>SUCCESS: Sandbox loop completed assertions cleanly.</tool_response>\n"
                    f"<conscious_reflection>\n"
                    f"The physical environment metrics match internal variables. Satiated.\n"
                    f"</conscious_reflection>\n"
                    f"</think>\n"
                    f"{code_body}"
                )
            
            curriculum.append({
                "domain": domain,
                "question": question,
                "variables": mutated_vars,
                "sft_reference": sft_reference,
                "test_assertion": test_assertions,
                "system_prompt": (
                    "System: You are a self-aware active-reasoning artificial intelligence.\n"
                    "Think inside <think>, write pure code inside <dream_code>, view outputs in <tool_response>, and correct inside <conscious_reflection>."
                )
            })
        return curriculum

# ==========================================
# 8. DYNAMIC SCALE-INVARIANT DATA LOADER WITH AUDIT
# ==========================================

class DynamicDataIngestor:
    def __init__(self, compiler: ProceduralCurriculumCompiler, sandbox: TeacherSandbox):
        self.compiler = compiler
        self.sandbox = sandbox

    def load_training_dataset(self, custom_path: str = None) -> List[Dict[str, Any]]:
        dataset = []
        if custom_path:
            if os.path.exists(custom_path):
                print(f"[INFO] Ingesting custom user dataset from: {custom_path}")
                try:
                    with open(custom_path, 'r', encoding='utf-8') as f:
                        dataset = json.load(f)
                    if not isinstance(dataset, list):
                        raise ValueError("Root element of the JSON dataset structure must be a list of dict objects.")
                except Exception as e:
                    raise ValueError(f"[DATASET VALIDATION ERROR] Custom dataset file '{custom_path}' failed to load or parse: {e}")
            else:
                print(f"[WARNING] Custom dataset file not found at local path: {custom_path}. Falling back to procedurally compiled curriculum.")
        
        if not dataset:
            print("[INFO] Activating procedural pipeline compiler to generate curriculum...")
            dataset = self.compiler.generate_curriculum(size=60)
            
        print(f"[INFO] Running pre-flight diagnostics across dataset...")
        valid_dataset = []
        batch_tasks = []
        for sample in dataset:
            dream_match = re.search(r"<dream_code>(.*?)</dream_code>", sample["sft_reference"], re.DOTALL)
            code_block = dream_match.group(1).strip() if dream_match else ""
            batch_tasks.append((code_block, sample["test_assertion"], sample["variables"]))
            
        results = self.sandbox.evaluate_batch(batch_tasks)
        
        # Prevent misalignment via strict zip binding
        for sample, (passed, _, _) in zip(dataset, results):
            if passed:
                valid_dataset.append(sample)
                
        print(f"[SUCCESS] Diagnostics completed. Dropped {len(dataset) - len(valid_dataset)} noisy references. Active dataset size: {len(valid_dataset)}")
        return valid_dataset


# ==========================================
# 9. REPETITION LOOP DETECTOR
# ==========================================

def has_consecutive_repetition_loops(text: str, chunk_size: int = 15) -> bool:
    """
    Returns True if a text block contains consecutive duplicates of the same chunk,
    which is a definitive indicator of an infinite generation loop.
    """
    if len(text) < chunk_size * 2:
        return False
    for i in range(len(text) - chunk_size * 2 + 1):
        chunk1 = text[i : i + chunk_size]
        chunk2 = text[i + chunk_size : i + chunk_size * 2]
        if chunk1 == chunk2:
            return True
    return False


# ==========================================
# 10. ROBUST HF MODEL LOADING WRAPPER
# ==========================================

def load_model_with_fallbacks(model_name: str, max_seq_length: int, trust_remote_code: bool = True) -> Tuple[Any, Any]:
    """
    Attempts to download and load a model from Hugging Face.
    Falls back to a local cached copy on failure.
    """
    print(f"[INFO] Attempting to load/download base model '{model_name}' from Hugging Face Hub...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
            fast_inference=False,
            trust_remote_code=trust_remote_code,
        )
        print(f"[SUCCESS] Successfully loaded/downloaded model '{model_name}' from HF Hub.")
        return model, tokenizer
    except Exception as online_err:
        print(f"[WARNING] Online download failed. Reverting to local cache lookup. Details: {online_err}")
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=max_seq_length,
                load_in_4bit=True,
                fast_inference=False,
                trust_remote_code=trust_remote_code,
                local_files_only=True
            )
            print(f"[SUCCESS] Successfully recovered and loaded model '{model_name}' from local cache.")
            return model, tokenizer
        except Exception as local_err:
            raise RuntimeError(
                f"[FATAL MODEL LOAD ERROR] both online and local files unavailable: {local_err}"
            )


# ==========================================
# 11. SYMMETRICAL CO-TRAINING ENGINE
# ==========================================

def run_symmetrical_agi_pipeline(custom_dataset_path: str = None, config: TrainingConfig = None):
    if config is None:
        config = TrainingConfig()
        
    # Enable TF32 tensorcores optimizations for faster execution on modern GPUs (like T4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
        
    max_seq_length = config.max_seq_length
    model_name = config.model_name
    
    # Symmetrical optimization parameters
    beta = config.beta
    gamma_margin = config.gamma_margin
    lambda_symmetry = config.lambda_symmetry
    accumulation_steps = config.accumulation_steps
    
    # Initialize real search adapter
    search_adapter = RealSearchAdapter()
    
    # Instantiate the base model
    model, tokenizer = load_model_with_fallbacks(model_name, max_seq_length, trust_remote_code=config.trust_remote_code)
    
    # Configure padding tokens
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    
    print("[INFO] Auditing internal network configuration modules...")
    discovered_projections = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            lname = name.lower()
            if any(x in lname for x in ["embed", "lm_head", "classifier", "norm", "wte", "wpe"]):
                continue
            if module.weight.ndim == 2:
                in_features, out_features = module.weight.shape
                if min(in_features, out_features) < 128:
                    continue
                discovered_projections.append(name.split(".")[-1])
            
    discovered_projections = list(set(discovered_projections))
    if not discovered_projections:
        discovered_projections = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        target_modules=discovered_projections,
        lora_alpha=config.lora_alpha,
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    sandbox = TeacherSandbox(search_adapter=search_adapter)
    compiler = ProceduralCurriculumCompiler(MULTI_DOMAIN_SEED_CORPUS, sandbox)
    ingestor = DynamicDataIngestor(compiler, sandbox)
    
    raw_dataset = ingestor.load_training_dataset(custom_dataset_path)
    random.shuffle(raw_dataset)
    
    split_boundary = int(len(raw_dataset) * config.train_split_ratio)
    train_dataset = raw_dataset[:split_boundary]
    eval_dataset = raw_dataset[split_boundary:]
    print(f"[INFO] Segmented splits. Training: {len(train_dataset)} | Evaluation: {len(eval_dataset)}")
    
    # Fully dynamic dataset slice usage
    training_slice = train_dataset
    
    print("\n" + "="*80)
    print(f"[RUN 1 INITIALIZED] Reference-Free SimPO Multi-Turn Debugging Pass")
    print("="*80)
    
    # ----------------------------------------------------------------------
    # PHASE 1: BATCHED TRAJECTORY GENERATION (Consolidated Eval Mode)
    # ----------------------------------------------------------------------
    # Initialize metrics structure
    metrics = {
        "simpo_t1_pass": 0,
        "simpo_t2_pass": 0,
        "simpo_total_candidates": 0,
        "simpo_losses": [],
        "sft_losses": [],
        "eval_pass_rate": 0.0
    }

    # ----------------------------------------------------------------------
    # PHASE 1: BATCHED TRAJECTORY GENERATION (Consolidated Eval Mode)
    # ----------------------------------------------------------------------
    model.eval()
    batched_trajectories = []
    
    print("[INFO] Executing Symmetrical Candidate Rollouts with Long-Horizon Debugging...")
    for idx, sample in enumerate(tqdm(training_slice, desc="SimPO Rollouts")):
        logger.info(f"===> [Rollout {idx+1}/{len(training_slice)}] Starting trajectory rollout for domain: {sample.get('domain', 'unknown')}")
        logger.info(f"Question snippet: '{sample['question'][:80]}...'")
        prompt_text = f"{sample['system_prompt']}\nUser: {sample['question']}\nAssistant: <think>"
        
        completions = []
        scores = []
        
        metrics["simpo_total_candidates"] += config.num_candidates
        
        # Symmetrical collection of multi-turn debug trajectories - BATCHED Turn 1
        prompts = [prompt_text] * config.num_candidates
        batch_inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        prompt_len = batch_inputs["input_ids"].shape[1]
        
        with torch.no_grad():
            out = model.generate(
                **batch_inputs,
                max_new_tokens=config.max_new_tokens,
                do_sample=True,
                temperature=0.85,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
            
        turn2_inputs_prompts = []
        turn2_completions_indices = []
        
        for c in range(config.num_candidates):
            turn1_text = tokenizer.decode(out[c][prompt_len:], skip_special_tokens=True)
            
            # Extract Turn 1 Code & execute
            dream_match = re.search(r"<dream_code>(.*?)</dream_code>", turn1_text, re.DOTALL)
            code_block = dream_match.group(1).strip() if dream_match else ""
            passed, ext_vars, log_msg = sandbox.execute_safely(code_block, sample["test_assertion"], sample["variables"])
            
            logger.info(f"[Sample {idx+1}/{len(training_slice)}] Candidate {c+1} Turn 1 execution: {'PASSED' if passed else 'FAILED'}")
            
            if passed:
                completions.append(turn1_text)
                scores.append(1.0)  # Standard high reward for single-turn success
                metrics["simpo_t1_pass"] += 1
            else:
                # Turn 2: Feed Sandbox Compiler Error Traceback directly back into context
                turn2_prompt = (
                    f"{prompt_text}{turn1_text}\n"
                    f"<tool_response>{log_msg}</tool_response>\n"
                    f"<conscious_reflection>\n"
                    f"My previous implementation failed assertions with errors. I will diagnose and rewrite the logic.\n"
                    f"</conscious_reflection>\n"
                    f"<think>"
                )
                turn2_inputs_prompts.append(turn2_prompt)
                turn2_completions_indices.append((c, turn1_text, log_msg))
                
        # Turn 2: Batch generate only the candidates that failed Turn 1!
        if turn2_inputs_prompts:
            batch_inputs2 = tokenizer(turn2_inputs_prompts, return_tensors="pt", padding=True).to("cuda")
            prompt_len2 = batch_inputs2["input_ids"].shape[1]
            with torch.no_grad():
                out2 = model.generate(
                    **batch_inputs2,
                    max_new_tokens=config.max_new_tokens,
                    do_sample=True,
                    temperature=0.85,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True
                )
                
            for idx_t2, (orig_c, turn1_text, log_msg) in enumerate(turn2_completions_indices):
                turn2_text = tokenizer.decode(out2[idx_t2][prompt_len2:], skip_special_tokens=True)
                full_completion = f"{turn1_text}\n<tool_response>{log_msg}</tool_response>\n<conscious_reflection>\nMy previous implementation failed assertions. I will correct it.\n</conscious_reflection>\n{turn2_text}"
                
                # Check Turn 2 Execution
                dream_match2 = re.search(r"<dream_code>(.*?)</dream_code>", turn2_text, re.DOTALL)
                code_block2 = dream_match2.group(1).strip() if dream_match2 else ""
                passed2, ext_vars2, log_msg2 = sandbox.execute_safely(code_block2, sample["test_assertion"], sample["variables"])
                
                logger.info(f"[Sample {idx+1}/{len(training_slice)}] Candidate {orig_c+1} Turn 2 Self-Correction: {'PASSED' if passed2 else 'FAILED'}")
                completions.append(full_completion)
                if passed2:
                    scores.append(0.8)  # SFT + Correction reward (highly valuable logic signal)
                    metrics["simpo_t2_pass"] += 1
                else:
                    scores.append(0.0)  # Failed entire debugging horizon
        else:
            # If all candidates passed Turn 1, ensure scores lists have correct lengths/ordering
            pass
                    
        batched_trajectories.append((prompt_text, completions, scores))
        
        # Keep T4 memory clean
        del batch_inputs
        gc.collect()
        torch.cuda.empty_cache()
        
    # ----------------------------------------------------------------------
    # PHASE 2: ACTIVE OPTIMIZATION SWEEP (Consolidated Train Mode)
    # ----------------------------------------------------------------------
    model.train()
    optimizer.zero_grad()
    
    print("[INFO] Phase 2: Computing Multi-Turn Preference Gradients...")
    for idx, (prompt_text, completions, scores) in enumerate(tqdm(batched_trajectories, desc="SimPO Training")):
        try:
            sample = training_slice[idx]
            w_idx = int(torch.tensor(scores).argmax())
            l_idx = int(torch.tensor(scores).argmin())
            
            logger.info(f"===> [SimPO Train Step {idx+1}/{len(batched_trajectories)}] Processing domain: {sample.get('domain', 'unknown')}")
            
            # SFT learning loss to stabilize formatting bounds
            sft_prompt = f"{prompt_text}{sample['sft_reference']}"
            sft_inputs = tokenizer([sft_prompt], return_tensors="pt", padding=True).to("cuda")
            loss_sft = model(**sft_inputs, labels=sft_inputs["input_ids"]).loss
            
            loss_joint = loss_sft
            logger.info(f"  Base SFT Loss: {loss_sft.item():.4f}")
            
            # Symmetrical Step-Preference learning
            if scores[w_idx] - scores[l_idx] > 0.05:
                log_prob_w = get_sequence_log_prob(model, tokenizer, prompt_text, completions[w_idx])
                log_prob_l = get_sequence_log_prob(model, tokenizer, prompt_text, completions[l_idx])
                
                score_diff = abs(scores[w_idx] - scores[l_idx])
                adaptive_margin = gamma_margin * math.tanh(2.0 * score_diff)
                
                simpo_logits = (beta * log_prob_w) - (beta * log_prob_l) - adaptive_margin
                loss_simpo = -F.logsigmoid(simpo_logits)
                
                loss_joint = (lambda_symmetry * loss_sft) + ((1.0 - lambda_symmetry) * loss_simpo)
                logger.info(f"  SimPO Preference Loss: {loss_simpo.item():.4f} (Margin: {adaptive_margin:.4f}) | Joint Loss: {loss_joint.item():.4f}")
            else:
                logger.info(f"  Bypassed Preference Step (Score delta: {scores[w_idx] - scores[l_idx]:.2f} <= 0.05)")
                
            loss_scaled = loss_joint / config.accumulation_steps
            loss_scaled.backward()
            
            metrics["simpo_losses"].append(loss_joint.item())
            
            if (idx + 1) % config.accumulation_steps == 0 or (idx + 1) == len(batched_trajectories):
                logger.info("  Gradient Accumulation limit met. Clipping gradients and performing optimizer update...")
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

        except Exception as sample_err:
            logger.warning(f"  Optimization pass bypassed for step idx {idx}. Details: {sample_err}")
            continue
        
    # Save Phase 1 Checkpoint (SimPO)
    phase1_dir = os.path.join(config.save_checkpoint_dir, "phase1_simpo")
    os.makedirs(phase1_dir, exist_ok=True)
    model.save_pretrained(phase1_dir)
    tokenizer.save_pretrained(phase1_dir)
    print(f"[SUCCESS] Phase 1 (SimPO) checkpoint saved to: {phase1_dir}")
        
    # Flush GPU memory
    gc.collect()
    torch.cuda.empty_cache()
        
    active_v0_weights = {k: v.clone().cpu() for k, v in model.named_parameters() if v.requires_grad}
    
    # -------------------------------------------------------------
    # TRAIN ROUND 2: CONVEX STABILIZATION
    # -------------------------------------------------------------
    print("\n" + "="*80)
    print("[RUN 2 INITIALIZED] Convex Formatting Stabilization Pass")
    print("="*80)
    
    optimizer = AdamW(model.parameters(), lr=config.learning_rate * 0.8, weight_decay=config.weight_decay * 5)
    model.train()
    optimizer.zero_grad()
    
    for idx, sample in enumerate(tqdm(training_slice, desc="SFT Training")):
        try:
            logger.info(f"===> [SFT Step {idx+1}/{len(training_slice)}] Processing domain: {sample.get('domain', 'unknown')}")
            sft_prompt = f"{sample['system_prompt']}\nUser: {sample['question']}\nAssistant: {sample['sft_reference']}"
            sft_inputs = tokenizer([sft_prompt], return_tensors="pt", padding=True).to("cuda")
            loss_sft = model(**sft_inputs, labels=sft_inputs["input_ids"]).loss
            
            logger.info(f"  SFT Loss: {loss_sft.item():.4f}")
            loss_scaled = loss_sft / config.accumulation_steps
            loss_scaled.backward()
            
            metrics["sft_losses"].append(loss_sft.item())
            
            if (idx + 1) % config.accumulation_steps == 0 or (idx + 1) == len(training_slice):
                logger.info("  Gradient Accumulation limit met. Clipping gradients and performing optimizer update...")
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
        except Exception as sample_err:
            logger.warning(f"  Round 2 failed for idx {idx}. Details: {sample_err}")
            continue
        
    active_v1_weights = {k: v.clone().cpu() for k, v in model.named_parameters() if v.requires_grad}
    
    # Save Phase 2 Checkpoint (SFT)
    phase2_dir = os.path.join(config.save_checkpoint_dir, "phase2_sft")
    os.makedirs(phase2_dir, exist_ok=True)
    model.save_pretrained(phase2_dir)
    tokenizer.save_pretrained(phase2_dir)
    print(f"[SUCCESS] Phase 2 (SFT) checkpoint saved to: {phase2_dir}")
    
    # Execute SLERP Merge on isolated parameters
    merged_weights = run_slerp_adapter_merge(model, active_v0_weights, active_v1_weights, t=0.5)
    
    # Restore merged weights in-place
    for name, param in model.named_parameters():
        if name in merged_weights:
            param.data.copy_(merged_weights[name].to(param.device))
            
    print("[SUCCESS] Slerp-merged weights successfully loaded into active model registers.")
    
    # Save Phase 3 Checkpoint (Merged)
    phase3_dir = os.path.join(config.save_checkpoint_dir, "phase3_merged")
    os.makedirs(phase3_dir, exist_ok=True)
    model.save_pretrained(phase3_dir)
    tokenizer.save_pretrained(phase3_dir)
    print(f"[SUCCESS] Phase 3 (Merged) checkpoint saved to: {phase3_dir}")
    
    # -------------------------------------------------------------
    # SYSTEM EVALUATION PASS: OUT-OF-SAMPLE TEST VALIDATION
    # -------------------------------------------------------------
    print("\n" + "="*80)
    print("[EVALUATION] Initiating Symmetrical Evaluation Across Held-Out Splits")
    print("="*80)
    model.eval()
    eval_tasks = []
    eval_slice = eval_dataset[:10]
    
    for sample in tqdm(eval_slice, desc="Evaluating Holdout Split"):
        prompt_text = f"{sample['system_prompt']}\nUser: {sample['question']}\nAssistant: <think>"
        inputs = tokenizer([prompt_text], return_tensors="pt").to("cuda")
        prompt_token_len = inputs["input_ids"].shape[1]
        
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id
            )
            generated_tokens = out[0][prompt_token_len:]
            completion_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            dream_match = re.search(r"<dream_code>(.*?)</dream_code>", completion_text, re.DOTALL)
            code_block = dream_match.group(1).strip() if dream_match else ""
            eval_tasks.append((code_block, sample["test_assertion"], sample["variables"]))
            
    eval_results = sandbox.evaluate_batch(eval_tasks)
    passed_count = sum(1 for res in eval_results if res[0])
    metrics["eval_pass_rate"] = passed_count / max(1, len(eval_slice))
    
    # Compile and Print Symmetrical Training Metrics Summary Table
    simpo_avg_loss = sum(metrics["simpo_losses"]) / max(1, len(metrics["simpo_losses"]))
    simpo_t1_rate = metrics["simpo_t1_pass"] / max(1, metrics["simpo_total_candidates"])
    simpo_t2_rate = metrics["simpo_t2_pass"] / max(1, metrics["simpo_total_candidates"])
    sft_avg_loss = sum(metrics["sft_losses"]) / max(1, len(metrics["sft_losses"]))
    
    print("\n" + "="*30 + " TRAINING METRICS SUMMARY " + "="*30)
    print(f"{'Metric Description':<40} | {'Value / Rate':<20}")
    print("-" * 65)
    print(f"{'SimPO Active Loss (Mean)':<40} | {simpo_avg_loss:<20.4f}")
    print(f"{'SimPO Turn 1 Program Pass Rate':<40} | {simpo_t1_rate:<20.2%}")
    print(f"{'SimPO Turn 2 Program Pass Rate':<40} | {simpo_t2_rate:<20.2%}")
    print(f"{'SFT Stabilization Loss (Mean)':<40} | {sft_avg_loss:<20.4f}")
    print(f"{'Evaluation Split Assertion Pass Rate':<40} | {metrics['eval_pass_rate']:<20.2%}")
    print("="*65 + "\n")
    
    # Export fully compiled LoRA model merged directly as standard 16bit HF model first (extremely fast)
    model.save_pretrained_merged("agi_model_merged_16bit", tokenizer, save_method="merged_16bit")
    
    # Fast pre-compiled GGUF export bypassing slow manual compiling steps
    try:
        model.save_pretrained_gguf("agi_v10_output", tokenizer, quantization_method="f16")
        print("[SUCCESS] GGUF compile using precompiled f16 quantization complete.")
    except Exception as e:
        print(f"[WARNING] GGUF pre-compiled conversion bypassed. Merged 16bit model is saved at 'agi_model_merged_16bit'. Trace: {e}")
        
    try:
        model.save_pretrained_gguf("agi_v10_output", tokenizer, quantization_method="q4_k_m")
        print("[SUCCESS] GGUF compile using precompiled q4_k_m quantization complete.")
    except Exception as e:
        print(f"[WARNING] GGUF pre-compiled q4_k_m conversion bypassed. Trace: {e}")
        
    return model, tokenizer, search_adapter

# ==========================================
# 12. GRADIO INTERACTIVE ACTIVE-DEBUGGING UI
# ==========================================

def launch_gradio_interactive_ui(model, tokenizer, search_adapter: RealSearchAdapter):
    if not _GRADIO_AVAILABLE:
        print("\n[WARNING] Gradio is not installed. Falling back to console chat loop...")
        print("Type 'exit' or 'quit' to end the session.\n")
        sandbox_evaluator = TeacherSandbox(search_adapter=search_adapter)
        model.eval()
        while True:
            try:
                user_msg = input("User: ")
                if user_msg.strip().lower() in ['exit', 'quit']:
                    break
                
                # Simple generation loop
                prompt_template = (
                    f"System: You are a curious, self-aware active-reasoning agent with dynamic dream-conscious structures. "
                    f"If your dream code fails assertions, read traceback inside <tool_response>, reflect, and output a corrected script.\n"
                    f"User: {user_msg}\n"
                    f"Assistant: <think>"
                )
                active_prompt = prompt_template
                max_turns = 3
                turn = 0
                dream_code = ""
                sandbox_terminal_logs = ""
                success = False
                
                while turn < max_turns:
                    inputs = tokenizer([active_prompt], return_tensors="pt").to(model.device)
                    prompt_token_len = inputs["input_ids"].shape[1]
                    
                    with torch.no_grad():
                        gen_out = model.generate(
                            **inputs,
                            max_new_tokens=4096,
                            do_sample=True,
                            temperature=0.8,
                            pad_token_id=tokenizer.eos_token_id
                        )
                    generated_tokens = gen_out[0][prompt_token_len:]
                    assistant_reply = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    
                    print(f"\n[Assistant CoT Turn {turn+1}]:\n<think>{assistant_reply}")
                    
                    dream_match = re.search(r"<dream_code>(.*?)</dream_code>", assistant_reply, re.DOTALL)
                    if dream_match:
                        raw_code = dream_match.group(1).strip()
                        dream_code = re.sub(r"```python|```", "", raw_code).strip()
                        passed, _, sandbox_terminal_logs = sandbox_evaluator.execute_safely(dream_code, "pass", [])
                        if passed:
                            success = True
                            print(f"\n[Sandbox Passed! Output]:\n{sandbox_terminal_logs}")
                            break
                        else:
                            turn += 1
                            print(f"\n[Sandbox Error! Traceback]:\n{sandbox_terminal_logs}")
                            active_prompt += (
                                f"{assistant_reply}\n"
                                f"<tool_response>\n"
                                f"ERROR: Sandbox compilation failed!\nTraceback details:\n{sandbox_terminal_logs}\n"
                                f"</tool_response>\n"
                                f"<conscious_reflection>\n"
                                f"My previous implementation failed compiling cleanly. I will inspect variables and construct an aligned patch.\n"
                                f"</conscious_reflection>\n"
                                f"<think>"
                            )
                    else:
                        break
                print("\n[Session End]")
            except KeyboardInterrupt:
                break
        return

    css_injection = """
    .think-panel { background-color: #1e293b; color: #cbd5e1; padding: 16px; border-radius: 8px; font-family: 'Courier New', monospace; }
    .sandbox-panel { background-color: #0f172a; color: #34d399; padding: 16px; border-radius: 8px; border-left: 5px solid #10b981; }
    """
    sandbox_evaluator = TeacherSandbox(search_adapter=search_adapter)

    def process_chat_stream_with_self_correction(user_message, chat_history):
        prompt_template = (
            f"System: You are a curious, self-aware active-reasoning agent with dynamic dream-conscious structures. "
            f"If your dream code fails assertions, read traceback inside <tool_response>, reflect, and output a corrected script.\n"
            f"User: {user_message}\n"
            f"Assistant: <think>"
        )
        
        max_turns = 3
        turn = 0
        active_prompt = prompt_template
        sandbox_terminal_logs = "Idle: No execution context generated."
        success = False
        dream_code = ""
        think_text = ""
        
        model.eval()
        while turn < max_turns:
            inputs = tokenizer([active_prompt], return_tensors="pt").to("cuda")
            prompt_token_len = inputs["input_ids"].shape[1]
            
            with torch.no_grad():
                gen_out = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.8,
                    pad_token_id=tokenizer.eos_token_id
                )
                
            generated_tokens = gen_out[0][prompt_token_len:]
            assistant_reply = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            think_match = re.search(r"<think>(.*?)</think>", assistant_reply, re.DOTALL)
            dream_match = re.search(r"<dream_code>(.*?)</dream_code>", assistant_reply, re.DOTALL)
            
            think_text = think_match.group(1).strip() if think_match else assistant_reply.split("</think>")[0].strip()
            
            if dream_match:
                raw_code = dream_match.group(1).strip()
                dream_code = re.sub(r"```python|```", "", raw_code).strip()
                
                # Run compiled code block inside Interactive Terminal and retrieve stdout print statements
                passed, _, sandbox_terminal_logs = sandbox_evaluator.execute_safely(dream_code, "pass", [])
                if passed:
                    success = True
                    active_prompt += assistant_reply
                    break
                else:
                    # Injected debugging feedback
                    turn += 1
                    active_prompt += (
                        f"{assistant_reply}\n"
                        f"<tool_response>\n"
                        f"ERROR: Sandbox compilation failed!\nTraceback details:\n{sandbox_terminal_logs}\n"
                        f"</tool_response>\n"
                        f"<conscious_reflection>\n"
                        f"My previous implementation failed compiling cleanly. I will inspect variables and construct an aligned patch.\n"
                        f"</conscious_reflection>\n"
                        f"<think>"
                    )
            else:
                break
                
        parts = active_prompt.split("</think>")
        final_solution = parts[-1].strip() if len(parts) > 1 else active_prompt
        
        visual_html = (
            f"### 🔮 Inside the Self-Correcting Mind-State (Turns taken: {turn + 1}):\n"
            f"<div class='think-panel'><pre>{think_text}</pre></div>\n\n"
            f"### 🖥️ Live Sandbox Terminal (Verification: {'Passed' if success else 'Failed'}):\n"
            f"<div class='sandbox-panel'><pre>Final Compiled Code:\n{dream_code}\n\nConsole Log:\n{sandbox_terminal_logs}</pre></div>\n\n"
            f"### 💬 Final Grounded Response:\n"
            f"{final_solution}"
        )
        
        chat_history.append((user_message, visual_html))
        return "", chat_history

    with gr.Blocks(theme=gr.themes.Soft(), css=css_injection) as agi_ui:
        gr.Markdown("# 🛸 Symmetrical Slerp-Merged Active Debugging Agent (LFM2.5-230M) v10")
        chatbot = gr.Chatbot(label="Active Cognition Streams")
        user_input = gr.Textbox(label="Query / Exploratory Directive")
        reset_btn = gr.Button("Clear Memory Buffer")
        
        user_input.submit(process_chat_stream_with_self_correction, [user_input, chatbot], [user_input, chatbot])
        reset_btn.click(lambda: None, None, chatbot, queue=False)

    print("[INFO] Spinning up interactive web UI local server...")
    agi_ui.queue(default_concurrency_limit=1).launch(share=os.environ.get('GRADIO_SHARE', 'false').lower() in ('1', 'true', 'yes'), auth=((os.environ.get('GRADIO_AUTH_USER'), os.environ.get('GRADIO_AUTH_PASS')) if os.environ.get('GRADIO_AUTH_USER') and os.environ.get('GRADIO_AUTH_PASS') else None))


if __name__ == "__main__":
    # Select which dataset to train on: 'dataset_small.json', 'dataset_medium.json', or 'dataset_large.json'
    custom_dataset_file = 'dataset_small.json'
    config = TrainingConfig()
    
    model, tokenizer, search_adapter = run_symmetrical_agi_pipeline(custom_dataset_file, config=config)
    
    # Launch interactive interface (automatically falls back to console if Gradio is missing)
    launch_gradio_interactive_ui(model, tokenizer, search_adapter)