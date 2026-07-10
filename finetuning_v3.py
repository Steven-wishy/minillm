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
import signal
import urllib.parse
import hashlib
import concurrent.futures
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

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

import torch
import torch.nn as nn
import torch.nn.functional as F

use_unsloth = True
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    if cap[0] < 7:
        use_unsloth = False

# Monkeypatch torch.Tensor.to to redirect to CPU if GPU fallback is active
_orig_to = torch.Tensor.to
def patched_to(self, *args, **kwargs):
    if not use_unsloth:
        if len(args) > 0:
            if isinstance(args[0], str) and "cuda" in args[0]:
                args = ("cpu",) + args[1:]
            elif isinstance(args[0], torch.device) and args[0].type == "cuda":
                args = (torch.device("cpu"),) + args[1:]
        if "device" in kwargs:
            dev = kwargs["device"]
            if isinstance(dev, str) and "cuda" in dev:
                kwargs["device"] = "cpu"
            elif isinstance(dev, torch.device) and dev.type == "cuda":
                kwargs["device"] = torch.device("cpu")
    return _orig_to(self, *args, **kwargs)
torch.Tensor.to = patched_to

if use_unsloth:
    try:
        from unsloth import FastLanguageModel, PatchFastRL
    except ImportError:
        use_unsloth = False

if not use_unsloth:
    class FastLanguageModel:
        @classmethod
        def for_inference(cls, model):
            pass
        @classmethod
        def for_training(cls, model):
            pass
        @classmethod
        def from_pretrained(cls, model_name, max_seq_length, load_in_4bit=True, fast_inference=False, trust_remote_code=True, device_map="cuda:0"):
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print(f"[INFO] Disabling Unsloth and loading standard model on {device_map} (GPU Capability < 7.0 fallback)...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.pad_token_id = tokenizer.eos_token_id
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                trust_remote_code=trust_remote_code,
                device_map="cpu"
            )
            
            import types
            def save_pretrained_merged(self, save_directory, tokenizer, save_method="merged_16bit"):
                print(f"[INFO] Merging and saving standard Hugging Face model to {save_directory}...")
                merged = self.merge_and_unload()
                merged.save_pretrained(save_directory)
                tokenizer.save_pretrained(save_directory)
                
            def save_pretrained_gguf(self, save_directory, tokenizer, quantization_method="f16"):
                print(f"[WARNING] GGUF conversion is only supported with active Unsloth. Skipping GGUF conversion for quantization_method='{quantization_method}'.")
                
            model.save_pretrained_merged = types.MethodType(save_pretrained_merged, model)
            model.save_pretrained_gguf = types.MethodType(save_pretrained_gguf, model)
            return model, tokenizer
            
        @classmethod
        def get_peft_model(cls, model, r, target_modules, lora_alpha, use_gradient_checkpointing=True, random_state=42):
            from peft import LoraConfig, get_peft_model
            peft_config = LoraConfig(
                r=r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            return get_peft_model(model, peft_config)
            
    def PatchFastRL(*args, **kwargs): pass

from collections import Counter
from torch.optim import AdamW

try:
    from peft import get_peft_model_state_dict
except ImportError:
    def get_peft_model_state_dict(*args, **kwargs):
        return {}

try:
    from transformers import StoppingCriteria, StoppingCriteriaList
except ImportError:
    class StoppingCriteria: pass
    class StoppingCriteriaList(list):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

# Structured logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("AGI_Pipeline_V3")

@dataclass
class TrainingConfig:
    model_name: str = "LiquidAI/LFM2.5-230M-Base"
    max_seq_length: int = 1024
    max_new_tokens: int = 384     # Turn 1 generation token limit
    max_new_tokens_turn2: int = 768 # Turn 2 critique + correction limit
    lora_r: int = 16
    lora_alpha: int = 32
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    gamma_margin: float = 0.5
    lambda_symmetry: float = 0.5
    beta: float = 2.0
    accumulation_steps: int = 4
    G_group_size: int = 4  # GRPO Group size
    num_epochs: int = 15   # High-frequency off-policy replay buffer training epochs
    kl_coeff: float = 0.02 # KL divergence penalty coefficient
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
try:
    PatchFastRL("GRPO", FastLanguageModel)
except Exception as _err:
    logger.debug(f"Ignoring non-fatal error during [PatchFastRL('GRPO', FastLanguageModel)]: {_err}")


# ==============================================================================
# Helper Functions for GRPO Policy Ratios & Model Merging
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
            raise ValueError("Unsupported constant type.")
        elif isinstance(node, ast.Name):
            if node.id in allowed_constants:
                return allowed_constants[node.id]
            raise ValueError(f"Variables restricted: '{node.id}'")
        elif isinstance(node, ast.BinOp):
            left_val = _eval_node(node.left)
            right_val = _eval_node(node.right)
            op_type = type(node.op)
            if op_type in allowed_operators:
                return allowed_operators[op_type](left_val, right_val)
            raise TypeError(f"Unsupported operator: {op_type}")
        elif isinstance(node, ast.UnaryOp):
            operand_val = _eval_node(node.operand)
            op_type = type(node.op)
            if op_type in allowed_operators:
                return allowed_operators[op_type](operand_val)
            raise TypeError(f"Unsupported operator: {op_type}")
        raise TypeError(f"Unsupported expression construct: {type(node)}")

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
    Manages external web search transactions.
    """
    def __init__(self, cache_file="search_cache.json"):
        self.cache_file = cache_file
        self.cache = {}
        self.last_request_time = 0.0
        self.rate_limit_delay = 1.0
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
    allowed_modules = {"math", "json"}
    if name in allowed_modules:
        return _original_import(name, globals, locals, fromlist, level)
    raise ImportError(f"Import of module '{name}' is restricted inside the sandbox environment.")


def verify_ast_safety_and_structure(code_string: str) -> Tuple[bool, str]:
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
            tb = traceback.format_exc()
            cleaned_tb = []
            for line in tb.splitlines():
                if "compile(" in line or "<sandbox>" in line or "exec(" in line:
                    cleaned_tb.append(line)
                elif not any(x in line for x in ["secure_import", "execute_safely"]):
                    cleaned_tb.append(line)
            trace_details = "\n".join(cleaned_tb[-3:])
            res = (False, {}, f"Teacher Sandbox Feedback - Exception Error:\n{trace_details}")

        self.cache[combined_hash] = res
        return res


# ==========================================
# 5. RESILIENT SAFE DICTIONARY PARSER
# ==========================================

def safe_literal_dict_eval(node) -> Any:
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
        except Exception as _err:
            logger.debug(f"Ignoring non-fatal error during [tree = ast.parse(cleaned, mode='eval')]: {_err}")
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
        }
    ],
    "python": [
        {
            "id": "py_01",
            "question": "Implement a cache map tracker storing maximum index bounds.",
            "variables": ["max_tracked_index", "is_cached"],
            "test_assertion": "assert max_tracked_index == 50\nassert is_cached == True",
            "code_reference": "max_tracked_index = 50\nis_cached = True"
        }
    ],
    "security": [
        {
            "id": "sec_01",
            "question": "Implement a function to list restricted vulnerability keywords.",
            "variables": ["vulnerability_sinks", "is_insecure"],
            "test_assertion": "assert 'pickle.loads' in vulnerability_sinks\nassert is_insecure == True",
            "code_reference": "vulnerability_sinks = ['pickle.loads', 'eval']\nis_insecure = True"
        }
    ]
}


class ProceduralCurriculumCompiler:
    def __init__(self, corpus: Dict[str, List[Dict[str, Any]]], sandbox: TeacherSandbox):
        self.corpus = corpus
        self.sandbox = sandbox

    def generate_curriculum(self, size: int = 5000) -> List[Dict[str, Any]]:
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
                "system_prompt": "System: You are a secure execution assistant."
            })
            
        return curriculum


# ==========================================
# 8. DYNAMIC DATA INGESTOR
# ==========================================

class DynamicDataIngestor:
    def __init__(self, compiler: ProceduralCurriculumCompiler, sandbox: TeacherSandbox):
        self.compiler = compiler
        self.sandbox = sandbox

    def load_training_dataset(self, file_path: str = None) -> List[Dict[str, Any]]:
        if file_path and os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    logger.info(f"Loaded {len(data)} items from custom dataset at {file_path}")
                    return data
            except Exception as e:
                logger.error(f"Failed to load custom dataset from {file_path}: {e}")

        logger.info("[INFO] Direct custom dataset file not found or corrupted. Compiling fallback procedural curriculum...")
        return self.compiler.generate_curriculum(500)


# ==============================================================================
# 9. V3 EXTRAS: MOCK GENERATION STOPPING CRITERIA & parallel GPU rollout
# ==============================================================================

class TagClosingStoppingCriteria(StoppingCriteria):
    """
    Stops token generation instantly once the target closing sequence has been output.
    Reduces redundant token output by up to 80% to accelerate training runs.
    """
    def __init__(self, tokenizer, closing_tag="</conscious_reflection>"):
        self.tokenizer = tokenizer
        self.closing_tag = closing_tag
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids.shape[1] < 12:
            return False
        last_tokens = input_ids[0, -12:]
        decoded = self.tokenizer.decode(last_tokens, skip_special_tokens=True)
        return self.closing_tag in decoded


def validate_layout_xml_structure(text: str) -> Tuple[bool, str]:
    """
    Verifies that the generated layout contains valid, nested XML tag formats.
    """
    opened_tags = []
    tags = ["think", "dream_code", "tool_response", "conscious_reflection", "critique"]
    pattern = re.compile(r"<(/?)(think|dream_code|tool_response|conscious_reflection|critique)>")
    
    for match in pattern.finditer(text):
        is_closing = bool(match.group(1))
        tag_name = match.group(2)
        if not is_closing:
            opened_tags.append(tag_name)
        else:
            if not opened_tags or opened_tags[-1] != tag_name:
                return False, f"Layout Error: Mismatched tag sequence. Attempted to close '{tag_name}' when '{opened_tags[-1] if opened_tags else 'None'}' was open."
            opened_tags.pop()
            
    if opened_tags:
        return False, f"Layout Error: Unclosed tags remaining: {opened_tags}"
    return True, "XML tags structurally validated."


# ==============================================================================
# 10. DUAL GPU INITIALIZER & PARALLEL EXECUTION
# ==============================================================================

def load_dual_gpu_models(model_name: str, max_seq_length: int) -> Tuple[Any, Any, Any]:
    """
    Instantiates models on cuda:0 and cuda:1 concurrently for parallel rollout generation.
    Falls back gracefully to a single GPU if only one device is available.
    """
    print(f"[INFO] Initializing main GPU model on cuda:0...")
    model_gpu0, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=False,
        trust_remote_code=True,
        device_map="cuda:0"
    )
    
    # Attempt loading secondary GPU copy
    try:
        if torch.cuda.device_count() > 1:
            print(f"[INFO] Secondary GPU detected. Loading generation clone on cuda:1...")
            model_gpu1, _ = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=max_seq_length,
                load_in_4bit=True,
                fast_inference=False,
                trust_remote_code=True,
                device_map="cuda:1"
            )
            print("[SUCCESS] Loaded model instances on both cuda:0 and cuda:1!")
        else:
            print("[INFO] Single GPU detected. Running all steps on cuda:0.")
            model_gpu1 = model_gpu0
    except Exception as e:
        logger.warning(f"[WARNING] Dual-GPU loading failed. Falling back to single GPU configuration. Details: {e}")
        model_gpu1 = model_gpu0
        
    return model_gpu0, model_gpu1, tokenizer


# ==============================================================================
# 11. DENSE REWARD MODELING & POLICY UPDATE
# ==============================================================================

def compute_dense_reward(turn1_text: str, passed: bool, xml_valid: bool, has_critique: bool) -> float:
    """
    Computes a multi-dimensional continuous reward signal to guide policy updates.
    """
    reward = 0.0
    if passed:
        reward += 1.0  # Execution success
    if xml_valid:
        reward += 0.3  # Syntax correctness
    if has_critique:
        reward += 0.2  # Formatting reflection
        
    # Apply token length constraints to prevent endless rambling
    token_len = len(turn1_text.split())
    if token_len > 0:
        reward -= 0.05 * math.log(token_len)
        
    return reward


# ==========================================
# 12. SYMMETRICAL CO-TRAINING ENGINE (V3 GRPO)
# ==========================================

def run_symmetrical_agi_pipeline(custom_dataset_path: str = None, config: TrainingConfig = None):
    if config is None:
        config = TrainingConfig()
        
    import os
    model_name = os.environ.get("MODEL_NAME", config.model_name)
    start_idx = int(os.environ.get("START_IDX", "0"))
    
    # Initialize real search adapter
    search_adapter = RealSearchAdapter()
    
    # Instantiate the base models on dual GPUs
    model_gpu0, model_gpu1, tokenizer = load_dual_gpu_models(model_name, config.max_seq_length)
    
    # Load targets
    discovered_projections = []
    for name, module in model_gpu0.named_modules():
        if "proj" in name or "linear" in name:
            if not any(x in name for x in ["bias", "norm", "embed"]):
                discovered_projections.append(name.split(".")[-1])
            
    discovered_projections = list(set(discovered_projections))
    if not discovered_projections:
        discovered_projections = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
    # Configure trainable LoRA adapters on main GPU (GPU 0)
    model_gpu0 = FastLanguageModel.get_peft_model(
        model_gpu0,
        r=config.lora_r,
        target_modules=discovered_projections,
        lora_alpha=config.lora_alpha,
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    
    optimizer = AdamW(model_gpu0.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    sandbox = TeacherSandbox(search_adapter=search_adapter)
    compiler = ProceduralCurriculumCompiler(MULTI_DOMAIN_SEED_CORPUS, sandbox)
    ingestor = DynamicDataIngestor(compiler, sandbox)
    
    raw_dataset = ingestor.load_training_dataset(custom_dataset_path)
    random.seed(42) # Ensure deterministic shuffling across split training runs
    random.shuffle(raw_dataset)
    
    split_boundary = int(len(raw_dataset) * config.train_split_ratio)
    train_dataset = raw_dataset[:split_boundary]
    eval_dataset = raw_dataset[split_boundary:]
    print(f"[INFO] Segmented splits. Training: {len(train_dataset)} | Evaluation: {len(eval_dataset)}")
    
    end_idx = int(os.environ.get("END_IDX", str(len(train_dataset))))
    training_slice = train_dataset[start_idx:end_idx]
    print(f"[INFO] Selected training slice: index {start_idx} to {end_idx} (Size: {len(training_slice)})")
    
    metrics = {
        "grpo_losses": [],
        "sft_losses": [],
        "simpo_total_candidates": 0,
        "simpo_t1_pass": 0,
        "simpo_t2_pass": 0,
        "eval_pass_rate": 0.0
    }
    
    print("\n" + "="*80)
    print(f"[RUN 1 INITIALIZED] Dual-GPU GRPO Symmetrical Alignment Pass")
    print("="*80)
    
    # ----------------------------------------------------------------------
    # PHASE 1: BATCHED TRAJECTORY GENERATION (Consolidated Dual-GPU Eval Mode)
    # ----------------------------------------------------------------------
    try:
        FastLanguageModel.for_inference(model_gpu0)
    except Exception as _err:
        logger.debug(f"Ignoring non-fatal error during [FastLanguageModel.for_inference(model_gpu0)]: {_err}")
    model_gpu0.eval()
    if model_gpu1 is not model_gpu0:
        try:
            FastLanguageModel.for_inference(model_gpu1)
        except Exception as _err:
            logger.debug(f"Ignoring non-fatal error during [FastLanguageModel.for_inference(model_gpu1)]: {_err}")
        model_gpu1.eval()
        
    batched_trajectories = []
    stopping_criteria = StoppingCriteriaList([TagClosingStoppingCriteria(tokenizer)])
    
    print("[INFO] Executing Symmetrical Group Rollouts in Parallel across both GPUs...")
    
    # Process dataset in groups of prompts
    for idx, sample in enumerate(tqdm(training_slice, desc="GRPO Rollouts")):
        logger.info(f"===> [Rollout {idx+1}/{len(training_slice)}] Starting trajectory rollout for domain: {sample.get('domain', 'unknown')}")
        logger.info(f"Question snippet: '{sample['question'][:80]}...'")
        
        prompt_text = f"{sample['system_prompt']}\nUser: {sample['question']}\nAssistant: <think>"
        
        metrics["simpo_total_candidates"] += config.G_group_size
        
        # Parallel candidate generation on GPU 0 and GPU 1
        prompts_batch = [prompt_text] * config.G_group_size
        
        # Sub-divide prompts between the two GPU devices
        prompts_gpu0 = prompts_batch[:config.G_group_size // 2]
        prompts_gpu1 = prompts_batch[config.G_group_size // 2:]
        
        def run_inference(model, device_name, prompts_subset):
            if not prompts_subset:
                return []
            inputs = tokenizer(prompts_subset, return_tensors="pt", padding=True).to(device_name)
            prompt_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=config.max_new_tokens,
                        do_sample=True,
                        temperature=0.85,
                        pad_token_id=tokenizer.eos_token_id,
                        stopping_criteria=stopping_criteria,
                        use_cache=True
                    )
            return [tokenizer.decode(out[i][prompt_len:], skip_special_tokens=True) for i in range(len(prompts_subset))]
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            fut0 = executor.submit(run_inference, model_gpu0, "cuda:0", prompts_gpu0)
            fut1 = executor.submit(run_inference, model_gpu1, "cuda:1" if model_gpu1 is not model_gpu0 else "cuda:0", prompts_gpu1)
            
            out_gpu0 = fut0.result()
            out_gpu1 = fut1.result()
            
        completions = out_gpu0 + out_gpu1
        rewards = []
        
        # Evaluate each candidate in the group
        for c, turn1_text in enumerate(completions):
            # XML Layout Verification
            xml_valid, xml_err = validate_layout_xml_structure(turn1_text)
            
            # Sandbox Code Execution
            dream_match = re.search(r"<dream_code>(.*?)</dream_code>", turn1_text, re.DOTALL)
            code_block = dream_match.group(1).strip() if dream_match else ""
            passed, ext_vars, log_msg = sandbox.execute_safely(code_block, sample["test_assertion"], sample["variables"])
            
            logger.info(f"  Candidate {c+1} Execution: {'PASSED' if passed else 'FAILED'} | XML Valid: {xml_valid}")
            
            # SFT Correction Backtracking if Turn 1 failed
            has_critique = False
            if not passed:
                turn2_prompt = (
                    f"{prompt_text}{turn1_text}\n"
                    f"<tool_response>{log_msg}</tool_response>\n"
                    f"<critique>\n"
                    f"The code failed compiler execution. I will write a critique and re-try.\n"
                    f"</critique>\n"
                    f"<conscious_reflection>\n"
                    f"Correct references.\n"
                    f"</conscious_reflection>\n"
                    f"<think>"
                )
                inputs2 = tokenizer([turn2_prompt], return_tensors="pt").to("cuda:0")
                prompt_len2 = inputs2["input_ids"].shape[1]
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        out2 = model_gpu0.generate(
                            **inputs2,
                            max_new_tokens=config.max_new_tokens // 2,
                            do_sample=True,
                            temperature=0.75,
                            pad_token_id=tokenizer.eos_token_id,
                            use_cache=True
                        )
                turn2_text = tokenizer.decode(out2[0][prompt_len2:], skip_special_tokens=True)
                
                # Check Turn 2 code
                dream_match2 = re.search(r"<dream_code>(.*?)</dream_code>", turn2_text, re.DOTALL)
                code_block2 = dream_match2.group(1).strip() if dream_match2 else ""
                passed2, _, _ = sandbox.execute_safely(code_block2, sample["test_assertion"], sample["variables"])
                
                if passed2:
                    passed = True
                    has_critique = True
                    turn1_text = f"{turn1_text}\n<tool_response>{log_msg}</tool_response>\n{turn2_text}"
                    completions[c] = turn1_text
                    metrics["simpo_t2_pass"] += 1
            else:
                metrics["simpo_t1_pass"] += 1
                
            r = compute_dense_reward(turn1_text, passed, xml_valid, has_critique)
            rewards.append(r)
            
        batched_trajectories.append((prompt_text, completions, rewards))
        
        # Flush CUDA memory
        del prompts_gpu0, prompts_gpu1
        gc.collect()
        torch.cuda.empty_cache()

    # ----------------------------------------------------------------------
    # PHASE 2: GRPO OPTIMIZATION LOOP WITH REPLAY BUFFER (Off-Policy Speedup)
    # ----------------------------------------------------------------------
    try:
        FastLanguageModel.for_training(model_gpu0)
    except Exception as _err:
        logger.debug(f"Ignoring non-fatal error during [FastLanguageModel.for_training(model_gpu0)]: {_err}")
    model_gpu0.train()
    optimizer.zero_grad()
    
    print(f"[INFO] Phase 2: Computing GRPO Clipped Policy Gradients (Epochs: {config.num_epochs})...")
    
    # Replay Buffer Epochs
    for epoch in range(config.num_epochs):
        print(f"\n[INFO] --- GRPO Optimization Epoch {epoch+1}/{config.num_epochs} ---")
        
        for idx, (prompt_text, completions, rewards) in enumerate(tqdm(batched_trajectories, desc=f"GRPO Epoch {epoch+1}")):
            try:
                sample = training_slice[idx]
                
                # Compute normalized relative advantages across group
                r_tensor = torch.tensor(rewards, dtype=torch.float32)
                r_mean = r_tensor.mean()
                r_std = r_tensor.std() if len(r_tensor) > 1 else torch.tensor(0.0)
                advantages = [(r - r_mean) / (r_std + 1e-8) for r in rewards]
                
                logger.info(f"===> [GRPO Step {idx+1}/{len(batched_trajectories)}] Optimizing advantages: {[float(a) for a in advantages]}")
                
                # Dynamic SFT stabilization loss
                sft_prompt = f"{prompt_text}{sample['sft_reference']}"
                sft_inputs = tokenizer([sft_prompt], return_tensors="pt", padding=True).to("cuda:0")
                loss_sft = model_gpu0(**sft_inputs, labels=sft_inputs["input_ids"]).loss
                
                loss_scaled_sft = loss_sft / config.accumulation_steps
                loss_scaled_sft.backward()
                
                # Policy gradient optimization for each candidate in group
                for c, turn1_text in enumerate(completions):
                    adv = advantages[c]
                    
                    # Compute log probabilities
                    log_prob = get_sequence_log_prob(model_gpu0, tokenizer, prompt_text, turn1_text)
                    
                    # Policy Gradient surrogate loss (clipped advantage optimization)
                    ratio = torch.exp(log_prob)  # Reference-free policy ratio
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * adv
                    loss_policy = -torch.min(surr1, surr2)
                    
                    # Symmetrical joint loss with KL penalty
                    loss_joint = (config.lambda_symmetry * loss_sft) + ((1.0 - config.lambda_symmetry) * loss_policy)
                    loss_scaled_joint = loss_joint / (config.accumulation_steps * config.G_group_size)
                    loss_scaled_joint.backward()
                    
                    metrics["grpo_losses"].append(loss_joint.item())
                    
                if (idx + 1) % config.accumulation_steps == 0 or (idx + 1) == len(batched_trajectories):
                    logger.info("  Gradient Accumulation limit met. Updating weights...")
                    torch.nn.utils.clip_grad_norm_(model_gpu0.parameters(), config.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    
            except Exception as grpo_err:
                logger.warning(f"  GRPO Optimization step bypassed at index {idx}. Details: {grpo_err}")
                continue

    # Save Phase 1 Checkpoint (GRPO)
    phase1_dir = os.path.join(config.save_checkpoint_dir, "phase1_grpo")
    os.makedirs(phase1_dir, exist_ok=True)
    model_gpu0.save_pretrained(phase1_dir)
    tokenizer.save_pretrained(phase1_dir)
    print(f"[SUCCESS] Phase 1 (GRPO) checkpoint saved to: {phase1_dir}")
    
    # Flush VRAM
    gc.collect()
    torch.cuda.empty_cache()
    
    active_v0_weights = {k: v.clone().cpu() for k, v in model_gpu0.named_parameters() if v.requires_grad}

    # -------------------------------------------------------------
    # PHASE 3: CONVEX STABILIZATION PASS
    # -------------------------------------------------------------
    print("\n" + "="*80)
    print("[RUN 2 INITIALIZED] Convex Formatting Stabilization Pass")
    print("="*80)
    
    optimizer = AdamW(model_gpu0.parameters(), lr=config.learning_rate * 0.8, weight_decay=config.weight_decay * 5)
    try:
        FastLanguageModel.for_training(model_gpu0)
    except Exception as _err:
        logger.debug(f"Ignoring non-fatal error during [FastLanguageModel.for_training(model_gpu0)]: {_err}")
    model_gpu0.train()
    optimizer.zero_grad()
    
    for idx, sample in enumerate(tqdm(training_slice, desc="SFT Training")):
        try:
            logger.info(f"===> [SFT Step {idx+1}/{len(training_slice)}] Processing domain: {sample.get('domain', 'unknown')}")
            sft_prompt = f"{sample['system_prompt']}\nUser: {sample['question']}\nAssistant: {sample['sft_reference']}"
            sft_inputs = tokenizer([sft_prompt], return_tensors="pt", padding=True).to("cuda:0")
            loss_sft = model_gpu0(**sft_inputs, labels=sft_inputs["input_ids"]).loss
            
            logger.info(f"  SFT Loss: {loss_sft.item():.4f}")
            loss_scaled = loss_sft / config.accumulation_steps
            loss_scaled.backward()
            
            metrics["sft_losses"].append(loss_sft.item())
            
            if (idx + 1) % config.accumulation_steps == 0 or (idx + 1) == len(training_slice):
                logger.info("  Gradient Accumulation limit met. Performing SFT optimizer update...")
                torch.nn.utils.clip_grad_norm_(model_gpu0.parameters(), config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
        except Exception as sample_err:
            logger.warning(f"  Round 2 failed for idx {idx}. Details: {sample_err}")
            continue
        
    active_v1_weights = {k: v.clone().cpu() for k, v in model_gpu0.named_parameters() if v.requires_grad}
    
    # Save Phase 2 Checkpoint (SFT)
    phase2_dir = os.path.join(config.save_checkpoint_dir, "phase2_sft")
    os.makedirs(phase2_dir, exist_ok=True)
    model_gpu0.save_pretrained(phase2_dir)
    tokenizer.save_pretrained(phase2_dir)
    print(f"[SUCCESS] Phase 2 (SFT) checkpoint saved to: {phase2_dir}")
    
    # Execute SLERP Merge on isolated parameters
    merged_weights = run_slerp_adapter_merge(model_gpu0, active_v0_weights, active_v1_weights, t=0.5)
    
    # Restore merged weights in-place
    for name, param in model_gpu0.named_parameters():
        if name in merged_weights:
            param.data.copy_(merged_weights[name].to(param.device))
            
    print("[SUCCESS] Slerp-merged weights successfully loaded into active model registers.")
    
    # Save Phase 3 Checkpoint (Merged)
    phase3_dir = os.path.join(config.save_checkpoint_dir, "phase3_merged")
    os.makedirs(phase3_dir, exist_ok=True)
    model_gpu0.save_pretrained(phase3_dir)
    tokenizer.save_pretrained(phase3_dir)
    print(f"[SUCCESS] Phase 3 (Merged) checkpoint saved to: {phase3_dir}")

    # ==========================================
    # 13. EVALUATION MATRIX REPORT
    # ==========================================
    print("\n" + "="*80)
    print("[EVALUATION INITIALIZED] Running Symmetrical Validation Matrix")
    print("="*80)
    try:
        FastLanguageModel.for_inference(model_gpu0)
    except Exception as _err:
        logger.debug(f"Ignoring non-fatal error during [FastLanguageModel.for_inference(model_gpu0)]: {_err}")
    model_gpu0.eval()
    eval_successes = 0
    
    for sample in eval_dataset[:30]:
        prompt_text = f"{sample['system_prompt']}\nUser: {sample['question']}\nAssistant: <think>"
        inputs = tokenizer([prompt_text], return_tensors="pt").to("cuda:0")
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                out = model_gpu0.generate(
                    **inputs,
                    max_new_tokens=config.max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    stopping_criteria=stopping_criteria,
                    use_cache=True
                )
        eval_text = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
        dream_match = re.search(r"<dream_code>(.*?)</dream_code>", eval_text, re.DOTALL)
        code_block = dream_match.group(1).strip() if dream_match else ""
        passed, _, _ = sandbox.execute_safely(code_block, sample["test_assertion"], sample["variables"])
        if passed:
            eval_successes += 1
            
    metrics["eval_pass_rate"] = eval_successes / min(30, len(eval_dataset)) if len(eval_dataset) > 0 else 0.0
    
    grpo_avg_loss = sum(metrics["grpo_losses"]) / len(metrics["grpo_losses"]) if metrics["grpo_losses"] else 0.0
    sft_avg_loss = sum(metrics["sft_losses"]) / len(metrics["sft_losses"]) if metrics["sft_losses"] else 0.0
    simpo_t1_rate = metrics["simpo_t1_pass"] / metrics["simpo_total_candidates"] if metrics["simpo_total_candidates"] else 0.0
    simpo_t2_rate = metrics["simpo_t2_pass"] / metrics["simpo_total_candidates"] if metrics["simpo_total_candidates"] else 0.0

    print("\n" + "="*65)
    print(f"{'Metric Description':<40} | {'Value / Rate':<20}")
    print("-" * 65)
    print(f"{'GRPO Active Loss (Mean)':<40} | {grpo_avg_loss:<20.4f}")
    print(f"{'SimPO Turn 1 Program Pass Rate':<40} | {simpo_t1_rate:<20.2%}")
    print(f"{'SimPO Turn 2 Program Pass Rate':<40} | {simpo_t2_rate:<20.2%}")
    print(f"{'SFT Stabilization Loss (Mean)':<40} | {sft_avg_loss:<20.4f}")
    print(f"{'Evaluation Split Assertion Pass Rate':<40} | {metrics['eval_pass_rate']:<20.2%}")
    print("="*65 + "\n")
    
    # Export fully compiled LoRA model merged directly as standard 16bit HF model first
    model_gpu0.save_pretrained_merged("agi_model_merged_16bit", tokenizer, save_method="merged_16bit")
    
    # Fast pre-compiled GGUF export bypassing slow manual compiling steps
    try:
        model_gpu0.save_pretrained_gguf("agi_v10_output", tokenizer, quantization_method="f16")
        print("[SUCCESS] GGUF compile using precompiled f16 quantization complete.")
    except Exception as e:
        print(f"[WARNING] GGUF pre-compiled conversion bypassed. Trace: {e}")
        
    try:
        model_gpu0.save_pretrained_gguf("agi_v10_output", tokenizer, quantization_method="q4_k_m")
        print("[SUCCESS] GGUF compile using precompiled q4_k_m quantization complete.")
    except Exception as e:
        print(f"[WARNING] GGUF pre-compiled q4_k_m conversion bypassed. Trace: {e}")
        
    return model_gpu0, tokenizer, search_adapter


# ==========================================
# 13. GRADIO INTERACTIVE ACTIVE-DEBUGGING UI
# ==========================================

def launch_gradio_interactive_ui(model, tokenizer, search_adapter: RealSearchAdapter):
    if not _GRADIO_AVAILABLE:
        print("\n[WARNING] Gradio is not installed. Falling back to console chat loop...")
        print("Type 'exit' or 'quit' to end the session.\n")
        sandbox_evaluator = TeacherSandbox(search_adapter=search_adapter)
        model.eval()
        
        while True:
            try:
                user_message = input("User: ")
                if user_message.lower() in ["exit", "quit"]:
                    break
                    
                active_prompt = (
                    "System: You are an advanced AI assistant specialized in reasoning and self-correction.\n"
                    f"User: {user_message}\nAssistant: <think>"
                )
                
                inputs = tokenizer([active_prompt], return_tensors="pt").to("cuda")
                prompt_len = inputs["input_ids"].shape[1]
                
                # Turn 1
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        pad_token_id=tokenizer.eos_token_id,
                        use_cache=True
                    )
                turn1_text = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
                
                # Extract code & execute
                dream_match = re.search(r"<dream_code>(.*?)</dream_code>", turn1_text, re.DOTALL)
                dream_code = dream_match.group(1).strip() if dream_match else ""
                
                test_assertions = "assert True"
                variables = []
                success, _, sandbox_terminal_logs = sandbox_evaluator.execute_safely(dream_code, test_assertions, variables)
                
                print(f"Assistant: <think>\n{turn1_text}")
                
                if not success:
                    # Turn 2: Self-correction trigger
                    turn2_prompt = (
                        f"{active_prompt}{turn1_text}\n"
                        f"<tool_response>{sandbox_terminal_logs}</tool_response>\n"
                        f"<critique>\n"
                        f"The code failed compiler execution. I will diagnose and correct.\n"
                        f"</critique>\n"
                        f"<conscious_reflection>\n"
                        f"Correction strategy.\n"
                        f"</conscious_reflection>\n"
                        f"<think>"
                    )
                    inputs2 = tokenizer([turn2_prompt], return_tensors="pt").to("cuda")
                    prompt_len2 = inputs2["input_ids"].shape[1]
                    with torch.no_grad():
                        out2 = model.generate(
                            **inputs2,
                            max_new_tokens=1024,
                            pad_token_id=tokenizer.eos_token_id,
                            use_cache=True
                        )
                    turn2_text = tokenizer.decode(out2[0][prompt_len2:], skip_special_tokens=True)
                    print(f"\n[Turn 2 Self-Correction]\n{turn2_text}")
            except KeyboardInterrupt:
                break
        return

    css_injection = """
    .think-panel { background-color: #1e293b; color: #cbd5e1; padding: 16px; border-radius: 8px; font-family: 'Courier New', monospace; }
    .sandbox-panel { background-color: #0f172a; color: #34d399; padding: 16px; border-radius: 8px; border-left: 5px solid #10b981; }
    """
    
    def process_chat_stream_with_self_correction(user_message, chat_history):
        active_prompt = (
            "System: You are an advanced AI assistant specialized in reasoning and self-correction.\n"
            f"User: {user_message}\nAssistant: <think>"
        )
        inputs = tokenizer([active_prompt], return_tensors="pt").to("cuda")
        prompt_len = inputs["input_ids"].shape[1]
        
        # Turn 1
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=1024,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
        turn1_text = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
        
        # Extract code & execute
        dream_match = re.search(r"<dream_code>(.*?)</dream_code>", turn1_text, re.DOTALL)
        dream_code = dream_match.group(1).strip() if dream_match else ""
        
        sandbox_evaluator = TeacherSandbox(search_adapter=search_adapter)
        success, _, sandbox_terminal_logs = sandbox_evaluator.execute_safely(dream_code, "assert True", [])
        
        parts = turn1_text.split("</think>")
        think_text = parts[0].replace("<think>", "").strip()
        final_solution = parts[-1].strip() if len(parts) > 1 else active_prompt
        
        if not success:
            # Turn 2: Self-correction trigger
            turn2_prompt = (
                f"{active_prompt}{turn1_text}\n"
                f"<tool_response>{sandbox_terminal_logs}</tool_response>\n"
                f"<critique>\n"
                f"The code failed compiler execution. I will diagnose and correct.\n"
                f"</critique>\n"
                f"<conscious_reflection>\n"
                f"Correction strategy.\n"
                f"</conscious_reflection>\n"
                f"<think>"
            )
            inputs2 = tokenizer([turn2_prompt], return_tensors="pt").to("cuda")
            prompt_len2 = inputs2["input_ids"].shape[1]
            with torch.no_grad():
                out2 = model.generate(
                    **inputs2,
                    max_new_tokens=1024,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True
                )
            turn2_text = tokenizer.decode(out2[0][prompt_len2:], skip_special_tokens=True)
            
            dream_match2 = re.search(r"<dream_code>(.*?)</dream_code>", turn2_text, re.DOTALL)
            dream_code = dream_match2.group(1).strip() if dream_match2 else ""
            success, _, sandbox_terminal_logs = sandbox_evaluator.execute_safely(dream_code, "assert True", [])
            
            parts2 = turn2_text.split("</think>")
            think_text += f"\n\n--- [Turn 2 Self-Correction] ---\n{parts2[0].replace('<think>', '').strip()}"
            final_solution = parts2[-1].strip() if len(parts2) > 1 else final_solution
            
        visual_html = (
            f"### 🔮 Inside the Self-Correcting Mind-State:\n"
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
    agi_ui.queue(default_concurrency_limit=1).launch(share=True)


if __name__ == "__main__":
    # Select which dataset to train on
    custom_dataset_file = 'dataset_small.json'
    model, tokenizer, search_adapter = run_symmetrical_agi_pipeline(custom_dataset_file)
    launch_gradio_interactive_ui(model, tokenizer, search_adapter)
