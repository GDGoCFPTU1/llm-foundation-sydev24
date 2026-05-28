"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change function signatures.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v
"""
from google import genai
from google.genai import types
import anthropic
import openai
import os
import sys
import time
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Estimated costs per 1M INPUT & OUTPUT tokens (USD) as of March 2026
# Vietnamese text generally consumes ~1.5x - 2.0x more tokens than English due to Unicode/diacritics.
# ---------------------------------------------------------------------------
PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.300},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

# Standard Model Identifiers
OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-2.5-flash"
ANTHROPIC_MODEL = "claude-3-5-haiku"


# ---------------------------------------------------------------------------
# Task 1 — Call OpenAI (GPT-4o)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the OpenAI Chat Completions API and return the response text, latency,
    and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The OpenAI model to use (default: gpt-4o).
        temperature: Sampling temperature (0.0 – 2.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # response.usage contains input_tokens and output_tokens (prompt_tokens/completion_tokens)
    """
    # TODO: Import OpenAI, instantiate client, call chat.completions.create with parameters,
    #       measure execution start/end time, extract text and token usage, and return them.
    # Khởi tạo client, API key sẽ tự động lấy từ os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Định dạng tin nhắn gửi đi
    messages = [{"role": "user", "content": prompt}]
    
    # Bắt đầu đo thời gian
    start_time = time.time()
    
    # Gọi API tạo nội dung
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )
    
    # Kết thúc đo thời gian và tính toán độ trễ (latency)
    latency_seconds = time.time() - start_time
    
    # Trích xuất nội dung trả về
    response_text = response.choices[0].message.content
    
    # Trích xuất số lượng token usage vào một dictionary
    usage = {
        'input_tokens': response.usage.prompt_tokens,
        'output_tokens': response.usage.completion_tokens
    }
    
    # Trả về kết quả dưới dạng tuple
    return response_text, latency_seconds, usage
    raise NotImplementedError("Implement call_openai")


# ---------------------------------------------------------------------------
# Task 2 — Call Google Gemini 2.5 (Standard Practical Model)
# ---------------------------------------------------------------------------
def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Google Gemini API (using Gemini 2.5 Flash as standard) and return
    the response text, latency, and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The Gemini model to use (default: gemini-2.5-flash).
        temperature: Sampling temperature.
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        Option A (New Google GenAI SDK):
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            # Configure using types.GenerateContentConfig
            
        Option B (Legacy Google GenerativeAI SDK):
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            model_inst = genai.GenerativeModel(model)
            # Configure using genai.types.GenerationConfig
            
        Ensure your usage dictionary extracts 'input_tokens' and 'output_tokens' 
        from the response metadata (e.g. response.usage_metadata).
    """
    # TODO: Initialize Gemini client, set config parameters, call generate_content,
    #       measure latency, extract response text and usage metadata, and return the tuple.
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    # 2. Cấu hình các tham số thông qua GenerateContentConfig
    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens
    )
    
    try:
        # 3. Bắt đầu đo thời gian (Latency)
        start_time = time.time()
        
        # 4. Gọi API
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config
        )
        
        # 5. Kết thúc đo thời gian
        latency = time.time() - start_time
        
        # Lấy nội dung câu trả lời
        response_text = response.text
        
        # 6. Lấy Token Usage từ usage_metadata
        input_tokens = response.usage_metadata.prompt_token_count
        output_tokens = response.usage_metadata.candidates_token_count
        
        # Trả về kết quả dưới dạng tuple
        return response_text, round(latency, 4), {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
    except Exception:
        # Nếu có lỗi, trả lại ngoại lệ để caller xử lý
        raise


# ---------------------------------------------------------------------------
# Task 3 — Call Anthropic Claude (Exploratory track)
# ---------------------------------------------------------------------------
def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Anthropic Claude API (using Claude 3.5 Haiku as default) and return
    the response text, latency, and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The Claude model to use (default: claude-3-5-haiku).
        temperature: Sampling temperature (0.0 - 1.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum output tokens.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        # response.usage contains input_tokens and output_tokens
    """
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    start_time = time.time()
    response = client.messages.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens_to_sample=max_tokens,
    )
    latency_seconds = time.time() - start_time
    response_text = ""
    if response.content:
        response_text = response.content[-1].text

    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    return response_text, latency_seconds, usage


# ---------------------------------------------------------------------------
# Task 4 — Compare Models (OpenAI GPT-4o vs OpenAI Mini vs Gemini 2.5 Flash)
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    """
    Call OpenAI (gpt-4o), OpenAI Mini (gpt-4o-mini), and Gemini 2.5 Flash (gemini-2.5-flash)
    with the same prompt and return a structured comparison dictionary.

    Calculate the exact USD token cost for input + output using the prices in PRICING_1M_TOKENS.

    Args:
        prompt: The user message to send to all models.

    Returns:
        A dictionary containing:
            - "gpt4o": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
            - "gpt4o_mini": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
            - "gemini_flash": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
    """
    # TODO: Call call_openai with default gpt-4o model
    # TODO: Call call_openai with gpt-4o-mini model
    # TODO: Call call_gemini with default gemini-2.5-flash model
    # TODO: Calculate costs exactly based on input and output token counts using PRICING_1M_TOKENS
    #       Formula: Cost = (input_tokens * input_rate_per_1M + output_tokens * output_rate_per_1M) / 1,000,000
    # TODO: Assemble and return the comparison dictionary.
    gpt4o_text, gpt4o_latency, gpt4o_usage = call_openai(prompt=prompt, model="gpt-4o")
    
    gpt4o_in_tok = gpt4o_usage.get("input_tokens", 0)
    gpt4o_out_tok = gpt4o_usage.get("output_tokens", 0)
    
    # Tính cost: Cost = (input * rate_in + output * rate_out) / 1,000,000
    gpt4o_cost = (
        gpt4o_in_tok * PRICING_1M_TOKENS["gpt-4o"]["input"] + 
        gpt4o_out_tok * PRICING_1M_TOKENS["gpt-4o"]["output"]
    ) / 1_000_000


    # --- 2. Gọi GPT-4o-mini ---
    mini_text, mini_latency, mini_usage = call_openai(prompt=prompt, model="gpt-4o-mini")
    
    mini_in_tok = mini_usage.get("input_tokens", 0)
    mini_out_tok = mini_usage.get("output_tokens", 0)
    
    mini_cost = (
        mini_in_tok * PRICING_1M_TOKENS["gpt-4o-mini"]["input"] + 
        mini_out_tok * PRICING_1M_TOKENS["gpt-4o-mini"]["output"]
    ) / 1_000_000


    # --- 3. Gọi Gemini 2.5 Flash ---
    gemini_text, gemini_latency, gemini_usage = call_gemini(
        prompt=prompt,
        model="gemini-2.5-flash"
    )
    gemini_in_tok = gemini_usage.get("input_tokens", 0)
    gemini_out_tok = gemini_usage.get("output_tokens", 0)

    gemini_cost = (
        gemini_in_tok * PRICING_1M_TOKENS["gemini-2.5-flash"]["input"] + 
        gemini_out_tok * PRICING_1M_TOKENS["gemini-2.5-flash"]["output"]
    ) / 1_000_000


    # --- 4. Tổng hợp và trả về kết quả ---
    comparison_dict = {
        "gpt4o": {
            "response": gpt4o_text,
            "latency": round(gpt4o_latency, 4),
            "cost": gpt4o_cost,
            "input_tokens": gpt4o_in_tok,
            "output_tokens": gpt4o_out_tok
        },
        "gpt4o_mini": {
            "response": mini_text,
            "latency": round(mini_latency, 4),
            "cost": mini_cost,
            "input_tokens": mini_in_tok,
            "output_tokens": mini_out_tok
        },
        "gemini_flash": {
            "response": gemini_text,
            "latency": round(gemini_latency, 4),
            "cost": gemini_cost,
            "input_tokens": gemini_in_tok,
            "output_tokens": gemini_out_tok
        }
    }
    
    return comparison_dict
    raise NotImplementedError("Implement compare_models")


# ---------------------------------------------------------------------------
# Task 5 — Streaming chatbot with Gemini 2.5 (Focus Model)
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    print("=" * 60)
    print("🤖 CHATBOT GEMINI 2.5 FLASH (Chế độ Streaming)")
    print("💡 Mẹo: Gõ 'quit' hoặc 'exit' để kết thúc.")
    print("=" * 60)

    # Khởi tạo mảng lưu trữ lịch sử
    # Mỗi tin nhắn sẽ được lưu dưới dạng Dictionary tương thích với SDK
    history = []
    
    while True:
        try:
            # 2. Nhận input từ người dùng
            user_input = input("\nBạn: ")
            
            # Kiểm tra điều kiện thoát
            if user_input.strip().lower() in ['quit', 'exit']:
                print("\n👋 Đã thoát chatbot. Hẹn gặp lại!")
                break
            
            # Bỏ qua nếu người dùng ấn Enter mà không nhập gì
            if not user_input.strip():
                continue
                
            # 3. Thêm tin nhắn của User vào lịch sử
            history.append({
                "role": "user", 
                "parts": [{"text": user_input}]
            })
            
            # 4. Giới hạn lịch sử (3 turns)
            # 1 turn = 1 User + 1 Model (2 tin nhắn) -> 3 turns = 6 tin nhắn.
            # Vì ta vừa thêm tin nhắn User mới, số lượng tối đa cần giữ là 7 tin nhắn.
            if len(history) > 7:
                history = history[-7:]
                
            print("Gemini: ", end="", flush=True)
            
            # 5. Gọi API ở chế độ Streaming
            response_stream = client.models.generate_content_stream(
                model=GEMINI_MODEL,
                contents=history
            )
            
            full_response = ""
            
            # 6. In từng chunk dữ liệu ngay khi nhận được
            for chunk in response_stream:
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    full_response += chunk.text
                    
            print() # Xuống dòng khi model hoàn tất câu trả lời
            
            # 7. Lưu câu trả lời của Model vào lịch sử để làm ngữ cảnh cho câu tiếp theo
            history.append({
                "role": "model", 
                "parts": [{"text": full_response}]
            })
            
        except KeyboardInterrupt:
            # Xử lý khi người dùng nhấn Ctrl+C
            print("\n\n⚠️ Đã ngắt kết nối. Thoát chatbot.")
            sys.exit(0)
            
        except Exception as e:
            print(f"\n\n❌ [Lỗi]: {e}")
            # Nếu gọi API lỗi, hãy xóa tin nhắn user vừa thêm vào để tránh làm hỏng cấu trúc so le (user-model)
            if history and history[-1]["role"] == "user":
                history.pop()
    return None
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Call fn(). If it raises an exception, retry up to max_retries times
    with exponential backoff (delay = base_delay * 2^attempt).

    Args:
        fn:          Zero-argument callable to execute.
        max_retries: Maximum number of retry attempts.
        base_delay:  Initial delay in seconds before the first retry.

    Returns:
        The return value of fn() on success.

    Raises:
        The last exception raised by fn() after all retries are exhausted.
    """
    # TODO: implement retry loop with exponential backoff
    for attempt in range(max_retries + 1):
        try:
            # Thử gọi hàm
            return fn()
        except Exception as e:
            # Nếu đã hết số lần thử lại (attempt == max_retries), ném ra lỗi cuối cùng
            if attempt == max_retries:
                raise e
            
            # Tính toán thời gian chờ: delay = base_delay * (2 ^ attempt)
            delay = base_delay * (2 ** attempt)
            
            # Có thể in ra log để theo dõi quá trình retry (tuỳ chọn)
            print(f"[Cảnh báo] Lỗi xảy ra: {e}. Thử lại lần {attempt + 1}/{max_retries} sau {delay:.2f} giây...")
            
            # Tạm dừng thực thi
            time.sleep(delay)
    raise NotImplementedError("Implement retry_with_backoff")


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Run compare_models on each prompt in the list.

    Args:
        prompts: List of prompt strings.

    Returns:
        List of dicts, each being the compare_models result with an extra
        key "prompt" containing the original prompt string.
    """
    # TODO: iterate over prompts, call compare_models, and inject the original "prompt".
    results = []
    
    # Duyệt qua từng prompt trong danh sách
    for prompt in prompts:
        try:
            # Gọi hàm compare_models (Task 4)
            comparison_result = compare_models(prompt)
        except Exception as e:
            # Nếu có lỗi (VD: mất mạng khi đang chạy giữa chừng), tạo dict ghi nhận lỗi
            comparison_result = {"error": str(e)}
        
        # Thêm key "prompt" chứa nội dung gốc vào dictionary kết quả
        comparison_result["prompt"] = prompt
        
        # Đưa vào danh sách kết quả tổng
        results.append(comparison_result)
        
    return results
    raise NotImplementedError("Implement batch_compare")


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    """
    Format a list of batch compare results as a readable Markdown table string.

    Args:
        results: List of dicts as returned by batch_compare.

    Returns:
        A beautiful Markdown table string with columns:
        | Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |
    """
    def _truncate(text: str, length: int = 50) -> str:
        if text is None:
            return ""
        return text if len(text) <= length else text[:length - 3] + "..."

    header = [
        "Prompt",
        "Model",
        "Response (truncated)",
        "Latency",
        "Tokens (In/Out)",
        "Cost (USD)",
    ]
    lines = ["| " + " | ".join(header) + " |",
             "|" + " --- |" * len(header)]

    for result in results:
        prompt = result.get("prompt", "")
        if "error" in result:
            error_text = _truncate(result["error"], 50)
            for model_name in ["GPT-4o", "GPT-4o-Mini", "Gemini-Flash"]:
                lines.append(
                    f"| {prompt} | {model_name} | ERROR: {error_text} | - | - | - |"
                )
        else:
            rows = [
                ("GPT-4o", result.get("gpt4o", {})),
                ("GPT-4o-Mini", result.get("gpt4o_mini", {})),
                ("Gemini-Flash", result.get("gemini_flash", {})),
            ]
            for model_name, stats in rows:
                response = _truncate(stats.get("response", ""))
                latency = stats.get("latency", "-")
                in_tok = stats.get("input_tokens", "-")
                out_tok = stats.get("output_tokens", "-")
                cost = stats.get("cost", "-")
                lines.append(
                    f"| {prompt} | {model_name} | {response} | {latency} | {in_tok}/{out_tok} | {cost} |"
                )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Model Comparison Test ===")
    test_prompt = "Hãy giải thích sự khác biệt giữa temperature và top_p bằng tiếng Việt ngắn gọn trong 2 câu."
    try:
        # Note: Requires valid API keys set in environment variables
        result = compare_models(test_prompt)
        for model_name, stats in result.items():
            print(f"\n[{model_name.upper()}]")
            print(f"Latency: {stats['latency']:.2f}s | Cost: ${stats['cost']:.6f}")
            print(f"Tokens: {stats['input_tokens']} in / {stats['output_tokens']} out")
            print(f"Response: {stats['response']}")
    except Exception as e:
        print(f"Skipping live API comparison test: {e}")
        print("Set your API keys to run manual tests.")

    print("\n=== Starting Gemini 2.5 Chatbot (type 'quit' to exit) ===")
    try:
        streaming_chatbot()
    except Exception as e:
        print(f"Chatbot failed to start: {e}")
