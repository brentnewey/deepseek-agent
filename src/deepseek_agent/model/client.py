"""DeepSeek model client for Ollama integration"""

import json
from typing import AsyncGenerator, Dict, List, Optional, Any
import httpx
from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str


class ChatResponse(BaseModel):
    message: Optional[Message] = None
    done: bool
    created_at: Optional[str] = None
    model: Optional[str] = None
    # Support for tool calls
    tool_calls: Optional[List[Dict[str, Any]]] = None

    def __init__(self, **data):
        # Handle tool calls in message
        if 'message' in data and isinstance(data['message'], dict):
            if 'tool_calls' in data['message']:
                data['tool_calls'] = data['message'].get('tool_calls')
            # Convert message dict to Message object if needed
            if 'role' in data['message'] and 'content' in data['message']:
                data['message'] = Message(**data['message'])
        super().__init__(**data)


class DeepSeekClient:
    """Client for interacting with DeepSeek V2.5 via Ollama"""
    
    def __init__(self, host: str = "http://localhost:11434", model: str = "deepseek-v2.5"):
        self.host = host.rstrip("/")
        self.model = model
        self.client = httpx.AsyncClient(timeout=300.0)
    
    @staticmethod
    def _normalize_model_name(name: str) -> str:
        """Normalize model name for comparison"""
        return name.lower().split(":", 1)[0].strip()
    
    def _matches_target_model(self, entry: Any) -> bool:
        """Determine whether an entry from /api/tags matches the target model"""
        target = self._normalize_model_name(self.model)
        
        if isinstance(entry, str):
            return self._normalize_model_name(entry) == target
        
        if isinstance(entry, dict):
            for key in ("name", "model"):
                candidate = entry.get(key)
                if isinstance(candidate, str) and self._normalize_model_name(candidate) == target:
                    return True
        
        return False
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    async def check_model_availability(self) -> bool:
        """Check if the DeepSeek model is available"""
        try:
            response = await self.client.get(f"{self.host}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                for model_entry in models:
                    if self._matches_target_model(model_entry):
                        return True
            return False
        except (httpx.ConnectError, httpx.TimeoutException):
            # Ollama service not running
            return False
        except Exception:
            return False
    
    async def pull_model(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Pull the DeepSeek model if not available"""
        last_payload: Optional[Dict[str, Any]] = None
        async with self.client.stream(
            "POST",
            f"{self.host}/api/pull",
            json={"name": self.model}
        ) as response:
            if response.status_code != 200:
                error_body = await response.aread()
                decoded_error = error_body.decode("utf-8", errors="ignore")
                raise RuntimeError(
                    f"Failed to pull model {self.model}: {response.status_code} {decoded_error.strip()}"
                )
            
            async for line in response.aiter_lines():
                if not line or not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                last_payload = payload
                if "error" in payload:
                    raise RuntimeError(payload["error"])
                yield payload
        
        if not last_payload:
            raise RuntimeError(f"No data received while pulling model {self.model}")
        
        status = str(last_payload.get("status", "")).lower()
        if status not in {"success", "exists"}:
            raise RuntimeError(
                f"Model pull ended with unexpected status '{status or 'unknown'}'"
            )
    
    async def chat(
        self,
        messages: List[Message],
        stream: bool = True,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncGenerator[ChatResponse, None]:
        """Send chat messages to DeepSeek model"""
        
        # Prepare messages
        formatted_messages = []
        if system_prompt:
            formatted_messages.append({"role": "system", "content": system_prompt})

        for msg in messages:
            formatted_messages.append({"role": msg.role, "content": msg.content})

        # Prepare request
        request_data = {
            "model": self.model,
            "messages": formatted_messages,
            "stream": stream,
            "options": {
                "temperature": temperature,
            }
        }

        if max_tokens:
            request_data["options"]["num_predict"] = max_tokens

        # Add tools if provided
        if tools:
            request_data["tools"] = tools

        # Send request
        if stream:
            # Streaming mode
            async with self.client.stream(
                "POST",
                f"{self.host}/api/chat",
                json=request_data
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    raise Exception(f"API error: {response.status_code} - {error_text}")

                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            data = json.loads(line)
                            yield ChatResponse(**data)
                        except Exception as e:
                            print(f"Error parsing response: {e}")
                            continue
        else:
            # Non-streaming mode - single response
            response = await self.client.post(
                f"{self.host}/api/chat",
                json=request_data
            )
            if response.status_code != 200:
                raise Exception(f"API error: {response.status_code} - {response.text}")

            data = response.json()
            yield ChatResponse(**data)
    
    async def generate_code(
        self,
        prompt: str,
        language: Optional[str] = None,
        context: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Generate code using DeepSeek"""
        
        system_prompt = """You are an expert software engineer who creates working code files.

When asked to generate code:
1. Create complete, runnable code (not just snippets)
2. Include proper error handling and edge cases
3. Add helpful comments explaining key parts
4. Make the code self-contained and ready to execute

Always provide the COMPLETE file content that can be saved and run immediately."""
        
        if language:
            system_prompt += f" The code should be in {language}."
        
        user_prompt = prompt
        if context:
            user_prompt = f"Context:\n{context}\n\nTask:\n{prompt}"
        
        messages = [Message(role="user", content=user_prompt)]
        
        full_response = ""
        async for response in self.chat(messages, system_prompt=system_prompt):
            if response.message and response.message.content:
                chunk = response.message.content
                full_response += chunk
                yield chunk
    
    async def explain_code(self, code: str, language: Optional[str] = None) -> str:
        """Explain code functionality"""
        
        system_prompt = """You are an expert code reviewer. Explain code clearly and concisely.
Focus on what the code does, how it works, and any important patterns or considerations."""
        
        prompt = f"Explain this code:\n\n`{language or ''}\n{code}\n`"
        messages = [Message(role="user", content=prompt)]
        
        full_response = ""
        async for response in self.chat(messages, system_prompt=system_prompt, stream=False):
            if response.message and response.message.content:
                full_response += response.message.content
        
        return full_response
    
    async def review_code(self, code: str, language: Optional[str] = None) -> str:
        """Review code for issues and improvements"""
        
        system_prompt = """You are an expert code reviewer. Analyze code for:
- Bugs and potential issues
- Performance problems  
- Security vulnerabilities
- Code style and best practices
- Maintainability concerns
Provide specific, actionable feedback."""
        
        prompt = f"Review this code:\n\n`{language or ''}\n{code}\n`"
        messages = [Message(role="user", content=prompt)]
        
        full_response = ""
        async for response in self.chat(messages, system_prompt=system_prompt, stream=False):
            if response.message and response.message.content:
                full_response += response.message.content
        
        return full_response
