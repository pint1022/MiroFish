"""
LLM客户端封装
统一使用OpenAI格式调用
"""

import json
import re
from typing import Optional, Dict, Any, List
from openai import OpenAI, BadRequestError

from ..config import Config


class LLMClient:
    """LLM客户端"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model = model or Config.LLM_MODEL_NAME
        
        if not self.api_key:
            raise ValueError("LLM_API_KEY 未配置")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        # OpenAI推理模型（gpt-5/o系列）要求 max_completion_tokens 且不支持自定义 temperature；
        # 首次遇到对应400错误后自动切换并记住，兼容其他OpenAI格式服务商
        self._use_max_completion_tokens = False
        self._temperature_unsupported = False
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None
    ) -> str:
        """
        发送聊天请求
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            response_format: 响应格式（如JSON模式）
            
        Returns:
            模型响应文本
        """
        token_budget = max_tokens
        for _ in range(5):
            kwargs = {
                "model": self.model,
                "messages": messages,
            }
            if not self._temperature_unsupported:
                kwargs["temperature"] = temperature
            if self._use_max_completion_tokens:
                # 推理模型的思考token也计入预算，需要远大于期望输出长度的预算
                kwargs["max_completion_tokens"] = max(token_budget, 16384)
            else:
                kwargs["max_tokens"] = token_budget

            if response_format:
                kwargs["response_format"] = response_format

            try:
                response = self.client.chat.completions.create(**kwargs)
            except BadRequestError as e:
                error_text = str(e)
                if not self._use_max_completion_tokens and "max_completion_tokens" in error_text:
                    self._use_max_completion_tokens = True
                    continue
                if not self._temperature_unsupported and "temperature" in error_text and (
                    "unsupported" in error_text.lower() or "not support" in error_text.lower()
                ):
                    self._temperature_unsupported = True
                    continue
                raise

            content = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason

            # 推理模型可能把预算全部耗在思考上，返回空内容，加倍预算重试
            if not content.strip() and finish_reason == "length":
                token_budget = max(token_budget, 16384) * 2
                continue

            # 部分模型（如MiniMax M2.5）会在content中包含<think>思考内容，需要移除
            return re.sub(r'<think>[\s\S]*?</think>', '', content).strip()

        raise RuntimeError(
            f"LLM返回空内容（模型 {self.model} 的思考token耗尽了输出预算，已重试多次）"
        )
    
    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """
        发送聊天请求并返回JSON
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            
        Returns:
            解析后的JSON对象
        """
        response = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
        # 清理markdown代码块标记
        cleaned_response = response.strip()
        cleaned_response = re.sub(r'^```(?:json)?\s*\n?', '', cleaned_response, flags=re.IGNORECASE)
        cleaned_response = re.sub(r'\n?```\s*$', '', cleaned_response)
        cleaned_response = cleaned_response.strip()

        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            raise ValueError(f"LLM返回的JSON格式无效: {cleaned_response}")

