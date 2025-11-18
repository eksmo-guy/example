import asyncio

from typing import TypedDict
from typing import Unpack

from loguru import logger
from openai import AsyncOpenAI

from app.outsource.llm.llm_types import AiAnswer
from eksmo_src.eksmo_types import Usage


class AskParams(TypedDict):
    system_prompt: str
    prompt: str
    model: str
    max_tokens: int
    temperature: float


class LLMClient:
    """Пример LLM клиента."""

    def __init__(self, token: str) -> None:
        """Инициализация LLM клиента.

        Args:
            token: Токен OpenAPI
        """
        self._client = AsyncOpenAI(api_key=token)

    async def ask(
        self,
        *,
        system_prompt: str,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> AiAnswer:
        """Запрос к LLM.

        Args:
            system_prompt: Системный промпт.
            prompt: Основной текст запроса.
            model: Модель LLM.
            max_tokens: Максимальное количество токенов в ответе.
            temperature: Температура (0–1).

        Returns:
            AiAnswer: Объект с текстом ответа модели и статистикой использования токенов.
        """

        resp = await self._client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt},
            ],
        )

        usage = (
            v.model_dump()
            if (v := resp.usage)
            else Usage(
                prompt_tokens=0,
                completion_tokens=0,
            )
        )

        return AiAnswer(
            message=resp.choices[0].message.content,
            usage=usage,
        )

    async def ask_with_retries(
        self,
        retry: int = 3,
        delay_seconds: int = 5,
        **params: Unpack[AskParams],
    ) -> AiAnswer:
        """Отправляет запрос к AI API и возвращает ответ модели.

        Args:
            retry: Количество повторных попыток в случае ошибки.
            delay_seconds: Задержка между попытками в секундах.
            **params: Дополнительные аргументы, пробрасываемые в метод ask.

        Returns:
            AiAnswer: Объект с текстом ответа модели и статистикой использования токенов.

        Raises:
            RuntimeError: Если все попытки обращения к API неудачны.
        """
        retries = 0

        while True:
            try:
                return await self.ask(**params)
            except Exception as e:
                retries += 1

                logger.warning(
                    f"Ошибка при обращении к '{params.get('model')}', попытка "
                    f'{retries}/{retry}: {e}'
                )

                if retries >= retry:
                    logger.error(
                        f"Все попытки ({retry}) обращения к '{params.get('model')}' исчерпаны: {e}"
                    )
                    raise RuntimeError(f'Ошибка при вызове LLM после {retry} попыток: {e}') from e

                await asyncio.sleep(delay_seconds)
