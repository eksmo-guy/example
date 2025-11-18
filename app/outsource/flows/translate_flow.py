import re

from pathlib import Path

from loguru import logger

from app.consts import AI__API_KEY
from app.consts import AI__MAX_TOKENS
from app.consts import AI__MODEL
from app.consts import AI__TEMPERATURE
from app.consts import SYSTEM_PROMPT
from app.outsource.llm.llm_client import LLMClient
from eksmo_src.eksmo_types import Usage


class TranslateFlow:
    @classmethod
    async def run(
        cls,
        *,
        total_usage: Usage,
        input_file_path: Path,
        output_file_path: Path,
    ) -> None:
        """Запуск флоу LLM перевода.

        Args:
            total_usage: Аккумулятор токенов, который будет обновлён после выполнения запроса.
            input_file_path: Путь к файлу с исходным текстом для перевода.
            output_file_path: Путь к файлу, куда будет записан переведённый текст.
        """
        logger.debug('Запуск флоу LLM перевода')

        llm = LLMClient(token=AI__API_KEY)

        original_text = input_file_path.read_text()
        # Разбиваем по знакам конца предложения, сохраняя их
        chunks = re.findall(r'.+?[.!?](?:\s+|$)', original_text)

        for chunk in chunks:
            answer = await llm.ask_with_retries(
                system_prompt=SYSTEM_PROMPT,
                prompt=chunk,
                model=AI__MODEL,
                max_tokens=AI__MAX_TOKENS,
                temperature=AI__TEMPERATURE,
            )

            logger.info(f'Получен ответ от {AI__MODEL}: {answer.model_dump()}')

            total_usage += answer.usage
            output_file_path.write_text(answer.message)

            logger.debug('Флоу LLM перевода завершен успешно')
