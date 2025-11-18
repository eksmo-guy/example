import asyncio

from pathlib import Path

from loguru import logger

from app.outsource.flows.translate_flow import TranslateFlow
from eksmo_src.eksmo_types import Usage

DATA_DIR_PATH = Path(__file__).resolve().parent.parent / 'demonstration_data'

INPUT_FILE_PATH = DATA_DIR_PATH / 'text_for_translate.md'
OUTPUT_FILE_PATH = DATA_DIR_PATH / 'translated.md'


async def main() -> None:
    """Точка входа для всего проекта."""
    logger.info('Запуск демонстрации работы модуля')
    logger.info(f'Файл с текстом для перевода: {INPUT_FILE_PATH}')

    usage = Usage()

    await TranslateFlow.run(
        total_usage=usage, input_file_path=INPUT_FILE_PATH, output_file_path=OUTPUT_FILE_PATH
    )

    logger.info(f'Использовано токенов: {usage.model_dump()}')
    logger.info(f'Файл с результатом:{OUTPUT_FILE_PATH}')


asyncio.run(main())
