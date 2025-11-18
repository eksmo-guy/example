from typing import Self

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import computed_field


class Usage(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        validate_assignment=True,
        from_attributes=True,
    )

    input_tokens: int = Field(0, alias='prompt_tokens')
    output_tokens: int = Field(0, alias='completion_tokens')

    @computed_field
    def total_tokens(self) -> int:
        """Сумма токенов."""
        return self.input_tokens + self.output_tokens

    def __iadd__(self, other: Self) -> Self:
        """Складывает Usage и мутирует текущий экземпляр."""
        if not isinstance(other, Usage):
            return NotImplemented

        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        return self
