from cat.mad_hatter.decorators import plugin
from pydantic import BaseModel, Field, validator


class ContextGuardianEnricherSettings(BaseModel):
    double_pass: bool = Field(
        title="Double pass in memory",
        default=False,
        description="Whether to add all the retrieved context or doing a double pass in the memory to get \"post\" answer context"
    )
    utm_source: str = Field(
        title="UTM Source",
        default="",
        description="UTM source parameter to add to outgoing links for tracking. Leave empty to disable UTM tracking.",
    )
    min_query_length: int = Field(
        title="Minimum Query Length",
        default=10,
        description="Minimum number of characters a query must have to be accepted for processing",
    )
    default_message: str = Field(
        title="Default Message",
        default="Sorry, I can't help you. To answer adequately: • Write short, complete sentences • Express one request at a time",
        description="The default message to use when no declarative memory is found",
        extra={"type": "TextArea"}
    )
    panic_button_enabled: bool = Field(
        title="Panic Button Enabled",
        default=False,
        description="Enable panic button mode - always returns the panic button text regardless of context"
    )
    panic_button_text: str = Field(
        title="Panic Button Text",
        default="Sorry, I'm under maintenance right now. Please try again later.",
        description="The text to return when panic button mode is enabled",
        extra={"type": "TextArea"}
    )

    @validator('min_query_length')
    def validate_min_query_length(cls, v):
        """Validate that min_query_length is non-negative"""
        if not v >= 0:
            raise ValueError('Minimum query length must be non-negative')
        return v


@plugin
def settings_model():
    return ContextGuardianEnricherSettings