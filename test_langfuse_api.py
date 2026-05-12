from langfuse import observe
from app.utils.langfuse_client import langfuse_client

@observe()
def my_pipeline():
    my_span()

@observe(as_type="span")
def my_span():
    langfuse_client.update_current_span(metadata={"key": "value"})

my_pipeline()
