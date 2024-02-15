"""
This module is a basic implementation of langchain's features.
The purpose of this is twofold:
1) It turns out Celery and LangChain aren't working well together. API calls using Claude's model cause segfaults.
Attempting to debug this led to dead ends.
2) It turns out the functionality we're using in langchain can be easily rewritten in a few lines of code.
This allows us to be independent of their library. We kept the same API so that we can technically swap back to langchain
if we want.
"""