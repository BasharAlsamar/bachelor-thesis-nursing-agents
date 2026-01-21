"""
Workaround für PaddleOCR Kompatibilitätsprobleme mit LangChain 0.3+
"""
import sys
from unittest.mock import MagicMock

# Mock alle fehlenden LangChain-Module
mock_module = MagicMock()
sys.modules['langchain.docstore'] = mock_module
sys.modules['langchain.docstore.document'] = mock_module
sys.modules['langchain.text_splitter'] = mock_module
sys.modules['langchain.embeddings'] = mock_module
sys.modules['langchain.vectorstores'] = mock_module

# Jetzt kann PaddleOCR importiert werden
from paddleocr import PaddleOCR

__all__ = ['PaddleOCR']
