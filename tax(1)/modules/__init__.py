"""
税则查询系统模块包
包含PDF解析、向量数据库和查询API三个核心模块
"""

from .pdf_parser import PDFParser, TariffRecord
from .vector_database import VectorDatabase
from .query_api import TaxQueryAPI

__version__ = "1.0.0"
__author__ = "Tax System Team"

__all__ = [
    'PDFParser',
    'TariffRecord', 
    'VectorDatabase',
    'TaxQueryAPI'
]
