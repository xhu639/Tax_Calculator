"""
PDF解析模块
将PDF文件解析为结构化的CSV数据
支持多种PDF解析引擎和层级结构识别
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

import pdfplumber
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

@dataclass
class TariffRecord:
    """税则记录数据结构"""
    tariff_code: str
    description: str
    import_duty: str
    import_excise: str
    import_vagst: str
    export_duty: str
    unit: str
    sitc_code: str
    hierarchy_level: int
    level_name: str
    has_tax_info: bool
    metadata: Dict[str, Any]

class PDFParser:
    """PDF解析器，专门用于解析关税税则表PDF"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化解析器"""
        self.config = config or {}
        self.debug = self.config.get('debug', False)
        
        # 层级上下文存储
        self.hierarchy_context = {
            'main_category': '',      # 大类描述
            'sub_category': '',       # 子类描述
            'sub_sub_category': ''    # 二级子类描述
        }
        
        logger.info("PDF解析器初始化完成")
    
    def parse_pdf(self, pdf_path: str, output_csv: str = None) -> List[Dict[str, Any]]:
        """
        解析PDF文件并返回结构化数据
        
        Args:
            pdf_path: PDF文件路径
            output_csv: 输出CSV文件路径（可选）
            
        Returns:
            解析后的税则记录列表
        """
        logger.info(f"开始解析PDF文件: {pdf_path}")
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")
        
        all_records = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                logger.info(f"PDF总页数: {total_pages}")
                
                for page_num, page in enumerate(pdf.pages, 1):
                    if self.debug and page_num % 50 == 0:
                        logger.info(f"处理页面: {page_num}/{total_pages}")
                    
                    # 提取页面数据
                    page_records = self._extract_page_data(page, page_num)
                    if page_records:
                        all_records.extend(page_records)
                        logger.debug(f"页面 {page_num} 提取到 {len(page_records)} 条记录")
        
        except Exception as e:
            logger.error(f"PDF解析失败: {e}")
            raise
        
        logger.info(f"成功提取 {len(all_records)} 条记录")
        
        # 保存为CSV文件
        if output_csv:
            self._save_to_csv(all_records, output_csv)
        
        return all_records
    
    def _extract_page_data(self, page, page_num: int) -> List[Dict[str, Any]]:
        """提取页面数据"""
        records = []
        
        # 提取表格数据
        tables = page.extract_tables()
        if tables:
            for table in tables:
                table_records = self._process_table(table, page_num)
                records.extend(table_records)
        
        # 提取文本数据（作为补充）
        text = page.extract_text()
        if text:
            text_records = self._process_text(text, page_num)
            records.extend(text_records)
        
        return records
    
    def _process_table(self, table: List[List[str]], page_num: int) -> List[Dict[str, Any]]:
        """处理表格数据"""
        records = []
        
        if not table or len(table) < 2:
            return records
        
        # 跳过表头
        for row_idx, row in enumerate(table[1:], 1):
            if not row or len(row) < 8:
                continue
            
            # 提取税则号
            tariff_code = self._extract_tariff_code(row[0])
            if not tariff_code:
                continue
            
            # 提取描述
            description = self._clean_description(row[1])
            if not description:
                continue
            
            # 更新层级上下文
            self._update_hierarchy_context(tariff_code, description)
            
            # 提取税率信息
            tax_info = self._extract_tax_info(row[2:6])
            
            # 创建记录
            record = TariffRecord(
                tariff_code=tariff_code,
                description=description,
                import_duty=tax_info['import_duty'],
                import_excise=tax_info['import_excise'],
                import_vagst=tax_info['import_vagst'],
                export_duty=tax_info['export_duty'],
                unit=row[6] if len(row) > 6 else '',
                sitc_code=row[7] if len(row) > 7 else '',
                hierarchy_level=self._get_hierarchy_level(tariff_code),
                level_name=self._get_level_name(tariff_code),
                has_tax_info=tax_info['has_tax_info'],
                metadata={
                    'page_number': page_num,
                    'row_index': row_idx,
                    'extraction_method': 'table'
                }
            )
            
            records.append(record)
        
        return records
    
    def _process_text(self, text: str, page_num: int) -> List[Dict[str, Any]]:
        """处理文本数据（补充表格数据）"""
        records = []
        
        # 按行分割文本
        lines = text.split('\n')
        
        for line_idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # 查找税则号
            tariff_code = self._extract_tariff_code_from_text(line)
            if not tariff_code:
                continue
            
            # 提取描述和税率信息
            description = self._extract_description_from_text(line)
            tax_info = self._extract_tax_info_from_text(line)
            
            if description:
                # 更新层级上下文
                self._update_hierarchy_context(tariff_code, description)
                
                # 创建记录
                record = TariffRecord(
                    tariff_code=tariff_code,
                    description=description,
                    import_duty=tax_info['import_duty'],
                    import_excise=tax_info['import_excise'],
                    import_vagst=tax_info['import_vagst'],
                    export_duty=tax_info['export_duty'],
                    unit='',
                    sitc_code='',
                    hierarchy_level=self._get_hierarchy_level(tariff_code),
                    level_name=self._get_level_name(tariff_code),
                    has_tax_info=tax_info['has_tax_info'],
                    metadata={
                        'page_number': page_num,
                        'line_index': line_idx,
                        'extraction_method': 'text'
                    }
                )
                
                records.append(record)
        
        return records
    
    def _extract_tariff_code(self, text: str) -> str:
        """从文本中提取税则号"""
        if not text:
            return ""
        
        # 清理文本
        text = str(text).strip()
        
        # 匹配税则号模式
        patterns = [
            r'(\d{4}\.\d{2}\.\d{2})',  # 标准格式：0101.2100
            r'(\d{2}\.\d{2})',         # 简化格式：01.01
            r'(\d{4}\.\d{2})',         # 中间格式：0101.21
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        return ""
    
    def _extract_tariff_code_from_text(self, text: str) -> str:
        """从文本行中提取税则号"""
        # 查找行首的税则号
        patterns = [
            r'^(\d{4}\.\d{2}\.\d{2})',  # 标准格式
            r'^(\d{2}\.\d{2})',         # 简化格式
            r'^(\d{4}\.\d{2})',         # 中间格式
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        return ""
    
    def _clean_description(self, text: str) -> str:
        """清理描述文本"""
        if not text:
            return ""
        
        # 转换为字符串并清理
        text = str(text).strip()
        
        # 移除多余的空格和换行
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def _extract_description_from_text(self, text: str) -> str:
        """从文本行中提取描述"""
        # 移除税则号部分
        text = re.sub(r'^\d{4}\.\d{2}(?:\.\d{2})?', '', text).strip()
        
        # 移除税率信息
        text = re.sub(r'\d+(?:\.\d+)?%', '', text).strip()
        text = re.sub(r'Free', '', text).strip()
        
        return text
    
    def _extract_tax_info(self, tax_columns: List[str]) -> Dict[str, Any]:
        """提取税率信息"""
        tax_info = {
            'import_duty': '',
            'import_excise': '',
            'import_vagst': '',
            'export_duty': '',
            'has_tax_info': False
        }
        
        if len(tax_columns) >= 4:
            tax_info['import_duty'] = self._clean_tax_value(tax_columns[0])
            tax_info['import_excise'] = self._clean_tax_value(tax_columns[1])
            tax_info['import_vagst'] = self._clean_tax_value(tax_columns[2])
            tax_info['export_duty'] = self._clean_tax_value(tax_columns[3])
        
        # 检查是否有税率信息
        tax_info['has_tax_info'] = any(
            value and value != '' and value != 'Free' 
            for value in tax_info.values() 
            if isinstance(value, str)
        )
        
        return tax_info
    
    def _extract_tax_info_from_text(self, text: str) -> Dict[str, Any]:
        """从文本中提取税率信息"""
        tax_info = {
            'import_duty': '',
            'import_excise': '',
            'import_vagst': '',
            'export_duty': '',
            'has_tax_info': False
        }
        
        # 查找税率模式
        rate_pattern = r'(\d+(?:\.\d+)?%|Free)'
        rates = re.findall(rate_pattern, text)
        
        if rates:
            # 简单分配税率（实际应用中需要更复杂的逻辑）
            for i, rate in enumerate(rates[:4]):
                if i == 0:
                    tax_info['import_duty'] = rate
                elif i == 1:
                    tax_info['import_excise'] = rate
                elif i == 2:
                    tax_info['import_vagst'] = rate
                elif i == 3:
                    tax_info['export_duty'] = rate
            
            tax_info['has_tax_info'] = True
        
        return tax_info
    
    def _clean_tax_value(self, value: str) -> str:
        """清理税率值"""
        if not value:
            return ""
        
        value = str(value).strip()
        
        # 处理特殊值
        if value.lower() in ['free', '免税', '0']:
            return "Free"
        
        # 确保百分比格式
        if value and not value.endswith('%') and value != 'Free':
            try:
                float(value)
                return f"{value}%"
            except ValueError:
                pass
        
        return value
    
    def _update_hierarchy_context(self, tariff_code: str, description: str):
        """更新层级上下文"""
        if not tariff_code:
            return
        
        # 根据税则号确定层级
        if '.' not in tariff_code:
            # 一级分类
            self.hierarchy_context['main_category'] = description
            self.hierarchy_context['sub_category'] = ''
            self.hierarchy_context['sub_sub_category'] = ''
        elif tariff_code.count('.') == 1:
            # 二级分类
            self.hierarchy_context['sub_category'] = description
            self.hierarchy_context['sub_sub_category'] = ''
        elif tariff_code.count('.') == 2:
            # 三级分类
            self.hierarchy_context['sub_sub_category'] = description
    
    def _get_hierarchy_level(self, tariff_code: str) -> int:
        """获取层级级别"""
        if not tariff_code:
            return 0
        
        if '.' not in tariff_code:
            return 1
        elif tariff_code.count('.') == 1:
            return 2
        elif tariff_code.count('.') == 2:
            return 3
        else:
            return 4
    
    def _get_level_name(self, tariff_code: str) -> str:
        """获取层级名称"""
        if not tariff_code:
            return ""
        
        level = self._get_hierarchy_level(tariff_code)
        
        if level == 1:
            return self.hierarchy_context['main_category']
        elif level == 2:
            return self.hierarchy_context['sub_category']
        elif level == 3:
            return self.hierarchy_context['sub_sub_category']
        else:
            return ""
    
    def _save_to_csv(self, records: List[TariffRecord], output_path: str):
        """保存记录到CSV文件"""
        logger.info(f"保存 {len(records)} 条记录到 {output_path}")
        
        # 创建输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 转换为DataFrame
        data = []
        for record in records:
            data.append({
                'tariff_code': record.tariff_code,
                'description': record.description,
                'import_duty': record.import_duty,
                'import_excise': record.import_excise,
                'import_vagst': record.import_vagst,
                'export_duty': record.export_duty,
                'unit': record.unit,
                'sitc_code': record.sitc_code,
                'hierarchy_level': record.hierarchy_level,
                'level_name': record.level_name,
                'has_tax_info': record.has_tax_info,
                'page_number': record.metadata.get('page_number', ''),
                'extraction_method': record.metadata.get('extraction_method', '')
            })
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        logger.info(f"CSV文件保存成功: {output_path}")

def main():
    """主函数，用于测试PDF解析功能"""
    import argparse
    
    parser = argparse.ArgumentParser(description='PDF解析器')
    parser.add_argument('pdf_path', help='PDF文件路径')
    parser.add_argument('-o', '--output', help='输出CSV文件路径')
    parser.add_argument('-d', '--debug', action='store_true', help='启用调试模式')
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建解析器
    config = {'debug': args.debug}
    parser_instance = PDFParser(config)
    
    # 解析PDF
    output_path = args.output or f"output/parsed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    try:
        records = parser_instance.parse_pdf(args.pdf_path, output_path)
        print(f"解析完成！共提取 {len(records)} 条记录")
        print(f"输出文件: {output_path}")
    except Exception as e:
        print(f"解析失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
