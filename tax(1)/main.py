"""
税则查询系统主程序
整合PDF解析、向量数据库和查询API三个模块
"""

import os
import sys
import logging
import argparse
from typing import Optional, List, Dict, Any
from datetime import datetime
import yaml

# 添加模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

from modules.pdf_parser import PDFParser
from modules.vector_database import VectorDatabase
from modules.query_api import TaxQueryAPI

logger = logging.getLogger(__name__)

def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """加载配置文件"""
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}

class TaxSystem:
    """
    税则查询系统主类
    整合所有功能模块
    """
    
    def __init__(self, config: Dict[str, Any] = None, config_path: str = 'config.yaml'):
        """
        初始化系统
        
        Args:
            config: 配置字典（优先级高于配置文件）
            config_path: 配置文件路径
        """
        # 先加载配置文件
        file_config = load_config(config_path)
        
        # 合并配置（传入的config优先级更高）
        if config:
            file_config.update(config)
        
        self.config = file_config
        self.setup_logging()
        
        # 从配置文件读取模型路径
        model_config = self.config.get('model', {})
        self.model_path = model_config.get('embedding_model', r"D:/D:\Program Files\关税优化\BAAI_bge-large-en")
        
        # 模块实例
        self.pdf_parser = None
        self.vector_db = None
        self.query_api = None
        
        # 路径配置
        self.output_dir = self.config.get('output_dir', 'output')
        self.vector_db_path = os.path.join(self.output_dir, 'vector_db.pkl')
        self.csv_data_path = os.path.join(self.output_dir, 'parsed_data.csv')
        
        logger.info("税则查询系统初始化完成")
    
    def setup_logging(self):
        """设置日志"""
        log_level = self.config.get('log_level', 'INFO')
        verbose = self.config.get('verbose', False)
        
        if verbose:
            log_level = 'DEBUG'
        
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('tax_system.log', encoding='utf-8')
            ]
        )
    
    def parse_pdf(self, pdf_path: str, output_csv: str = None) -> bool:
        """
        解析PDF文件
        
        Args:
            pdf_path: PDF文件路径
            output_csv: 输出CSV文件路径
            
        Returns:
            是否成功解析
        """
        try:
            logger.info(f"开始解析PDF文件: {pdf_path}")
            
            # 创建PDF解析器
            self.pdf_parser = PDFParser(self.config)
            
            # 设置输出路径
            if not output_csv:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_csv = os.path.join(self.output_dir, f'parsed_data_{timestamp}.csv')
            
            # 解析PDF
            records = self.pdf_parser.parse_pdf(pdf_path, output_csv)
            
            if records:
                logger.info(f"PDF解析成功 - 提取 {len(records)} 条记录")
                self.csv_data_path = output_csv
                return True
            else:
                logger.error("PDF解析失败 - 未提取到记录")
                return False
                
        except Exception as e:
            logger.error(f"PDF解析失败: {e}")
            return False
    
    def build_vector_database(self, csv_path: str = None, force_rebuild: bool = False) -> bool:
        """
        构建向量数据库
        
        Args:
            csv_path: CSV文件路径
            force_rebuild: 是否强制重建
            
        Returns:
            是否成功构建
        """
        try:
            # 检查是否需要重建
            if not force_rebuild and os.path.exists(self.vector_db_path):
                logger.info("向量数据库已存在，跳过构建")
                return True
            
            # 确定CSV文件路径
            if not csv_path:
                csv_path = self.csv_data_path
            
            if not os.path.exists(csv_path):
                logger.error(f"CSV文件不存在: {csv_path}")
                return False
            
            logger.info(f"开始构建向量数据库: {csv_path}")
            
            # 创建向量数据库
            self.vector_db = VectorDatabase(
                model_path=self.model_path,
                use_gpu=self.config.get('use_gpu', True)
            )
            
            # 构建索引
            success = self.vector_db.build_from_csv(csv_path, self.vector_db_path)
            
            if success:
                logger.info("向量数据库构建成功")
                return True
            else:
                logger.error("向量数据库构建失败")
                return False
                
        except Exception as e:
            logger.error(f"构建向量数据库失败: {e}")
            return False
    
    def initialize_query_api(self, csv_path: str = None) -> bool:
        """
        初始化查询API
        
        Args:
            csv_path: CSV文件路径（可选）
            
        Returns:
            是否成功初始化
        """
        try:
            logger.info("初始化查询API")
            
            # 使用指定的CSV路径或默认路径
            csv_data_path = csv_path or self.csv_data_path
            
            # 创建查询API
            self.query_api = TaxQueryAPI(
                vector_db_path=self.vector_db_path,
                csv_data_path=csv_data_path,
                model_path=self.model_path
            )
            
            # 初始化
            success = self.query_api.initialize()
            
            if success:
                logger.info("查询API初始化成功")
                return True
            else:
                logger.error("查询API初始化失败")
                return False
                
        except Exception as e:
            logger.error(f"初始化查询API失败: {e}")
            return False
    
    def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        搜索税则条目
        
        Args:
            query: 查询文本
            **kwargs: 其他搜索参数
            
        Returns:
            搜索结果列表
        """
        if not self.query_api:
            logger.error("查询API未初始化")
            return []
        
        try:
            return self.query_api.search(query, **kwargs)
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'modules': {
                'pdf_parser': self.pdf_parser is not None,
                'vector_db': self.vector_db is not None,
                'query_api': self.query_api is not None
            },
            'files': {
                'vector_db_exists': os.path.exists(self.vector_db_path),
                'csv_data_exists': os.path.exists(self.csv_data_path)
            }
        }
        
        # 获取数据库信息
        if self.query_api:
            status['database_info'] = self.query_api.get_database_info()
        
        return status
    
    def run_full_pipeline(self, pdf_path: str) -> bool:
        """
        运行完整管道
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            是否成功运行
        """
        try:
            logger.info("开始运行完整管道")
            
            # 1. 解析PDF
            if not self.parse_pdf(pdf_path):
                logger.error("PDF解析失败")
                return False
            
            # 2. 构建向量数据库
            if not self.build_vector_database(force_rebuild=True):
                logger.error("向量数据库构建失败")
                return False
            
            # 3. 初始化查询API
            if not self.initialize_query_api():
                logger.error("查询API初始化失败")
                return False
            
            logger.info("完整管道运行成功")
            return True
            
        except Exception as e:
            logger.error(f"完整管道运行失败: {e}")
            return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='税则查询系统')
    parser.add_argument('action', choices=['parse', 'build', 'search', 'pipeline', 'status'], 
                       help='操作类型')
    parser.add_argument('-p', '--pdf', help='PDF文件路径')
    parser.add_argument('-c', '--csv', help='CSV文件路径')
    parser.add_argument('-q', '--query', help='查询文本')
    parser.add_argument('-k', '--top_k', type=int, default=10, help='返回结果数量')
    parser.add_argument('-l', '--level', type=int, help='层级过滤')
    parser.add_argument('--tax-only', action='store_true', help='只返回有税率信息的条目')
    parser.add_argument('--tax-rate', nargs=2, type=float, metavar=('MIN', 'MAX'), help='税率范围过滤 (最小值 最大值)')
    parser.add_argument('-o', '--output', default='output', help='输出目录')
    # 先加载默认配置
    default_config = load_config('config.yaml')
    default_model = default_config.get('model', {}).get('embedding_model', r"D:\Program Files\关税优化\BAAI_bge-large-en")
    parser.add_argument('-m', '--model', default=default_model, help='模型路径')
    parser.add_argument('-g', '--gpu', action='store_true', help='使用GPU')
    parser.add_argument('-v', '--verbose', action='store_true', help='详细输出')
    parser.add_argument('--force', action='store_true', help='强制重建')
    
    args = parser.parse_args()
    
    # 创建系统配置
    config = {
        'output_dir': args.output,
        'use_gpu': args.gpu,
        'verbose': args.verbose,
        'log_level': 'DEBUG' if args.verbose else 'INFO'
    }
    
    # 如果指定了模型路径，更新配置
    if args.model != default_model:
        config['model'] = {
            'embedding_model': args.model
        }
    
    # 创建系统实例
    system = TaxSystem(config)
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    if args.action == 'parse':
        if not args.pdf:
            print("错误: 需要指定PDF文件路径")
            return 1
        
        success = system.parse_pdf(args.pdf, args.csv)
        if success:
            print("PDF解析成功")
        else:
            print("PDF解析失败")
            return 1
    
    elif args.action == 'build':
        if not args.csv:
            print("错误: 需要指定CSV文件路径")
            return 1
        
        success = system.build_vector_database(args.csv, args.force)
        if success:
            print("向量数据库构建成功")
        else:
            print("向量数据库构建失败")
            return 1
    
    elif args.action == 'search':
        if not args.query:
            print("错误: 需要指定查询文本")
            return 1
        
        # 初始化查询API
        csv_path = args.csv if hasattr(args, 'csv') and args.csv else None
        if not system.initialize_query_api(csv_path):
            print("查询API初始化失败")
            return 1
        
        # 执行搜索 - 默认只返回有税率信息的记录
        search_kwargs = {
            'query': args.query,
            'top_k': args.top_k,
            'only_with_tax_info': True,  # 默认只返回有税率信息的记录
            'hierarchy_level': args.level
        }
        
        # 添加税率范围过滤
        if hasattr(args, 'tax_rate') and args.tax_rate:
            search_kwargs['tax_rate_range'] = tuple(args.tax_rate)
        
        results = system.search(**search_kwargs)
        
        print(f"\n🔍 搜索结果 (查询: '{args.query}')")
        print("=" * 80)
        
        if not results:
            print("❌ 未找到匹配的结果")
            return 0
        
        for i, result in enumerate(results, 1):
            print(f"\n📦 结果 {i}:")
            print(f"   🏷️  商品名称: {result.get('description', 'N/A')}")
            print(f"   🔢 税则号: {result.get('tariff_code', 'N/A')}")
            print(f"   📊 层级: {result.get('hierarchy_level', 'N/A')} - {result.get('level_name', 'N/A')}")
            print(f"   📏 单位: {result.get('unit', 'N/A')}")
            print(f"   🎯 相似度: {result.get('score', 0):.4f}")
            
            # 显示税率信息
            print(f"   💰 税率信息:")
            tax_info = []
            has_tax = False
            
            # 进口关税
            import_duty = result.get('import_duty', '')
            if import_duty and str(import_duty).lower() not in ['nan', '']:
                tax_info.append(f"     🏛️  进口关税: {import_duty}")
                has_tax = True
            
            # 进口消费税
            import_excise = result.get('import_excise', '')
            if import_excise and str(import_excise).lower() not in ['nan', '']:
                tax_info.append(f"     🍷 进口消费税: {import_excise}")
                has_tax = True
            
            # 进口增值税
            import_vagst = result.get('import_vagst', '')
            if import_vagst and str(import_vagst).lower() not in ['nan', '']:
                tax_info.append(f"     📈 进口增值税: {import_vagst}")
                has_tax = True
            
            # 出口关税
            export_duty = result.get('export_duty', '')
            if export_duty and str(export_duty).lower() not in ['nan', '']:
                tax_info.append(f"     🚢 出口关税: {export_duty}")
                has_tax = True
            
            if tax_info:
                for tax in tax_info:
                    print(tax)
            else:
                print("     ❌ 无税率信息")
            
            # 显示其他信息
            if result.get('sitc_code'):
                print(f"   📋 SITC编码: {result.get('sitc_code')}")
            
            print("   " + "-" * 60)
    
    elif args.action == 'pipeline':
        if not args.pdf:
            print("错误: 需要指定PDF文件路径")
            return 1
        
        success = system.run_full_pipeline(args.pdf)
        if success:
            print("完整管道运行成功")
        else:
            print("完整管道运行失败")
            return 1
    
    elif args.action == 'status':
        status = system.get_system_status()
        print("系统状态:")
        import json
        print(json.dumps(status, indent=2, ensure_ascii=False))
    
    return 0

if __name__ == "__main__":
    exit(main())