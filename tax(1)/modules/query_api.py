"""
查询API模块
提供税则查询接口，返回有税率信息的条目
支持多种查询方式和过滤条件
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime

import pandas as pd
import yaml
from modules.vector_database import VectorDatabase

logger = logging.getLogger(__name__)

def _get_default_model_path() -> str:
    """从 config.yaml 读取默认模型路径"""
    default_path = r"D:\Program Files\关税优化\BAAI_bge-large-en"
    try:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                model_path = config.get('model', {}).get('embedding_model', default_path)
                return model_path
    except Exception as e:
        logger.warning(f"无法读取配置文件，使用默认路径: {e}")
    return default_path

class TaxQueryAPI:
    """
    税则查询API
    提供统一的查询接口，支持语义搜索和过滤
    """
    
    def __init__(self, 
                 vector_db_path: str = "output/vector_db.pkl",
                 csv_data_path: str = "output/optimized_hierarchical_data.csv",
                 model_path: str = None):
        """
        初始化查询API
        
        Args:
            vector_db_path: 向量数据库路径
            csv_data_path: CSV数据文件路径
            model_path: BGE模型路径（如果为None，则从config.yaml读取）
        """
        if model_path is None:
            model_path = _get_default_model_path()
        self.vector_db_path = vector_db_path
        self.csv_data_path = csv_data_path
        self.model_path = model_path
        
        # 组件
        self.vector_db = None
        self.csv_data = None
        
        # 状态
        self.is_initialized = False
        
        # 查询扩展同义词映射（用于改善搜索准确性）
        self.query_expansion_map = {
            'laptop': 'laptop portable automatic data processing computer notebook',
            'notebook': 'notebook portable automatic data processing computer laptop',
            'computer': 'computer automatic data processing machine',
            'pc': 'pc personal computer automatic data processing',
            'desktop': 'desktop computer automatic data processing',
            'tablet': 'tablet portable automatic data processing',
            'smartphone': 'smartphone mobile phone telephone',
            'phone': 'phone mobile telephone smartphone',
            'fish': 'fish seafood',
            'meat': 'meat beef pork chicken',
            'car': 'car automobile vehicle motor',
            'vehicle': 'vehicle car automobile motor',
        }
        
        logger.info("税则查询API初始化")
    
    def initialize(self) -> bool:
        """初始化API组件"""
        try:
            logger.info("开始初始化查询API")
            
            # 初始化向量数据库
            self.vector_db = VectorDatabase(model_path=self.model_path)
            
            if os.path.exists(self.vector_db_path):
                if not self.vector_db.load(self.vector_db_path):
                    logger.error("向量数据库加载失败")
                    return False
                logger.info("向量数据库加载成功")
            else:
                logger.warning(f"向量数据库文件不存在: {self.vector_db_path}")
                return False
            
            # 加载CSV数据（用于获取完整税率信息）
            if os.path.exists(self.csv_data_path):
                # 将税则号列读取为字符串类型，避免被转换为浮点数
                self.csv_data = pd.read_csv(self.csv_data_path, dtype={'tariff_code': str})
                # 确保税则号列是字符串类型
                self.csv_data['tariff_code'] = self.csv_data['tariff_code'].astype(str)
                logger.info(f"CSV数据加载成功: {len(self.csv_data)} 条记录")
            else:
                logger.warning(f"CSV数据文件不存在: {self.csv_data_path}")
                return False
            
            self.is_initialized = True
            logger.info("查询API初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"初始化失败: {e}")
            return False
    
    def search(self, 
               query: str,
               top_k: int = 10,
               only_with_tax_info: bool = True,
               hierarchy_level: Optional[int] = None,
               tariff_code_filter: Optional[str] = None,
               tax_rate_range: Optional[Tuple[float, float]] = None) -> List[Dict[str, Any]]:
        """
        搜索税则条目
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            only_with_tax_info: 只返回有税率信息的条目
            hierarchy_level: 层级过滤
            tariff_code_filter: 税则号过滤
            tax_rate_range: 税率范围过滤 (min, max)
            
        Returns:
            搜索结果列表
        """
        if not self.is_initialized:
            logger.error("API未初始化")
            return []
        
        try:
            logger.info(f"搜索查询: '{query}'")
            
            # 查询扩展：如果查询词在扩展映射中，则扩展查询
            expanded_query = self._expand_query(query)
            if expanded_query != query:
                logger.info(f"查询已扩展: '{query}' -> '{expanded_query}'")
            
            # 执行向量搜索（不使用税率过滤，因为向量数据库可能没有has_tax_info字段）
            results = self.vector_db.search(
                query=expanded_query,
                top_k=top_k * 3,  # 获取更多结果用于后续过滤
                filter_tax_info=None,  # 不在向量搜索阶段过滤税率
                filter_hierarchy_level=hierarchy_level,
                filter_tariff_code=tariff_code_filter
            )
            
            # 先增强结果信息（从CSV填充税率信息），这样才能正确过滤
            enhanced_results = self._enhance_results(results)
            
            # 应用税率信息过滤（在增强之后，此时税率信息已填充）
            if only_with_tax_info:
                enhanced_results = self._filter_by_tax_info(enhanced_results)
            
            # 应用税率范围过滤
            if tax_rate_range:
                enhanced_results = self._filter_by_tax_rate(enhanced_results, tax_rate_range)
            
            # 添加关键词匹配加分并重排序
            enhanced_results = self._rerank_with_keyword_match(enhanced_results, query)
            
            # 限制结果数量
            enhanced_results = enhanced_results[:top_k]
            
            logger.info(f"搜索完成 - 返回 {len(enhanced_results)} 条结果")
            return enhanced_results
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []
    
    def search_by_tariff_code(self, tariff_code: str) -> List[Dict[str, Any]]:
        """
        根据税则号搜索
        
        Args:
            tariff_code: 税则号
            
        Returns:
            匹配的条目列表
        """
        if not self.is_initialized:
            logger.error("API未初始化")
            return []
        
        try:
            # 从CSV数据中查找
            matching_rows = self.csv_data[self.csv_data['tariff_code'] == tariff_code]
            
            if matching_rows.empty:
                logger.info(f"未找到税则号: {tariff_code}")
                return []
            
            results = []
            for _, row in matching_rows.iterrows():
                result = {
                    'tariff_code': row.get('tariff_code', ''),
                    'description': row.get('description', ''),
                    'import_duty': row.get('import_duty', ''),
                    'import_excise': row.get('import_excise', ''),
                    'import_vagst': row.get('import_vagst', ''),
                    'export_duty': row.get('export_duty', ''),
                    'unit': row.get('unit', ''),
                    'sitc_code': row.get('sitc_code', ''),
                    'hierarchy_level': row.get('hierarchy_level', 0),
                    'level_name': row.get('level_name', ''),
                    'has_tax_info': row.get('has_tax_info', False),
                    'score': 1.0,  # 精确匹配
                    'search_method': 'tariff_code'
                }
                results.append(result)
            
            # 使用 _enhance_results 增强结果，包括构建层级路径
            enhanced_results = self._enhance_results(results)
            
            logger.info(f"税则号搜索完成 - 找到 {len(enhanced_results)} 条结果")
            return enhanced_results
            
        except Exception as e:
            logger.error(f"税则号搜索失败: {e}")
            return []
    
    def get_hierarchy_info(self, hierarchy_level: int) -> Dict[str, Any]:
        """
        获取层级信息
        
        Args:
            hierarchy_level: 层级级别
            
        Returns:
            层级统计信息
        """
        if not self.is_initialized:
            logger.error("API未初始化")
            return {}
        
        try:
            # 过滤指定层级的数据
            level_data = self.csv_data[self.csv_data['hierarchy_level'] == hierarchy_level]
            
            # 统计信息
            stats = {
                'level': hierarchy_level,
                'total_count': len(level_data),
                'with_tax_info': len(level_data[level_data['has_tax_info'] == True]),
                'sample_entries': []
            }
            
            # 获取示例条目
            sample_size = min(5, len(level_data))
            if sample_size > 0:
                sample_data = level_data.sample(n=sample_size)
                for _, row in sample_data.iterrows():
                    stats['sample_entries'].append({
                        'tariff_code': row.get('tariff_code', ''),
                        'description': row.get('description', ''),
                        'has_tax_info': row.get('has_tax_info', False)
                    })
            
            logger.info(f"层级 {hierarchy_level} 统计信息获取完成")
            return stats
            
        except Exception as e:
            logger.error(f"获取层级信息失败: {e}")
            return {}
    
    def get_tax_statistics(self) -> Dict[str, Any]:
        """获取税率统计信息"""
        if not self.is_initialized:
            logger.error("API未初始化")
            return {}
        
        try:
            # 基本统计
            total_entries = len(self.csv_data)
            with_tax_info = len(self.csv_data[self.csv_data['has_tax_info'] == True])
            
            # 层级统计
            level_stats = {}
            for level in [1, 2, 3, 4]:
                level_data = self.csv_data[self.csv_data['hierarchy_level'] == level]
                level_stats[f'level_{level}'] = {
                    'total': len(level_data),
                    'with_tax_info': len(level_data[level_data['has_tax_info'] == True])
                }
            
            # 税率分布
            duty_rates = self.csv_data['import_duty'].dropna()
            duty_distribution = duty_rates.value_counts().head(10).to_dict()
            
            stats = {
                'total_entries': total_entries,
                'with_tax_info': with_tax_info,
                'tax_info_coverage': f"{(with_tax_info / total_entries * 100):.1f}%" if total_entries > 0 else "0%",
                'level_statistics': level_stats,
                'duty_distribution': duty_distribution,
                'last_updated': datetime.now().isoformat()
            }
            
            logger.info("税率统计信息获取完成")
            return stats
            
        except Exception as e:
            logger.error(f"获取税率统计失败: {e}")
            return {}
    
    def _filter_by_tax_rate(self, results: List[Dict], tax_rate_range: Tuple[float, float]) -> List[Dict]:
        """根据税率范围过滤结果"""
        filtered_results = []
        
        logger.info(f"税率过滤范围: {tax_rate_range[0]}% - {tax_rate_range[1]}%")
        logger.info(f"原始结果数量: {len(results)}")
        
        for result in results:
            # 提取税率信息
            import_duty = result.get('import_duty', '')
            
            # 处理NaN值 - 如果用户要求税率过滤，跳过没有税率信息的条目
            if pd.isna(import_duty) or import_duty == '' or str(import_duty).lower() == 'nan':
                logger.debug(f"跳过NaN税率条目: {result.get('tariff_code', 'Unknown')}")
                continue
            
            # 处理免税情况
            if str(import_duty).lower() == 'free':
                # 如果范围包含0，则包含免税商品
                if tax_rate_range[0] <= 0 <= tax_rate_range[1]:
                    filtered_results.append(result)
                    logger.debug(f"包含免税条目: {result.get('tariff_code', 'Unknown')}")
                else:
                    logger.debug(f"跳过免税条目（范围不包含0）: {result.get('tariff_code', 'Unknown')}")
                continue
            
            try:
                # 解析税率
                if '%' in str(import_duty):
                    rate = float(str(import_duty).replace('%', ''))
                else:
                    rate = float(import_duty)
                
                # 检查是否在范围内
                if tax_rate_range[0] <= rate <= tax_rate_range[1]:
                    filtered_results.append(result)
                    logger.debug(f"包含税率条目: {result.get('tariff_code', 'Unknown')} - {rate}%")
                else:
                    logger.debug(f"跳过税率条目（超出范围）: {result.get('tariff_code', 'Unknown')} - {rate}%")
                    
            except (ValueError, TypeError) as e:
                # 如果无法解析税率，跳过该条目
                logger.debug(f"无法解析税率: {import_duty} - {e}")
                continue
        
        logger.info(f"税率过滤后结果数量: {len(filtered_results)}")
        return filtered_results
    
    def _filter_by_tax_info(self, results: List[Dict]) -> List[Dict]:
        """过滤有税率信息的条目"""
        filtered_results = []
        
        logger.info(f"过滤有税率信息的条目，原始结果数: {len(results)}")
        
        for result in results:
            import_duty = result.get('import_duty', '')
            
            # 检查是否有有效的税率信息
            has_valid_tax = False
            
            # 检查进口关税
            if not pd.isna(import_duty) and import_duty != '' and str(import_duty).lower() != 'nan':
                has_valid_tax = True
                logger.debug(f"找到进口关税: {result.get('tariff_code', 'Unknown')} - {import_duty}")
            
            # 检查其他税率字段
            for tax_field in ['import_excise', 'import_vagst', 'export_duty']:
                tax_value = result.get(tax_field, '')
                if not pd.isna(tax_value) and tax_value != '' and str(tax_value).lower() != 'nan':
                    has_valid_tax = True
                    logger.debug(f"找到{tax_field}: {result.get('tariff_code', 'Unknown')} - {tax_value}")
                    break
            
            if has_valid_tax:
                filtered_results.append(result)
                logger.debug(f"包含有税率信息的条目: {result.get('tariff_code', 'Unknown')}")
            else:
                logger.debug(f"跳过无税率信息的条目: {result.get('tariff_code', 'Unknown')} - 进口关税: {import_duty}")
        
        logger.info(f"税率信息过滤后结果数: {len(filtered_results)}")
        return filtered_results
    
    def _build_full_hierarchy_path(self, tariff_code: str) -> str:
        """
        构建完整的层级路径
        
        Args:
            tariff_code: 税则号
            
        Returns:
            完整的层级路径字符串，格式：父级1 > 父级2 > 当前级
        """
        if not tariff_code:
            return ''
        
        try:
            tariff_code = str(tariff_code).strip()
            path_parts = []
            
            # 获取当前层级的描述
            current_match = self.csv_data[self.csv_data['tariff_code'] == tariff_code]
            if current_match.empty:
                # 尝试标准化匹配
                normalized_code = tariff_code.replace('.', '').replace(',', '').lstrip('0')
                if normalized_code:
                    csv_normalized = self.csv_data['tariff_code'].astype(str).str.replace('.', '', regex=False).str.replace(',', '', regex=False).str.lstrip('0')
                    matching_indices = csv_normalized == normalized_code
                    if matching_indices.any():
                        current_match = self.csv_data[matching_indices]
            
            if current_match.empty:
                return ''
            
            current_desc = current_match.iloc[0].get('description', '')
            if not current_desc or pd.isna(current_desc):
                current_desc = ''
            
            # 获取CSV中实际的税则号格式（用于确定正确的格式）
            actual_code_in_csv = str(current_match.iloc[0].get('tariff_code', tariff_code)).strip()
            current_hierarchy_level = current_match.iloc[0].get('hierarchy_level', 0)
            
            # 移除点和逗号以便分析
            code_clean = actual_code_in_csv.replace('.', '').replace(',', '')
            
            # 如果是二级或三级分类，查找父级（一级分类）
            # 二级分类的格式可能是：XXXX.XX（如 0804.20）或 XXX.XX（如 804.2）
            # 三级分类的格式可能是：XXXX.XXXX（如 0804.2000）
            # 父级格式是：XX.XX（如 08.04）
            if current_hierarchy_level in [2, 3] or (len(code_clean) >= 4 and '.' in actual_code_in_csv and ',' not in actual_code_in_csv):
                # 提取前4位数字，转换为 XX.XX 格式
                # 例如：0804 -> 08.04, 8042 -> 08.04（如果前导零被移除）
                if len(code_clean) >= 4:
                    # 取前4位
                    first_four = code_clean[:4]
                    # 转换为 XX.XX 格式（前2位.后2位）
                    parent_code = f"{first_four[:2]}.{first_four[2:4]}"
                    parent_match = self.csv_data[self.csv_data['tariff_code'] == parent_code]
                    
                    if parent_match.empty:
                        # 尝试标准化匹配（移除前导零）
                        normalized = parent_code.replace('.', '').lstrip('0')
                        if normalized:
                            csv_normalized = self.csv_data['tariff_code'].astype(str).str.replace('.', '', regex=False).str.replace(',', '', regex=False).str.lstrip('0')
                            matching_indices = csv_normalized == normalized
                            if matching_indices.any():
                                parent_match = self.csv_data[matching_indices]
                                # 如果通过标准化匹配找到了，使用CSV中的实际格式
                                if not parent_match.empty:
                                    parent_code = str(parent_match.iloc[0].get('tariff_code', parent_code)).strip()
                    
                    if not parent_match.empty:
                        parent_desc = parent_match.iloc[0].get('description', '')
                        if parent_desc and not pd.isna(parent_desc):
                            path_parts.append(parent_desc)
            
            # 如果是三级分类 (格式：XXXX.XXXX，长度为8位数字，或 XXXX.XX，长度为6位数字)
            # 注意：这里处理的是有二级父级的情况，如果二级父级不存在，上面已经处理了一级父级
            if (len(code_clean) == 6 or len(code_clean) == 8) and '.' in actual_code_in_csv and ',' not in actual_code_in_csv:
                # 父级是二级分类：前2位 + '.' + 后2位
                # 注意：保持前导零，例如 "070959" -> "07.09"
                parent_code = f"{code_clean[:2]}.{code_clean[2:4]}"
                parent_match = self.csv_data[self.csv_data['tariff_code'] == parent_code]
                
                if parent_match.empty:
                    # 尝试标准化匹配（移除前导零）
                    normalized = parent_code.replace('.', '').lstrip('0')
                    if normalized:
                        csv_normalized = self.csv_data['tariff_code'].astype(str).str.replace('.', '', regex=False).str.lstrip('0')
                        matching_indices = csv_normalized == normalized
                        if matching_indices.any():
                            parent_match = self.csv_data[matching_indices]
                            # 如果通过标准化匹配找到了，使用CSV中的实际格式
                            if not parent_match.empty:
                                parent_code = str(parent_match.iloc[0].get('tariff_code', parent_code)).strip()
                
                if not parent_match.empty:
                    parent_desc = parent_match.iloc[0].get('description', '')
                    # 避免重复添加相同的父级描述
                    if parent_desc and not pd.isna(parent_desc) and parent_desc not in path_parts:
                        path_parts.append(parent_desc)
        
            # 如果是四级分类 (格式：XXXX.XX,XX，包含逗号)
            elif ',' in actual_code_in_csv:
                # 父级是三级分类：去掉逗号后的部分
                parent_code = actual_code_in_csv.split(',')[0]
                parent_match = self.csv_data[self.csv_data['tariff_code'] == parent_code]
                
                if parent_match.empty:
                    # 尝试标准化匹配
                    normalized = parent_code.replace('.', '').lstrip('0')
                    if normalized:
                        csv_normalized = self.csv_data['tariff_code'].astype(str).str.replace('.', '', regex=False).str.lstrip('0')
                        matching_indices = csv_normalized == normalized
                        if matching_indices.any():
                            parent_match = self.csv_data[matching_indices]
                            # 如果通过标准化匹配找到了，使用CSV中的实际格式
                            if not parent_match.empty:
                                parent_code = str(parent_match.iloc[0].get('tariff_code', parent_code)).strip()
                
                if not parent_match.empty:
                    parent_desc = parent_match.iloc[0].get('description', '')
                    if parent_desc and not pd.isna(parent_desc):
                        path_parts.append(parent_desc)
                    
                    # 继续查找二级父级
                    parent_clean = parent_code.replace('.', '')
                    if len(parent_clean) == 6:
                        level2_code = f"{parent_clean[:2]}.{parent_clean[2:4]}"
                        level2_match = self.csv_data[self.csv_data['tariff_code'] == level2_code]
                        
                        if level2_match.empty:
                            normalized = level2_code.replace('.', '').lstrip('0')
                            if normalized:
                                csv_normalized = self.csv_data['tariff_code'].astype(str).str.replace('.', '', regex=False).str.lstrip('0')
                                matching_indices = csv_normalized == normalized
                                if matching_indices.any():
                                    level2_match = self.csv_data[matching_indices]
                                    # 如果通过标准化匹配找到了，使用CSV中的实际格式
                                    if not level2_match.empty:
                                        level2_code = str(level2_match.iloc[0].get('tariff_code', level2_code)).strip()
                        
                        if not level2_match.empty:
                            level2_desc = level2_match.iloc[0].get('description', '')
                            if level2_desc and not pd.isna(level2_desc):
                                # 避免重复添加
                                if not path_parts or level2_desc not in path_parts[0]:
                                    path_parts.insert(0, level2_desc)
        
            # 添加当前层级的描述
            if current_desc:
                path_parts.append(current_desc)
            
            # 组合成完整路径
            if path_parts:
                return ' > '.join(path_parts)
            else:
                return current_desc if current_desc else ''
                
        except Exception as e:
            logger.warning(f"构建层级路径失败 (税则号: {tariff_code}): {e}")
            return ''
    
    def _enhance_results(self, results: List[Dict]) -> List[Dict]:
        """增强搜索结果信息"""
        enhanced_results = []
        
        for result in results:
            try:
                # 添加搜索方法标识
                result['search_method'] = 'semantic_search'
                
                # 添加完整税率信息
                if result.get('tariff_code'):
                    try:
                        # 标准化税则号格式以便匹配（移除前导零，统一格式）
                        tariff_code = str(result['tariff_code']).strip()
                        if not tariff_code:
                            enhanced_results.append(result)
                            continue
                        
                        # 首先尝试精确匹配
                        csv_match = self.csv_data[self.csv_data['tariff_code'] == tariff_code]
                        
                        # 如果精确匹配失败，尝试标准化格式匹配
                        if csv_match.empty:
                            # 标准化税则号：移除前导零，统一格式
                            normalized_code = tariff_code.replace('.', '').replace(',', '').lstrip('0')
                            if normalized_code:
                                # 尝试匹配标准化后的格式（使用更高效的方法）
                                try:
                                    # 先尝试使用pandas的字符串操作
                                    csv_normalized = self.csv_data['tariff_code'].astype(str).str.replace('.', '', regex=False).str.replace(',', '', regex=False).str.lstrip('0')
                                    # 首先尝试精确匹配
                                    matching_indices = csv_normalized == normalized_code
                                    if matching_indices.any():
                                        csv_match = self.csv_data[matching_indices]
                                    else:
                                        # 如果精确匹配失败，尝试前缀匹配（处理709.59匹配0709.5900的情况）
                                        # 例如：70959 应该匹配 7095900（如果70959是7095900的前缀）
                                        # 优先匹配：1) 长度最接近的 2) 层级更高的（更具体的）
                                        prefix_match_indices = csv_normalized.str.startswith(normalized_code)
                                        if not prefix_match_indices.any():
                                            # 如果输入是前缀，尝试反向匹配（输入是CSV条目的前缀）
                                            prefix_match_indices = normalized_code.str.startswith(csv_normalized) if hasattr(normalized_code, 'str') else pd.Series([normalized_code.startswith(str(c)) for c in csv_normalized], index=csv_normalized.index)
                                        
                                        if prefix_match_indices.any() if hasattr(prefix_match_indices, 'any') else any(prefix_match_indices):
                                            # 如果有多个匹配，选择最接近的（长度最接近的，层级更高的）
                                            matches = self.csv_data[prefix_match_indices] if hasattr(prefix_match_indices, '__getitem__') else self.csv_data[[i for i, v in enumerate(prefix_match_indices) if v]]
                                            if len(matches) > 0:
                                                # 计算每个匹配的优先级：1) 长度差最小 2) 层级最高（数字最大）
                                                matches_with_priority = []
                                                for idx, row in matches.iterrows():
                                                    csv_code_normalized = str(row.get('tariff_code', '')).replace('.', '').replace(',', '').lstrip('0')
                                                    length_diff = abs(len(csv_code_normalized) - len(normalized_code))
                                                    hierarchy_level = row.get('hierarchy_level', 0)
                                                    # 优先级：长度差越小越好，层级越高越好
                                                    # 使用负数层级，这样层级高的（数字大）优先级更高
                                                    priority = (length_diff, -hierarchy_level)
                                                    matches_with_priority.append((priority, idx, row))
                                                # 按优先级排序
                                                matches_with_priority.sort(key=lambda x: x[0])
                                                best_match_idx = matches_with_priority[0][1]
                                                csv_match = self.csv_data[self.csv_data.index == best_match_idx]
                                except Exception as e:
                                    logger.debug(f"标准化匹配失败，使用迭代方法: {e}")
                                    # 回退到迭代方法
                                    for _, row in self.csv_data.iterrows():
                                        try:
                                            csv_code = str(row.get('tariff_code', '')).replace('.', '').replace(',', '').lstrip('0')
                                            if csv_code == normalized_code or csv_code.startswith(normalized_code) or normalized_code.startswith(csv_code):
                                                csv_match = self.csv_data[self.csv_data.index == row.name]
                                                break
                                        except Exception:
                                            continue
                        
                        if not csv_match.empty:
                            row = csv_match.iloc[0]
                            # 使用CSV中实际的税则号（可能有前导零）来构建层级路径
                            actual_tariff_code = str(row.get('tariff_code', tariff_code)).strip()
                            # 构建完整的层级路径（包含所有父级层级）
                            full_hierarchy_path = self._build_full_hierarchy_path(actual_tariff_code)
                            
                            # 如果构建了完整路径，使用完整路径；否则使用CSV中的描述
                            if full_hierarchy_path:
                                result['description'] = full_hierarchy_path
                                result['full_hierarchy_path'] = full_hierarchy_path
                            else:
                                csv_description = row.get('description', '')
                                if csv_description and isinstance(csv_description, str) and csv_description.strip():
                                    result['description'] = csv_description
                            
                            result.update({
                                'import_duty': row.get('import_duty', ''),
                                'import_excise': row.get('import_excise', ''),
                                'import_vagst': row.get('import_vagst', ''),
                                'export_duty': row.get('export_duty', ''),
                                'unit': row.get('unit', ''),
                                'sitc_code': row.get('sitc_code', ''),
                                'hierarchy_level': row.get('hierarchy_level', 0),
                                'level_name': row.get('level_name', ''),
                                'has_tax_info': row.get('has_tax_info', False)
                            })
                    except Exception as e:
                        # 如果匹配过程出错，记录日志但继续处理，保留原始结果
                        logger.warning(f"增强结果时出错 (税则号: {result.get('tariff_code', 'N/A')}): {e}")
                
                enhanced_results.append(result)
            except Exception as e:
                # 如果处理单个结果时出错，记录日志但继续处理其他结果
                logger.error(f"处理搜索结果时出错: {e}")
                # 仍然添加原始结果，避免丢失数据
                enhanced_results.append(result)
        
        return enhanced_results
    
    def _expand_query(self, query: str) -> str:
        """
        扩展查询词，添加同义词和相关词
        
        Args:
            query: 原始查询文本
            
        Returns:
            扩展后的查询文本
        """
        if not query:
            return query
        
        query_lower = query.lower().strip()
        
        # 检查是否有完全匹配的扩展映射
        if query_lower in self.query_expansion_map:
            return self.query_expansion_map[query_lower]
        
        # 检查查询词是否包含映射中的关键词
        expanded_terms = []
        original_terms = query_lower.split()
        
        for term in original_terms:
            if term in self.query_expansion_map:
                expanded_terms.append(self.query_expansion_map[term])
            else:
                expanded_terms.append(term)
        
        # 如果进行了扩展，合并所有扩展词
        if any(term in self.query_expansion_map for term in original_terms):
            # 合并所有扩展词和原始词
            all_terms = set(original_terms)
            for term in original_terms:
                if term in self.query_expansion_map:
                    expanded = self.query_expansion_map[term].split()
                    all_terms.update(expanded)
            return ' '.join(all_terms)
        
        return query
    
    def _rerank_with_keyword_match(self, results: List[Dict], query: str) -> List[Dict]:
        """
        使用关键词匹配对结果进行重排序
        
        Args:
            results: 搜索结果列表
            query: 查询文本（原始查询，用于关键词匹配）
            
        Returns:
            重排序后的结果列表
        """
        if not results or not query:
            return results
        
        # 提取查询关键词（转换为小写，去除标点）
        import re
        query_lower = query.lower().strip()
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        
        # 如果查询词在扩展映射中，也添加扩展后的同义词
        if query_lower in self.query_expansion_map:
            expanded = self.query_expansion_map[query_lower]
            expanded_words = set(re.findall(r'\b\w+\b', expanded.lower()))
            query_words.update(expanded_words)
        
        # 如果查询词太短（少于2个字符），不进行关键词匹配
        if not query_words or all(len(w) < 2 for w in query_words):
            return results
        
        # 为每个结果计算关键词匹配分数
        reranked_results = []
        for result in results:
            # 获取描述文本（优先使用full_hierarchy_path，否则使用description）
            desc_text = result.get('full_hierarchy_path', result.get('description', ''))
            if not desc_text:
                desc_text = str(result.get('description', ''))
            
            desc_lower = desc_text.lower()
            
            # 计算关键词匹配分数
            keyword_score = 0.0
            matched_words = []
            
            for word in query_words:
                if len(word) >= 2:  # 只匹配长度>=2的词
                    # 完全匹配加分更多
                    if f' {word} ' in f' {desc_lower} ' or desc_lower.startswith(word + ' ') or desc_lower.endswith(' ' + word):
                        keyword_score += 1.0
                        matched_words.append(word)
                    # 部分匹配（包含该词）加分较少
                    elif word in desc_lower:
                        keyword_score += 0.5
                        matched_words.append(word)
            
            # 归一化关键词分数（基于查询词数量）
            if query_words:
                keyword_score = keyword_score / len(query_words)
            
            # 获取原始向量相似度分数
            vector_score = result.get('score', 0.0)
            
            # 混合分数：优先关键词匹配
            # 如果有关键词匹配，显著提升排名
            if keyword_score > 0:
                # 有关键词匹配的结果，大幅提升排名
                # 使用更高的权重，确保包含关键词的结果排在前面
                final_score = vector_score * 0.3 + keyword_score * 0.7 + 0.2  # 额外加0.2分
            else:
                # 没有关键词匹配的结果，降低排名
                final_score = vector_score * 0.7
            
            # 更新结果
            result['keyword_score'] = keyword_score
            result['matched_keywords'] = matched_words
            result['final_score'] = final_score
            result['original_score'] = vector_score
            
            reranked_results.append(result)
        
        # 按最终分数降序排序
        reranked_results.sort(key=lambda x: x.get('final_score', 0.0), reverse=True)
        
        # 更新score字段为final_score，以便UI显示
        for result in reranked_results:
            result['score'] = result.get('final_score', result.get('score', 0.0))
        
        logger.debug(f"关键词重排序完成 - 查询: '{query}', 结果数: {len(reranked_results)}")
        return reranked_results
    
    def get_database_info(self) -> Dict[str, Any]:
        """获取数据库信息"""
        if not self.is_initialized:
            return {'status': 'not_initialized'}
        
        try:
            vector_stats = self.vector_db.get_statistics()
            
            info = {
                'status': 'initialized',
                'vector_database': vector_stats,
                'csv_data': {
                    'total_records': len(self.csv_data),
                    'file_path': self.csv_data_path
                },
                'last_check': datetime.now().isoformat()
            }
            
            return info
            
        except Exception as e:
            logger.error(f"获取数据库信息失败: {e}")
            return {'status': 'error', 'message': str(e)}

def main():
    """主函数，用于测试查询API"""
    import argparse
    
    parser = argparse.ArgumentParser(description='税则查询API')
    parser.add_argument('action', choices=['search', 'tariff', 'stats', 'info'], help='操作类型')
    parser.add_argument('-q', '--query', help='查询文本')
    parser.add_argument('-t', '--tariff', help='税则号')
    parser.add_argument('-k', '--top_k', type=int, default=10, help='返回结果数量')
    parser.add_argument('-l', '--level', type=int, help='层级过滤')
    parser.add_argument('--tax-only', action='store_true', help='只返回有税率信息的条目')
    parser.add_argument('-v', '--vector_db', default='output/vector_db.pkl', help='向量数据库路径')
    parser.add_argument('-c', '--csv', default='output/optimized_hierarchical_data.csv', help='CSV数据路径')
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建查询API
    api = TaxQueryAPI(
        vector_db_path=args.vector_db,
        csv_data_path=args.csv
    )
    
    # 初始化
    if not api.initialize():
        print("API初始化失败")
        return 1
    
    if args.action == 'search':
        if not args.query:
            print("错误: 需要指定查询文本")
            return 1
        
        results = api.search(
            query=args.query,
            top_k=args.top_k,
            only_with_tax_info=args.tax_only,
            hierarchy_level=args.level
        )
        
        print(f"搜索结果 (查询: '{args.query}'):")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['description'][:50]}...")
            print(f"     税则号: {result.get('tariff_code', 'N/A')}")
            print(f"     税率: 进口关税={result.get('import_duty', 'N/A')}, 消费税={result.get('import_excise', 'N/A')}")
            print(f"     相似度: {result.get('score', 0):.4f}")
            print()
    
    elif args.action == 'tariff':
        if not args.tariff:
            print("错误: 需要指定税则号")
            return 1
        
        results = api.search_by_tariff_code(args.tariff)
        
        if results:
            print(f"税则号搜索结果 (税则号: {args.tariff}):")
            for result in results:
                print(f"  描述: {result['description']}")
                print(f"  税率: 进口关税={result.get('import_duty', 'N/A')}, 消费税={result.get('import_excise', 'N/A')}")
                print(f"  层级: {result.get('hierarchy_level', 'N/A')}")
        else:
            print(f"未找到税则号: {args.tariff}")
    
    elif args.action == 'stats':
        stats = api.get_tax_statistics()
        print("税率统计信息:")
        print(json.dumps(stats, indent=2, ensure_ascii=False))
    
    elif args.action == 'info':
        info = api.get_database_info()
        print("数据库信息:")
        print(json.dumps(info, indent=2, ensure_ascii=False))
    
    return 0

if __name__ == "__main__":
    exit(main())
